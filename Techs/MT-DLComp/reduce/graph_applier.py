import copy
import os

import onnx

from mutation.edge_node import EdgeNode, convert_onnx_to_edge
from mutation.fcb_mut import read_mut_info
from mutation.node_gen import NodeChainGen
from mutation.mutate_utils import insert_list
from utils.onnx_utils import get_max_node_idx, get_max_edge_idx
from reduce.edge_info import PlainEdge, SubsEdge, set_subs_place
from reduce.reduce_utils import make_model


def convert_edge_to_plain(edge: EdgeNode, name_info_mapping: dict):
    dep_edges_info = [name_info_mapping[i] for i in edge.def_node.input]
    edge_info = PlainEdge(edge.name, edge.shape, edge.def_node,
                          dep_edges_info, edge.zero)
    name_info_mapping.update({edge_info.name: edge_info})
    return edge_info


def convert_edge_to_subs(subs_ori_edge: EdgeNode, subs_new_edge: EdgeNode,
                         name_info_mapping: dict):
    add_node = subs_ori_edge.def_node
    add_in_name = [i for i in add_node.input if i != subs_new_edge.name][0]
    add_out_name = subs_ori_edge.name
    add_in_info = name_info_mapping[add_in_name]
    add_out_info = name_info_mapping[add_out_name]
    edge_info = SubsEdge(subs_new_edge.name, subs_new_edge.shape,
                         add_node, [add_in_info, add_out_info],
                         subs_new_edge.zero)
    name_info_mapping.update({edge_info.name: edge_info})
    return edge_info


def get_potential_subs_places(graph, subs_node, ori_node_name_set):
    pot_places = [subs_node.name]
    st_idx = [n.name for n in graph.node].index(pot_places[0]) + 2
    for node in graph.node[st_idx:]:
        pot_places.append(node.name)
        if node.name in ori_node_name_set:
            break
    return pot_places


def convert_ori_edges_to_info_dict(model):
    all_edge_node = convert_onnx_to_edge(model.graph)
    return {e.name: PlainEdge(e.name, e.shape, None, None, False, True)
            for e in all_edge_node}


def construct_deltas(model_dir, mut_info_dir, test_end):
    seed_model = onnx.load(os.path.join(model_dir, "seed.onnx"))

    name_info_mapping = convert_ori_edges_to_info_dict(seed_model)
    ori_nodes_name_set = set(n.name for n in seed_model.graph.node)

    delta_list = []
    for model_id in range(1, test_end + 1):
        model = onnx.load(os.path.join(model_dir, "%d.onnx" % model_id))
        dead_edges, subs_new_edge, subs_ori_edge = read_mut_info(
            os.path.join(mut_info_dir, "%d.txt" % model_id)
        )
        delta = Delta(dead_edges, subs_ori_edge, subs_new_edge,
                      name_info_mapping, model.graph, ori_nodes_name_set)
        delta_list.append(delta)

    return seed_model, delta_list, name_info_mapping


class Delta:
    def __init__(self, dead_edges, subs_ori_edge,
                 subs_new_edge, name_info_mapping, graph, ori_nodes_name_set):
        self.non_subs_edges = [convert_edge_to_plain(e, name_info_mapping)
                               for e in dead_edges]
        self.subs_edge = convert_edge_to_subs(subs_ori_edge, subs_new_edge,
                                              name_info_mapping)
        self.potential_ins_places = get_potential_subs_places(
            graph, subs_new_edge.def_node, ori_nodes_name_set
        )

    def reset(self):
        for edge in self.non_subs_edges:
            edge.reset()
        self.subs_edge.reset()

    def apply(self, graph_nodes, name_info_mapping, generator):
        ins_idx, subs_node = set_subs_place(
            self.subs_edge, self.potential_ins_places,
            graph_nodes, name_info_mapping, generator)

        new_nodes = []
        for edge in self.non_subs_edges:
            dead_nodes = edge.apply()
            new_nodes.extend(dead_nodes)

        subs_nodes = self.subs_edge.apply()

        new_nodes.extend(subs_nodes)

        insert_list(graph_nodes, new_nodes, ins_idx)


class GraphApplier:
    def __init__(self, model_dir, mut_info_dir, test_end):
        self.seed_model, self.delta_list, self.name_info_mapping = \
            construct_deltas(model_dir, mut_info_dir, test_end)
        final_model = onnx.load(os.path.join(model_dir, "%d.onnx" % test_end))
        self.max_node_idx = get_max_node_idx(final_model.graph)
        self.max_edge_idx = get_max_edge_idx(final_model.graph)

    def valid_range(self):
        return range(len(self.delta_list))

    def reset(self):
        for delta in self.delta_list:
            delta.reset()

    def apply(self, delta_ids):
        model = copy.copy(self.seed_model)
        self.reset()
        gen = NodeChainGen(self.max_node_idx+1, self.max_edge_idx+1)
        for delta_id in delta_ids:
            delta = self.delta_list[delta_id]
            delta.apply(model.graph.node, self.name_info_mapping, gen)

        make_model(model.graph.node, model)

        return model
