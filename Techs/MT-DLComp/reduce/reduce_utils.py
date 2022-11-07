import os
import shutil

import onnx
from onnx import shape_inference

import numpy as np

from collections import deque

from utils.onnx_utils import name_obj_dict


def parse_node_idx(node):
    return int(node.name.split('_')[1])


def edge_node_mapping(graph, by_input=False):
    if by_input:
        return {o: n for n in graph.node for o in n.input}
    else:
        return {o: n for n in graph.node for o in n.output}


def find_node_name_by_edge(graph, edge_name, by_input=True):
    # TODO: check when multiple nodes reference edge_name, whether
    #  only adding the first node is reasonable
    if by_input:
        return [n.name for n in graph.node if edge_name in n.input][0]
    else:
        return [n.name for n in graph.node if edge_name in n.output][0]


def union_sort(l1: list, l2: list):
    union_list = list(set(l1).union(l2))
    union_list.sort()
    return union_list


def get_edge_chain(edge_name, node_list, break_node_name, forward: bool):
    cur_edge = edge_name
    edge_chain = [cur_edge]
    while True:
        in_range = False
        for n in node_list:
            if forward and cur_edge in n.input:
                if n.name != break_node_name:
                    in_range = True
                    cur_edge = n.output[0]
                    break
            elif not forward and cur_edge in n.output:
                if n.name != break_node_name:
                    in_range = True
                    cur_edge = n.input[0]
                    break
        if not in_range:
            break
        else:
            edge_chain.append(cur_edge)
    return edge_chain


def make_zero_node(shape, node_id, edge_id):
    return make_constant(np.zeros(shape, dtype=np.float32), node_id, edge_id)


def make_constant(np_value, node_id, edge_id):
    shape = tuple(np_value.shape)
    value_tensor = onnx.helper.make_tensor(
        name="Node_%d_tensor" % node_id,
        data_type=onnx.TensorProto.FLOAT,
        dims=shape,
        vals=np_value.flatten()
    )

    node = onnx.helper.make_node(
        'Constant',
        [],
        ["%d" % edge_id],
        "Constant_%d" % node_id,
        value=value_tensor
    )
    return node


def topological_sort(node_list):
    input_node_mapping = {n.output[0]: [] for n in node_list}
    input_node_mapping.update({i: [] for n in node_list for i in n.input})
    for node in node_list:
        for i in node.input:
            input_node_mapping[i].append(node)
    name_node_mapping = {n.name: n for n in node_list}
    node_num_ref = {n.name: 0 for n in node_list}
    for node in node_list:
        for ref_n in input_node_mapping[node.output[0]]:
            node_num_ref[ref_n.name] += 1
    sorted_nodes = []
    zero_in = deque(name_node_mapping[n]
                    for n, ref in node_num_ref.items() if ref == 0)
    while zero_in:
        node = zero_in.popleft()
        sorted_nodes.append(node)
        for ref_n in input_node_mapping[node.output[0]]:
            node_num_ref[ref_n.name] -= 1
            if node_num_ref[ref_n.name] == 0:
                zero_in.append(ref_n)
    if len(sorted_nodes) != len(node_list):
        remain = []
        for n in node_list:
            if n not in sorted_nodes:
                remain.append("in: %s, out: %s, name: %s" %
                              (str(list(n.input)), str(list(n.output)), n.name))
        for r in remain:
            print(r)
        raise Exception("There's a loop in the graph")
    return sorted_nodes


def make_model(graph_nodes, model):
    sorted_graph_nodes = topological_sort(graph_nodes)
    ori_nodes = [n for n in model.graph.node]
    for n in ori_nodes:
        model.graph.node.remove(n)
    for n in reversed(sorted_graph_nodes):
        model.graph.node.insert(0, n)
    onnx.checker.check_model(model)


def replace_node(graph, ori_node_name, new_node):
    idx = [i for i, n in enumerate(graph.node) if n.name == ori_node_name][0]
    ori_node = graph.node[idx]
    graph.node.remove(ori_node)
    graph.node.insert(idx, new_node)


def remove_unref_nodes(graph):
    edge_node_map = {n.output[0]: n for n in graph.node}
    retained_node_names = set()

    edge_init_map = {i.name: i for i in graph.initializer}
    retained_init_names = set()

    edge_input_map = {i.name: i for i in graph.input}
    retained_input_names = set()

    searched_edges = deque(o.name for o in graph.output)
    while searched_edges:
        edge = searched_edges.popleft()
        if edge in edge_node_map.keys():
            def_node = edge_node_map[edge]
            retained_node_names.update([def_node.name])
            for i in def_node.input:
                searched_edges.append(i)
        elif edge in edge_init_map.keys():
            retained_init_names.update([edge])
        elif edge in edge_input_map.keys():
            retained_input_names.update([edge])
        else:
            continue

    deleted_set = set(n.name for n in graph.node).difference(
        retained_node_names)
    name_node_mapping = {n.name: n for n in graph.node}
    for name in deleted_set:
        n = name_node_mapping[name]
        graph.node.remove(n)

    deleted_init = set(edge_init_map.keys()).difference(retained_init_names)
    for name in deleted_init:
        graph.initializer.remove(edge_init_map[name])

    deleted_input = set(edge_input_map.keys()).difference(retained_input_names)
    for name in deleted_input:
        graph.input.remove(edge_input_map[name])


def prepare_run_dir(save_dir):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    build_dir = os.path.join(save_dir, "build")
    os.makedirs(build_dir)

    model_path = os.path.join(save_dir, "model.onnx")

    return model_path, build_dir

def get_inner_non_const_nodes(model):
    output_names = [o.name for o in model.graph.output]
    return [n for n in model.graph.node
            if n.op_type != 'Constant' and n.output[0] not in output_names]


def get_non_const_edges_info(model):
    model = shape_inference.infer_shapes(model)
    edges_name = [n.output[0] for n in get_inner_non_const_nodes(model)]
    name_info_mapping = name_obj_dict(model.graph.value_info)
    edges_info = [name_info_mapping[name] for name in edges_name]
    return edges_info
