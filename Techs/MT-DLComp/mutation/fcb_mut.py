import copy
import random
import os
import tqdm
import pickle

import onnx
from onnx import shape_inference

from utils.path_utils import remove_file, clear_and_make_dir
from mutation import edge_node, mutate_utils
from mutation.node_gen import make_node_chain_generator
from mutation.guard_gen import GuardDispatcher
from mutation.dead_gen import DeadGenerator


def select_places(sequence, k):
    for i in range(5):
        chosen = random.choices(sequence, k=k)
        subs_place = max(chosen)
        chosen.remove(subs_place)
        if max(chosen) != subs_place:
            return subs_place, chosen
    raise Exception("Cannot find suitable places")


def save_mut_info(save_file, dead_edges, subs_new_edge, subs_ori_edge):
    with open(save_file, 'wb') as f:
        pickle.dump({'dead_edges': dead_edges,
                     'subs_new_edge': subs_new_edge,
                     'subs_ori_edge': subs_ori_edge}, f)


def read_mut_info(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data['dead_edges'], data['subs_new_edge'], data['subs_ori_edge']


class FCBMutator:
    def __init__(self, seed_model, mode, saving_dir, input_data_path):
        self.gen = make_node_chain_generator(seed_model)

        self.seed_model = shape_inference.infer_shapes(seed_model)

        self.dead_gen = DeadGenerator(self.gen)

        self.temp_model_save_path = os.path.join(saving_dir, "tmp.onnx")
        clear_and_make_dir(saving_dir)
        self.root_save_dir = saving_dir

        self.guard_gen = GuardDispatcher(self.gen, mode, self.seed_model,
                                         self.temp_model_save_path, input_data_path)

    def mutate(self, times, saving_frequency):
        model = copy.copy(self.seed_model)
        all_edges = edge_node.convert_onnx_to_edge(model.graph)

        model_dir = os.path.join(self.root_save_dir, "models")
        edge_dir = os.path.join(self.root_save_dir, "mut_info")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(edge_dir, exist_ok=True)

        onnx.save(model, os.path.join(model_dir, "seed.onnx"))

        for i in tqdm.tqdm(range(1, times + 1)):
            dead_edges, subs_new_edge, subs_add = \
                self.mutate_once(model, all_edges)
            if i % saving_frequency == 0:
                onnx.save(model, os.path.join(model_dir, "%d.onnx" % i))
            # onnx.checker.check_model(model)

            save_mut_info(os.path.join(edge_dir, "%d.txt" % i),
                          dead_edges, subs_new_edge, subs_add)

        remove_file(self.temp_model_save_path)

    def mutate_once(self, model, all_edges):
        subs_place, dep_places = \
            select_places(range(2, len(all_edges) - 1), 5)

        dead_in_a, dead_in_b = all_edges[dep_places[0]], all_edges[dep_places[1]]
        dead_edges = self.dead_gen.gen_dead(dead_in_a, dead_in_b)

        guard_in_a, guard_in_b = \
            all_edges[dep_places[2]], all_edges[dep_places[3]]

        guard_edges = self.guard_gen.gen_guard(model, guard_in_a, guard_in_b)

        mul_edges = self.gen.make_multi_input_node(
            'Mul', [dead_edges[-1], guard_edges[-1]], True, True)

        new_edges = dead_edges + guard_edges + mul_edges
        return insert_dead_edges(
            all_edges, model, subs_place, self.gen, new_edges)


def insert_dead_edges(edge_node_list, model, subs_place, gen, dead_edges):
    subs_ori_edge = edge_node_list[subs_place]
    subs_node = subs_ori_edge.def_node

    edge_node_list.remove(subs_ori_edge)
    model.graph.node.remove(subs_node)

    subs_add_in_matched = []
    subs_add_in = gen.unilateral_shape_matching(
        subs_add_in_matched, dead_edges[-1], subs_ori_edge.shape, True
    )

    subs_new, subs_add = gen.make_subs_add(subs_ori_edge, subs_add_in)

    new_edges = dead_edges + subs_add_in_matched + [subs_new, subs_add]

    new_onnx_edges = edge_node.convert_edge_to_value_info(new_edges[:-1])
    for onnx_edge in new_onnx_edges:
        model.graph.value_info.append(onnx_edge)

    new_onnx_nodes = edge_node.retrieve_node_from_edges(new_edges)
    mutate_utils.insert_list(model.graph.node, new_onnx_nodes, subs_place)

    mutate_utils.insert_list(edge_node_list, new_edges, subs_place)

    return new_edges[:-2], subs_new, subs_add
