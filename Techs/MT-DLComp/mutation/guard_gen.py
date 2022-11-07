import os
import random
import numpy as np

from mutation.node_gen import NodeChainGen
from mutation.edge_node import EdgeNode, convert_edge_to_value_info
from mutation import mutate_utils


class UniversalGuard:
    def __init__(self, generator: NodeChainGen, seed_model):
        self.gen = generator

        eps_edge = self.gen.make_constant(np.array([1e-9], dtype=np.float32))

        minus_two_edge = self.gen.make_constant(
            np.array([-2], dtype=np.float32))

        seed_model.graph.node.insert(0, eps_edge.def_node)
        seed_model.graph.node.insert(0, minus_two_edge.def_node)

        self.eps = eps_edge
        self.minus_two = minus_two_edge

    def gen_guard(self, edge_a: EdgeNode, edge_b: EdgeNode):
        edges = self.guard_formula(edge_a, edge_b)
        relu_edges = self.add_relu(edges[-1])
        edges.extend(relu_edges)

        return edges

    def guard_formula(self, edge_a: EdgeNode, edge_b: EdgeNode):
        edges, (edge_a, edge_b), c_shape = self.gen.bilateral_shape_matching(
            [edge_a, edge_b], True)

        a2_edge = self.gen.make_edge_node(
            'Mul', [edge_a, edge_a], edge_a.shape, edge_a.zero)
        edges.append(a2_edge)

        b2_edge = self.gen.make_edge_node(
            'Mul', [edge_b, edge_b], edge_b.shape, edge_b.zero)
        edges.append(b2_edge)

        ab_edge = self.gen.make_edge_node(
            'Mul', [edge_a, edge_b], c_shape, edge_a.zero or edge_b.zero)
        edges.append(ab_edge)

        ab2_edge = self.gen.make_edge_node(
            'Mul', [ab_edge, self.minus_two], c_shape, ab_edge.zero)
        edges.append(ab2_edge)

        sum_e = self.gen.make_edge_node(
            'Add', [a2_edge, b2_edge], c_shape,
            edge_a.zero and edge_b.zero
        )
        edges.append(sum_e)

        sum_e = self.gen.make_edge_node(
            'Add', [sum_e, ab2_edge], c_shape,
            (edge_a.zero and edge_b.zero) or (edge_a == edge_b)
        )
        edges.append(sum_e)

        return edges

    def add_relu(self, in_edge: EdgeNode):
        in_shape = in_edge.shape
        edges = []

        edge = self.gen.make_edge_node(
            'Add', [in_edge, self.eps], in_shape, in_edge.zero)
        edges.append(edge)

        edge = self.gen.make_edge_node('Neg', edge, in_shape, edge.zero)
        edges.append(edge)

        edge = self.gen.make_edge_node('Relu', edge, in_shape, True)
        edges.append(edge)

        return edges


class InputDependentGuard:
    def __init__(self, generator: NodeChainGen, tmp_save_path, input_data):
        self.gen = generator
        self.tmp_path = tmp_save_path
        self.input_data = input_data

    def gen_guard(self, model, guarded_edge):
        new_edges = []
        reduce_edge = self.gen.make_reduce(
            new_edges, guarded_edge, keep_dims=True)

        if new_edges:
            edge_idx = [n.output[0] for n in model.graph.node].index(
                guarded_edge.name)
            model.graph.node.insert(edge_idx + 1, reduce_edge.def_node)

        onnx_edge = convert_edge_to_value_info([reduce_edge])[0]
        guard_val = mutate_utils.get_internal_edge_output(
            model, onnx_edge, self.input_data, self.tmp_path)

        if new_edges:
            model.graph.node.remove(reduce_edge.def_node)

        const_edge = self.gen.make_constant(guard_val)
        new_edges.append(const_edge)

        sub_edge = self.gen.make_edge_node(
            'Sub', [reduce_edge, const_edge], reduce_edge.shape, True)
        new_edges.append(sub_edge)

        return new_edges


class HybridGuard:
    def __init__(self, generator: NodeChainGen, seed_model,
                 temp_model_save_path, input_data_path):
        self.uni_guard = UniversalGuard(generator, seed_model)

        self.dep_guard = InputDependentGuard(
            generator, temp_model_save_path, np.load(input_data_path))

    def gen_guard(self, model, *guard_input_edges):
        if random.randint(0, 1):
            guard_edges = self.uni_guard.gen_guard(guard_input_edges[0], guard_input_edges[1])
        else:
            guard_edges = self.dep_guard.gen_guard(model, guard_input_edges[0])
        return guard_edges


class GuardDispatcher:
    def __init__(self, generator: NodeChainGen, mode, seed_model=None,
                 temp_model_save_path=None, input_data_path=None):
        if mode == 'universal':
            self.guard = UniversalGuard(generator, seed_model)
        elif mode == 'per_input':
            self.guard = InputDependentGuard(
                generator, temp_model_save_path,
                np.load(input_data_path))
        elif mode == 'hybrid':
            self.guard = HybridGuard(
                generator, seed_model, temp_model_save_path, input_data_path)
        else:
            raise Exception("The mode should be in 'universal', 'per_input', or 'hybrid'.")

        self.mode = mode

    def gen_guard(self, model, *input_guard_edges):
        if self.mode == 'universal':
            random.randint(0, 1)  # for compatibility with hybrid
            guard_edges = self.guard.gen_guard(input_guard_edges[0], input_guard_edges[1])
        elif self.mode == 'per_input':
            random.randint(0, 1)  # for compatibility with hybrid
            guard_edges = self.guard.gen_guard(model, input_guard_edges[0])
        else:
            guard_edges = self.guard.gen_guard(model, *input_guard_edges)
        return guard_edges
