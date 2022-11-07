from mutation.node_gen import NodeChainGen
import random
import numpy as np


class DeadGenerator:
    def __init__(self, generator: NodeChainGen):
        self.gen = generator
        self.op_types = ['Add', 'Sub', 'Mul', 'Conv', 'Dense']
        self.kernel_size = [1, 'all', 3]

    def gen_dead_edge(self, op_type, edge_a, edge_b):
        zero = (edge_a.zero or edge_b.zero) if op_type == 'Mul' \
            else (edge_a.zero and edge_b.zero)
        return self.gen.make_multi_input_node(
            op_type, [edge_a, edge_b], True, zero)

    def gen_dead(self, edge_a, edge_b):
        op_type = self.op_types[random.randint(0, len(self.op_types) - 1)]
        if op_type == 'Conv':
            new_edges = []
            self.gen_conv(new_edges, edge_a)
            return new_edges
        elif op_type == "Dense":
            new_edges = []
            self.gen_dense(new_edges, edge_b)
            return new_edges
        else:
            return self.gen_dead_edge(op_type, edge_a, edge_b)

    def gen_dense(self, new_edges, in_edge):
        edge = self.gen.make_unsqueeze(new_edges, in_edge, 2)
        num_features = edge.shape[-1]

        mul_val = np.random.randn(num_features, num_features).astype(np.float32)
        mul_val_edge = self.gen.make_constant(mul_val)
        new_edges.append(mul_val_edge)

        mul_out = self.gen.make_edge_node(
            'MatMul', [edge, mul_val_edge], edge.shape, edge.zero)
        new_edges.append(mul_out)

        add_val = np.random.randn(num_features).astype(np.float32)
        add_val_edge = self.gen.make_constant(add_val)
        new_edges.append(add_val_edge)

        add_out = self.gen.make_edge_node(
            'Add', [mul_out, add_val_edge], mul_out.shape, False)
        new_edges.append(add_out)

        return add_out

    def get_kernel_shape(self, in_shape):
        min_dim = min(in_shape[2:])
        if min_dim < 3:
            k_size = self.kernel_size[random.randint(0, 1)]
        else:
            k_size = self.kernel_size[random.randint(0, 2)]

        if k_size == 'all':
            kernel_shape = in_shape[1], in_shape[1], in_shape[2], in_shape[3]
        else:
            kernel_shape = in_shape[1], in_shape[1], k_size, k_size
        return kernel_shape

    def gen_conv(self, new_edges, edge):
        rank_edge = self.gen.match_rank(new_edges, edge, 4)

        kernel_shape = self.get_kernel_shape(rank_edge.shape)
        np_kernel_val = np.random.randn(*kernel_shape).astype(np.float32)

        conv_edge = self.gen.make_conv(new_edges, rank_edge, np_kernel_val)
        return conv_edge
