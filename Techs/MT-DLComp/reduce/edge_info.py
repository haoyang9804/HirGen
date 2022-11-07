import copy

from mutation.node_gen import NodeChainGen
from mutation.edge_node import EdgeNode
from mutation.mutate_utils import replace_node_output
from reduce import reduce_utils


class EdgeInfo:
    def __init__(self, name, shape, defining_node, dep_edges,
                 zero, is_original=False):
        self.name = name
        self.shape = shape
        self.dep_edges = dep_edges
        self.def_node = defining_node
        self.is_original = is_original
        self.applied = True if is_original else False
        self.zero = False if self.is_original else zero

    def is_applied(self):
        if self.is_original:
            return True
        return self.applied

    def set_applied(self):
        if not self.is_original:
            self.applied = True

    def is_zero(self):
        return self.zero

    def reset(self):
        if not self.is_original:
            self.applied = False

    def propagate(self, retained_nodes: list, force_retain=False):
        raise NotImplementedError("Not belonging to any edge type")

    def apply(self):
        raise NotImplementedError("Apply not defined")

    def make_zero_node(self):
        return reduce_utils.make_zero_node(
            self.shape,
            reduce_utils.parse_node_idx(self.def_node),
            int(self.name)
        )

    def __hash__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other.name


class PlainEdge(EdgeInfo):
    def change_node_input(self, input_edges_name):
        node = copy.copy(self.def_node)
        ori_input = copy.copy(list(node.input))
        for e in ori_input:
            node.input.remove(e)
        for new_e in reversed(input_edges_name):
            node.input.insert(0, new_e)
        return node

    def apply(self):
        retained_nodes = []
        self.propagate(retained_nodes, True)
        return retained_nodes

    def propagate(self, retained_nodes: list, force_retain=False):
        if self.is_applied():
            return self.name
        self.set_applied()
        if not force_retain and self.is_zero():  # zero and not force retained
            retained_nodes.append(self.make_zero_node())
            return self.name
        # non-zero or force retained
        input_edges_name = []
        for dep_e in self.dep_edges:
            in_edge_name = dep_e.propagate(retained_nodes)
            input_edges_name.append(in_edge_name)
        retained_nodes.append(self.change_node_input(input_edges_name))
        return self.name


class SubsEdge(EdgeInfo):
    def __init__(self, name, shape, defining_node, dep_edges, zero):
        super().__init__(name, shape, defining_node, dep_edges, zero)
        self.mul_out_edge = dep_edges[0]
        self.ori_subs_edge = dep_edges[1]
        self.cur_subs_edge = None
        self.cur_subs_node = None
        self.chain_gen = None
        self.ori_shape = shape

    def set_subs(self, subs_edge: EdgeInfo, cur_subs_node,
                 chain_gen: NodeChainGen):
        self.cur_subs_edge = subs_edge
        self.cur_subs_node = cur_subs_node
        self.chain_gen = chain_gen

    def reset(self):
        super().reset()
        self.cur_subs_edge = None
        self.cur_subs_node = None
        self.chain_gen = None
        self.shape = self.ori_shape

    def is_same_place(self):
        if self.applied and self.cur_subs_edge == self.ori_subs_edge:
            if isinstance(self.cur_subs_edge, SubsEdge):
                return self.cur_subs_edge.is_same_place()
            else:
                return True
        return False

    def gen_shape_matching(self):
        in_name = self.mul_out_edge.name
        src_shape = self.mul_out_edge.shape
        tgt_shape = self.cur_subs_edge.shape

        matching_edges = []
        self.chain_gen.unilateral_shape_matching(
            matching_edges, EdgeNode(in_name, src_shape), tgt_shape, True
        )
        return [e.def_node for e in matching_edges]

    def apply(self):
        replace_node_output(self.cur_subs_node, self.name)
        self.shape = self.cur_subs_edge.shape

        retained_nodes = self.gen_shape_matching()

        add_node = copy.copy(self.def_node)
        replace_node_output(add_node, self.cur_subs_edge.name)
        if retained_nodes:
            add_node.input.remove(self.mul_out_edge.name)
            add_node.input.insert(0, retained_nodes[-1].output[0])

        retained_nodes.append(self.cur_subs_node)
        retained_nodes.append(add_node)
        self.set_applied()
        return retained_nodes

    def propagate(self, retained_nodes, force_retain=False):
        if self.is_same_place():  # applied and the same as original
            return self.name
        # not applied or applied but not the same as original
        return self.ori_subs_edge.propagate(retained_nodes)


def get_subs_place(graph_nodes, potential_ins_places):
    subs_node, idx = None, None
    graph_nodes_name = [n.name for n in graph_nodes]
    for potential_pos in potential_ins_places:
        try:
            idx = graph_nodes_name.index(potential_pos)
            subs_node = graph_nodes[idx]
            break
        except ValueError:
            continue
    return idx, subs_node


def set_subs_place(delta_subs_edge, potential_ins_places,
                   graph_nodes, name_info_mapping, chain_gen):
    idx, subs_node = get_subs_place(
        graph_nodes, potential_ins_places)

    subs_edge_info = name_info_mapping[subs_node.output[0]]
    delta_subs_edge.set_subs(subs_edge_info, subs_node, chain_gen)
    graph_nodes.remove(subs_node)
    return idx, subs_node
