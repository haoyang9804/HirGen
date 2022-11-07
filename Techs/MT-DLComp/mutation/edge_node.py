import utils.onnx_utils
from mutation import mutate_utils
from utils.onnx_utils import get_dim
from mutation.mutate_utils import make_value_info


class EdgeNode:
    def __init__(self, name, edge_shape, def_node=None, zero=False):
        self.name = name
        self.shape = edge_shape
        self.zero = zero
        self.def_node = def_node


def convert_onnx_to_edge(graph):
    name_edge_mapping = utils.onnx_utils.name_obj_dict(graph.value_info)
    edges = []

    for node in graph.node:
        e_name = node.output[0]
        try:
            onnx_edge = name_edge_mapping[e_name]
            edge = EdgeNode(e_name, get_dim(onnx_edge), node, False)
        except KeyError:
            if 'Constant' in node.name:
                value = node.attribute[0]
                edge_shape = tuple(value.t.dims)
                edge = EdgeNode(e_name, edge_shape, node, False)
            else:
                output_edge = [o for o in graph.output
                               if o.name in node.output][0]
                edge = EdgeNode(
                    output_edge.name, get_dim(output_edge), node, False)

        edges.append(edge)

    assert edges[-1].name == 'output'
    return edges


def retrieve_node_from_edges(edge_list: list):
    return [e.def_node for e in edge_list]


def convert_edge_to_value_info(edges):
    if isinstance(edges, list) or isinstance(edges, tuple):
        return [make_value_info(e.name, e.shape) for e in edges]
    else:
        return make_value_info(edges.name, edges.shape)
