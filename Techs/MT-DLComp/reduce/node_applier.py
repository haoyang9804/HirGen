import copy
import shutil

import onnx
import numpy as np
import os

from reduce import reduce_utils
from utils.onnx_utils import onnx_run


class NodeApplier:
    def __init__(self, fault_model, delta_nodes, value_list):
        self.fault_model = fault_model
        self.delta_nodes = delta_nodes
        self.value_list = value_list

    def valid_range(self):
        return range(len(self.delta_nodes))

    def apply(self, delta_ids):
        model = copy.copy(self.fault_model)
        deleted_ids = set(range(len(self.delta_nodes))).difference(delta_ids)

        for idx in deleted_ids:
            node = self.delta_nodes[idx]
            value = self.value_list[idx]
            const_node = reduce_utils.make_constant(
                value, reduce_utils.parse_node_idx(node), int(node.output[0]))
            reduce_utils.replace_node(model.graph, node.name, const_node)

        reduce_utils.remove_unref_nodes(model.graph)
        onnx.checker.check_model(model)
        return model


def get_edges_value(model, edge_info_list, input_data, temp_dir):
    model_path = os.path.join(temp_dir, "model.onnx")
    edge_value_list = []
    for edge in edge_info_list:
        model.graph.output.insert(0, edge)
        onnx.save(model, model_path)
        edge_value, _ = onnx_run(input_data, model_path)
        model.graph.output.remove(edge)
        edge_value_list.append(edge_value)

    return edge_value_list


def make_node_applier(model_path, input_file, temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
    model = onnx.load(model_path)
    delta_nodes = reduce_utils.get_inner_non_const_nodes(model)
    edges_info = reduce_utils.get_non_const_edges_info(model)

    edge_val = get_edges_value(model, edges_info, np.load(input_file), temp_dir)
    shutil.rmtree(temp_dir)

    return NodeApplier(model, delta_nodes, edge_val)
