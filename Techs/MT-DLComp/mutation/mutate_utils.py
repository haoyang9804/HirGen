import onnx

import numpy as np

from utils.onnx_utils import onnx_run, name_obj_dict


def convert2iter(o):
    if not isinstance(o, tuple) and not isinstance(o, list):
        return [o]
    else:
        return o


def numpy_onnx_type_mapping(np_type):
    if np_type == np.float32:
        return onnx.TensorProto.FLOAT
    elif np_type == np.int32:
        return onnx.TensorProto.INT32
    elif np_type == np.int64:
        return onnx.TensorProto.INT64
    else:
        raise Exception("The type cannot be matched to onnx type")


def get_constant_edge_val(model, edge):
    node = [n for n in model.graph.node if edge.name in n.output]
    if not node:
        return
    node = node[0]
    if node.op_type != 'Constant':
        return
    val = node.attribute[0].t.float_data
    val_shape = node.attribute[0].t.dims
    val = np.array(list(val), dtype=np.float32).reshape(val_shape)
    return val


def get_internal_edge_output(model, edge, input_data, temp_save_path):
    val = get_constant_edge_val(model, edge)
    if val is not None:
        return val
    model.graph.output.insert(0, edge)
    onnx.save(model, temp_save_path)

    out_list = onnx_run(input_data, temp_save_path)
    model.graph.output.remove(edge)
    return out_list[0]


def get_ordered_inner_edges(graph):
    value_info_name_mapping = name_obj_dict(graph.value_info)
    edge_def_order = [out for node in graph.node for out in node.output]
    value_info_name = set(v.name for v in graph.value_info)
    inner_edges_name = list(set(edge_def_order).intersection(value_info_name))
    inner_edges_name.sort(key=edge_def_order.index)
    return [value_info_name_mapping[edge] for edge in inner_edges_name]


def get_value_name_list(graph):
    names = [t.name for t in graph.value_info]
    names.extend([t.name for t in graph.input])
    names.extend([t.name for t in graph.output])
    names.extend([t.name for t in graph.initializer])
    return names


def non_node_output_edges(graph):
    non_node_def_edges = set(e.name for e in graph.initializer)
    non_node_def_edges.update(set(e.name for e in graph.input))
    return non_node_def_edges


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def make_value_info(name, shape, tensor_type=onnx.TensorProto.FLOAT):
    return onnx.helper.make_tensor_value_info(name, tensor_type, shape)


def insert_list(ins_obj, items: list, ins_index):
    for item in reversed(items):
        ins_obj.insert(ins_index, item)


def replace_node_output(node, new_output_name):
    ori_output = [o for o in node.output]
    for output in ori_output:
        node.output.remove(output)
    node.output.insert(0, new_output_name)


def is_val_zero(np_value):
    return np.max(np.abs(np_value)) < 1e-7
