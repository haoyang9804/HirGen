import subprocess
import json

import numpy as np

from utils.onnx_utils import get_model_input, get_dim


def is_channel_last(input_shape):
    return input_shape[1] > input_shape[3]


def move_channel_last(input_data):
    return np.transpose(input_data, (0, 2, 3, 1))


def execute_cmd(*command_list):
    r = subprocess.run(list(command_list), capture_output=True)
    if r.returncode:
        if r.returncode == -11:
            return "Segmentation fault"
        return r.stderr
    else:
        return None


def write_in_out_info(file, onnx_model):
    input_edges = get_model_input(onnx_model)
    output_edges = onnx_model.graph.output

    in_out_info = {'input': [{'name': i.name, 'shape': get_dim(i)} for i in input_edges],
                   'output': [{'name': o.name, 'shape': get_dim(o)} for o in output_edges]}

    with open(file, 'w') as f:
        json.dump(in_out_info, f)


def read_in_out_info(file):
    with open(file, 'r') as f:
        inout_info = json.load(f)

    return inout_info
