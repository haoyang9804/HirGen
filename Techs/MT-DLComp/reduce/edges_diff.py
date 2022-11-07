import os

import numpy as np
import shutil

import onnx

import compile.runner
from compile.output_diff import array_diff, write_output_diff
from reduce.reduce_utils import get_non_const_edges_info, prepare_run_dir
from utils.onnx_utils import onnx_run


def compare_onnx_compiler_edge_value(model, runner: compile.runner.Runner,
                                     fault_output,
                                     run_dir, data_path,
                                     output_file, onnx_edge_save_dir):
    os.makedirs(onnx_edge_save_dir, exist_ok=True)
    model_path, build_dir = prepare_run_dir(run_dir)

    input_data = np.load(data_path)

    edge_diff = []

    edges_info = get_non_const_edges_info(model)
    for edge in edges_info:
        model.graph.output.insert(0, edge)
        onnx.save(model, model_path)
        onnx_edge_value, onnx_output_value = onnx_run(input_data, model_path)
        np.savetxt(os.path.join(onnx_edge_save_dir, f"{edge.name}.txt"),
                   onnx_edge_value.flatten(), fmt="%.8f")

        try:
            runner_edge_value, runner_output_value = runner.compile_run(
                model_path, build_dir, data_path, view_edge=True)
        except Exception:
            model.graph.output.remove(edge)
            continue

        model.graph.output.remove(edge)

        output_abs_diff = np.max(
            np.abs(runner_output_value.flatten()- fault_output.flatten()))

        edge_diff.append(("%f" % output_abs_diff, *array_diff(runner_edge_value, onnx_edge_value)))

    write_output_diff(output_file, edge_diff, [e.name for e in edges_info])
    shutil.rmtree(run_dir)
