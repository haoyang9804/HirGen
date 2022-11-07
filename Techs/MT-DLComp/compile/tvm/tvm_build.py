import os
import time

import tvm
from tvm import relay
from tvm.contrib import graph_executor
from utils.onnx_utils import get_model_input, get_dim


def build_model(onnx_model, build_dir, opt_level=2):
    shape_dict = {i.name: get_dim(i) for i in get_model_input(onnx_model)}
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
    target = tvm.target.Target("llvm", host="llvm")
    with tvm.transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target=target, params=params)
    lib.export_library(os.path.join(build_dir, "compiled_lib.so"))


def load_lib(build_dir):
    dev = tvm.cpu(0)
    lib: tvm.runtime.Module = tvm.runtime.load_module(os.path.join(build_dir, "compiled_lib.so"))
    # Call the library factory function for default and create
    # a new runtime.Module, wrap with graph module.
    gmod = graph_executor.GraphModule(lib["default"](dev))
    return gmod


def run_graph_module(gmod, input_dict: dict, num_outputs):
    for name, value in input_dict.items():
        gmod.set_input(name, tvm.nd.array(value))
    gmod.run()
    return [gmod.get_output(i).numpy() for i in range(num_outputs)]


def cal_run_time(gmod, input_dict, repeat_times=20):
    st_time = time.time()
    for _ in range(repeat_times):
        for name, value in input_dict.items():
            gmod.set_input(name, tvm.nd.array(value))
        gmod.run()
    ed_time = time.time()
    return (ed_time - st_time) * 1000 / repeat_times
