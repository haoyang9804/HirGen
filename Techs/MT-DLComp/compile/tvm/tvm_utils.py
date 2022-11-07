import onnx

import tvm
from tvm import relay
from tvm.contrib import graph_runtime


def onnx_executor(model, input_shape, gpu=False):
    if gpu:
        target = "cuda"
        dev = tvm.gpu()
    else:
        target = "llvm"
        dev = tvm.cpu()
    mod, params = onnx2relay(model, input_shape)
    with tvm.transform.PassContext(opt_level=4):
        return relay.build_module.create_executor("graph", mod, dev, target), params


def onnx_exe_run(executor, params, np_input):
    out = executor.evaluate()(tvm.nd.array(np_input), **params)
    if not isinstance(out, list):
        out = [out]
    return [o.asnumpy() for o in out]


def onnx_compiler_run(model, np_input, gpu=False):
    exe, params = onnx_executor(model, tuple(np_input.shape), gpu)
    return onnx_exe_run(exe, params, np_input)


def onnx2relay(model, input_shape):
    shape_dict = {"input": input_shape}
    mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)
    return mod, params


def pytorch2onnx(torch_model, torch_input, sample_output, save_path):
    from onnxsim import simplify
    import torch
    # set the model to inference mode
    torch_model.eval()
    # Export the model
    torch.onnx.export(torch_model,  # model being run
                      torch_input,  # model input (or a tuple for multiple inputs)
                      save_path,
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      # opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      example_outputs=sample_output
                      )
    model = onnx.load(save_path)
    model_sim, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_sim, save_path)
    return model_sim


def get_context_target(gpu=False):
    if gpu:
        target = tvm.target.cuda()
    else:
        target = "llvm"
    ctx = tvm.context(target, 0)
    return ctx, target


def lib_load_run(lib_path, np_input, eval_time=False, gpu=False):
    lib = tvm.runtime.load_module(lib_path)
    ctx, _ = get_context_target(gpu)
    return lib_run(lib, np_input, ctx, eval_time)


def lib_build_run(onnx_model, np_input, output_shape=None, gpu=False, eval_time=False):
    target, ctx = get_context_target(gpu)
    lib = build_lib(onnx_model, np_input, gpu)
    return lib_run(lib, np_input, ctx, eval_time, output_shape)


def lib_run(lib, np_input, ctx, eval_time=False, output_shape=None):
    module = graph_runtime.GraphModule(lib['default'](ctx))
    return module_run(module, np_input, ctx, eval_time, output_shape)


def module_run(module, np_input, ctx, eval_time, output_shape):
    module.set_input('input', np_input)
    module.run()
    if output_shape is None:
        out = module.get_output(0).asnumpy()
    else:
        out = module.get_output(0, tvm.nd.empty(output_shape)).asnumpy()
    if eval_time:
        # evaluate 10 times
        sec = eval_module_run_time(module, ctx)
        return out, sec
    return out


def eval_module_run_time(module, ctx):
    time_eval = module.module.time_evaluator("run", ctx, number=10)
    sec = time_eval().results
    return sec[0]


def build_lib(onnx_model, np_input, target):
    input_shape = tuple(np_input.shape)
    net, params = relay.frontend.from_onnx(onnx_model, {'input': input_shape})
    with tvm.transform.PassContext(opt_level=4):
        lib = relay.build(net, target, params=params)
    return lib
