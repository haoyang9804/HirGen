import os
import onnx
from onnx import shape_inference

from compile.make_runner import make_runner

def test_tvm_build():
    root_path = "/export/d2/dwxiao"
    onnx_model_path = os.path.join(root_path, "results/resnet18/1/hybrid/mutants/models/seed.onnx")
    compile_path = os.path.join(root_path, "temp")

    model = shape_inference.infer_shapes(onnx.load_model(onnx_model_path))
    for edge in model.graph.value_info:
        if edge.name != 'input' and edge.name != 'output':
            print(edge.name)
            break
    model.graph.output.insert(0, edge)
    new_model_path = os.path.join(compile_path, "model.onnx")
    onnx.save(model, new_model_path)


    runner = make_runner("tvm", None, os.path.join(root_path, "data", "data.npy"), "default", False)
    runner.compile(new_model_path, compile_path)
    runner.run(compile_path, )

    print(runner.get_edge_value(compile_path))
    print()
    print(runner.get_output(compile_path))
