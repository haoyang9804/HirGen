import os
import re

from compile.compile_err import CompilationError
from compile.compile_utils import execute_cmd

# model_path = "/export/d1/dwxiao/TVM/results/resnet18/1/hybrid/glow/reduce/841/reduced_model.onnx"
# model_path = "../temp/reduced_model.onnx"
#
# cmd_list = ['/export/d1/dwxiao/glow-build/bin/model-compiler',
#                 '-backend=CPU',
#                 '-model=%s' % model_path,
#                 '-emit-bundle=/export/d1/dwxiao/TVM/tmp_models',
#                 '-network-name=model']
# r = execute_cmd(*cmd_list)
# s = bytes.decode(r)
#
# p = re.compile("Error code: (\w+)")
# m = p.search(s)
# if m:
#     print(m.group(1))

from compile.make_runner import make_runner

runner = make_runner('glow', "/export/d2/dwxiao/build_Release/bin/model-compiler", '../data/data.npy', 'default', False)

# runner.compile("/export/d2/dwxiao/results/resnet18/1/hybrid/mutants/models/334.onnx", "../temp")
runner.run("../temp", )

# print(os.getcwd())
# os.chdir("../temp")
# r = execute_cmd("./main", "../data/data.bin")
# print(r)
