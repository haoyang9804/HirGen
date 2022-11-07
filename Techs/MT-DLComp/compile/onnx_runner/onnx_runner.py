import os
import time

import onnx
import onnxruntime as rt
import numpy as np

from utils.onnx_utils import onnx_run
from compile.compile_err import CompilationError
from compile.runner import Runner

class OnnxRunner(Runner):
    def __init__(self, compiler_path, mode, cal_time):
        super().__init__(compiler_path, mode, cal_time)
        self.input_data = None

    def load_lib(self, build_dir):
        pass

    def run(self, run_dir):
        with open(os.path.join(run_dir, "model_path.txt"), 'r') as f:
            model_path = f.readline()
        result = onnx_run(self.input_data, model_path)
        np.save(os.path.join(run_dir, "out.npy"), result[0])
        if self.cal_time:
            return cal_onnx_run_time(self.input_data, model_path)

    @staticmethod
    def get_output(run_dir):
        return np.load(os.path.join(run_dir, "out.npy"))

    @staticmethod
    def get_edge_value(run_dir):
        pass

    def set_input(self, input_file):
        self.input_data = np.load(input_file)

    def compile(self, model_path, build_dir):
        with open(os.path.join(build_dir, "model_path.txt"), 'w') as f:
            f.write(model_path)

def cal_onnx_run_time(input_data, model_path, repeat_times=20):
    sess = rt.InferenceSession(model_path)
    if sess.get_inputs():
        input_name = sess.get_inputs()[0].name
        input_dict = {input_name: input_data}
    else:
        input_dict = {}
    output_name = [o.name for o in sess.get_outputs()]
    # sess.run(output_name, input_dict)
    st = time.time()
    for i in range(repeat_times):
        sess.run(output_name, input_dict)
    return (time.time() - st) * 1000 / repeat_times
