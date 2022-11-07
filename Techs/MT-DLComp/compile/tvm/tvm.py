import os

import numpy as np
import onnx
from tvm import TVMError

from compile import compile_utils
from compile.runner import Runner
from compile.tvm.tvm_err import TvmError
from compile.tvm import tvm_build


class TVMRunner(Runner):
    def __init__(self, compiler_path, mode, cal_time):
        super().__init__(compiler_path, mode, cal_time)
        self.input_data = None
        self.gmod = None

    def set_input(self, input_file):
        self.input_data = np.load(input_file)
        if self.channel_last:
            self.input_data = compile_utils.move_channel_last(self.input_data)

    def compile(self, model_path, build_dir):
        onnx_model = onnx.load(model_path)
        try:
            tvm_build.build_model(onnx_model, build_dir)
        except TVMError as e:
            raise TvmError(model_path, str(e))


    def load_lib(self, run_dir):
        gmod = tvm_build.load_lib(run_dir)
        self.gmod = gmod

    def run(self, run_dir):
        input_dict = {self.input_name: self.input_data} if self.input_name else {}

        result = tvm_build.run_graph_module(self.gmod, input_dict, len(self.output_names))

        self.save_result(result, run_dir)

        if self.cal_time:
            return tvm_build.cal_run_time(self.gmod, input_dict)

    def save_result(self, result, run_dir):
        if len(self.output_names) > 1:
            np.save(os.path.join(run_dir, "edge.npy"), result[0])
            np.save(os.path.join(run_dir, "out.npy"), result[1])
        else:
            np.save(os.path.join(run_dir, "out.npy"), result[0])

    @staticmethod
    def get_output(run_dir):
        return np.load(os.path.join(run_dir, 'out.npy'))

    @staticmethod
    def get_edge_value(run_dir):
        return np.load(os.path.join(run_dir, "edge.npy"))
