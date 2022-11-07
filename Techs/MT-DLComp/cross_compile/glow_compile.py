import os

import numpy as np
import onnx

from compile.make_runner import make_runner

from compile.glow.glow import glow_run
from compile.glow.glow import glow_compile, gcc_compile
from compile.glow.compile_manager import ModelZooManager

from compile.glow.form_cpp import get_inout_info

from cross_compile.fix_batch import fix_model_batch


def batch_fixing(ori_model_path, new_model_path):
    model = onnx.load(ori_model_path)
    fix_model_batch(model)
    onnx.checker.check_model(model)
    onnx.save(model, new_model_path)


class ModelZooCompiler:
    def __init__(self):
        self.input_data_path = None

    def set_input(self, input_data_path):
        pass

    def compile(self, model_path, build_dir):
        raise NotImplementedError()

    def load_lib(self, run_dir):
        raise NotImplementedError()

    def run(self, input_data_path):
        raise NotImplementedError()


class GlowModelZooCompiler(ModelZooCompiler):
    def __init__(self, compiler_path):
        super().__init__()
        self.compiler_path = compiler_path
        self.channel_last = False
        self.run_dir = None

    def compile(self, model_path, build_dir):
        glow_compile(self.compiler_path, model_path, build_dir)

        ModelZooManager.form_run_cpp(build_dir)
        gcc_compile(build_dir)

    def set_input(self, input_data_path):
        if self.channel_last:
            if input_data_path.endswith(".npy"):
                input_data = np.load(input_data_path)
            else:
                input_data = np.fromfile(input_data_path, dtype=np.float32)
            input_data = np.transpose(input_data, (0, 2, 3, 1))
            self.input_data_path = os.path.join(self.run_dir, "data.bin")
            input_data.flatten().tofile(self.input_data_path)
        else:
            if input_data_path.endswith(".npy"):
                input_data = np.load(input_data_path)
                self.input_data_path = os.path.join(self.run_dir, "data.bin")
                input_data.flatten().tofile(self.input_data_path)
            else:
                self.input_data_path = input_data_path

    def load_lib(self, run_dir):
        header_file = os.path.join(run_dir, "model.h")
        inout_info = get_inout_info(header_file)
        input_shape = inout_info['input']['shape']
        if input_shape[1] < input_shape[3]:
            self.channel_last = False
        else:
            self.channel_last = True

        self.run_dir = run_dir

    def run(self, run_dir):
        glow_run(run_dir, self.input_data_path, False)

    @staticmethod
    def get_output(run_dir):
        return np.fromfile(os.path.join(run_dir, "out.bin"), dtype=np.float32)


def select_compiler(compiler_name, compiler_path=None):
    if compiler_name == 'glow':
        return GlowModelZooCompiler(compiler_path)
    elif compiler_name == 'tvm':
        return make_runner('tvm', compiler_path, None, 'default', False)
    else:
        return make_runner('tensorflow', compiler_path, None, 'default', False)
