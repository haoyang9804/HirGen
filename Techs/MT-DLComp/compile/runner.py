import os

import onnx

from compile import compile_utils


class Runner:
    def __init__(self, compiler_path, mode, cal_time):
        self.compiler_path = compiler_path
        self.cal_time = cal_time
        self.mode = mode

        self.data_path = None
        self.input_name = None
        self.output_names = []
        self.channel_last = False

        self.run_return = None

    def set_input(self, data_path):
        self.data_path = os.path.abspath(data_path)

    def get_inout_info(self, run_dir):
        inout_info = compile_utils.read_in_out_info(os.path.join(run_dir, "in_out.json"))

        if inout_info['input']:
            self.input_name = inout_info['input'][0]['name']
            input_shape = inout_info['input'][0]['shape']
            self.channel_last = compile_utils.is_channel_last(input_shape)
        else:
            self.input_name = None
        self.output_names = [o['name'] for o in inout_info['output']]

    @staticmethod
    def save_inout_info(onnx_model, build_dir):
        compile_utils.write_in_out_info(
            os.path.join(build_dir, "in_out.json"), onnx_model)

    def load_lib(self, build_dir):
        raise NotImplementedError()

    def compile_run(self, model_path, build_dir, data_path, view_edge=False):
        self.build(model_path, build_dir)
        self.run_with_input(build_dir, data_path)
        if view_edge:
            return self.get_edge_value(build_dir), self.get_output(build_dir)
        else:
            return self.get_output(build_dir)

    def build(self, model_path, build_dir):
        self.compile(model_path, build_dir)
        model = onnx.load(model_path)
        self.save_inout_info(model, build_dir)

    def run_with_input(self, run_dir, data_path):
        self.set_input(data_path)
        self.run_return = self.run(run_dir)

    def prepare_run(self, run_dir):
        self.get_inout_info(run_dir)
        self.load_lib(run_dir)

    def compile(self, model_path, build_dir):
        raise NotImplementedError()

    def run(self, run_dir):
        raise NotImplementedError()

    def get_run_time(self):
        return self.run_return

    @staticmethod
    def get_output(run_dir):
        raise NotImplementedError()

    @staticmethod
    def get_edge_value(run_dir):
        raise NotImplementedError()
