import os

import onnx
import numpy as np
import tensorflow as tf

from onnx_tf.backend import prepare

from compile.runner import Runner
from utils.path_utils import clear_and_make_dir


class TfRunner(Runner):
    def __init__(self, compiler_path, mode, cal_time):
        super().__init__(compiler_path, mode, cal_time)
        self.f = None
        self.input_name = None
        self.output_name = None
        self.input_shape = None
        self.input_data = None

    def set_input(self, input_data_path):
        self.input_data = np.load(input_data_path)
        if self.input_shape is not None and self.input_data.shape[1] != self.input_shape[1]:
            self.input_data = np.transpose(self.input_data, (0, 2, 3, 1))

    def compile(self, model_path, build_dir):
        saved_model_path = os.path.join(build_dir, "tf_model")
        # if os.path.exists(saved_model_path):
        #     return
        # os.makedirs(saved_model_path, exist_ok=True)
        clear_and_make_dir(saved_model_path)

        tf_rep = prepare(onnx.load(model_path))
        tf_rep.export_graph(saved_model_path)

    def load_lib(self, run_dir):
        imported = tf.saved_model.load(os.path.join(run_dir, "tf_model"))
        f = imported.signatures["serving_default"]
        self.input_name = list(f.structured_input_signature[1].keys())[0]
        self.output_name = list(f.structured_outputs.keys())[0]
        self.input_shape = list(f.structured_input_signature[1].values())[0].shape
        self.f = f
        if self.input_data is not None and self.input_data.shape[1] != self.input_shape[1]:
            self.input_data = np.transpose(self.input_data, (0, 2, 3, 1))

    def run(self, run_dir):
        output = self.f(**{self.input_name: self.input_data})
        np.save(os.path.join(run_dir, "out.npy"), output[self.output_name])

    @staticmethod
    def get_output(run_dir):
        return np.load(os.path.join(run_dir, "out.npy"))

    @staticmethod
    def get_edge_value(run_dir):
        pass