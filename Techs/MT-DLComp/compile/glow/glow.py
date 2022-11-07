import os
import shutil
import subprocess

from utils.path_utils import change_ext
from compile.glow.glow_err import GlowError
from compile.glow.form_cpp import form_edge_diff_cpp
from compile.glow.compile_manager import select_manager
from compile.runner import Runner
from compile.compile_utils import execute_cmd

import numpy as np


class GlowRunner(Runner):
    def __init__(self, compiler_path, mode, cal_time):
        super().__init__(compiler_path, mode, cal_time)
        self.manager = select_manager(mode)
        # if mode == 'default':
        #     self.run_cpp_path = os.path.join(cur_dir, "run.cpp")
        #     self.form_cpp = trivial_prep_run_cpp
        # elif mode == 'node reduce':
        #     self.run_cpp_path = os.path.join(cur_dir, "node_reduce_run.cpp")
        #     self.form_cpp = trivial_prep_run_cpp
        # else:
        #     self.run_cpp_path = os.path.join(cur_dir, "edge_view.cpp")
        #     self.form_cpp = form_edge_diff_cpp

    def set_input(self, data_path):
        # if data_path.endswith(".bin"):
        #     self.data_path = data_path
        # else:
        #     self.data_path = np_to_bin(data_path)
        if self.channel_last:
            if data_path.endswith(".npy"):
                input_data = np.load(data_path)
            else:
                input_data = np.fromfile(data_path, dtype=np.float32)
            input_data = np.transpose(input_data, (0, 2, 3, 1))
            self.data_path = change_ext(data_path, 'bin')
            input_data.flatten().tofile(self.data_path)
        else:
            if data_path.endswith(".npy"):
                self.data_path = np_to_bin(data_path)
            else:
                self.data_path = data_path

    def load_lib(self, build_dir):
        pass

    def run(self, run_dir):
        glow_run(run_dir, self.data_path, self.cal_time)

        if self.cal_time:
            return get_run_time(os.path.join(run_dir, "time.bin"))

    def debug_compile(self, model_path, build_dir, debug_info_file):
        dump_ir(self.compiler_path, model_path, build_dir, debug_info_file)
        # self.form_cpp(build_dir, self.run_cpp_path)
        self.manager.form_run_cpp(build_dir)
        gcc_compile(build_dir)

    def compile(self, model_path, build_dir):
        glow_compile(self.compiler_path, model_path, build_dir)
        # self.form_cpp(build_dir, self.run_cpp_path)
        self.manager.form_run_cpp(build_dir)
        gcc_compile(build_dir)

    @staticmethod
    def get_output(run_dir):
        return np.fromfile(os.path.join(run_dir, "out.bin"), dtype=np.float32)

    @staticmethod
    def get_edge_value(run_dir):
        return np.fromfile(os.path.join(run_dir, "edge.bin"), dtype=np.float32)


def glow_compile(compiler_path, model_path, build_dir):
    r = execute_cmd(compiler_path, "-backend=CPU", f"-model={model_path}",
                    f"-emit-bundle={build_dir}", "-network-name=model")
    if r:
        raise GlowError(model_path, r)


def dump_ir(compiler_path, model_path, build_dir, dump_file):
    subprocess.run([compiler_path, "-backend=CPU", f"-model={model_path}",
                    f"-emit-bundle={build_dir}", "-network-name=model",
                    "-dump-ir-after-all-passes"],
                   stdout=open(dump_file, 'w'))


def gcc_compile(build_dir):
    last_wd = os.getcwd()
    os.chdir(build_dir)
    os.system("g++ -c run.cpp")
    r = execute_cmd("g++", "run.o", "model.o", "-o", "main", "-no-pie")
    os.chdir(last_wd)
    if r:
        raise GlowError(build_dir, r)


def glow_run(run_dir, data_path, cal_time=False):
    data_path = os.path.abspath(data_path)
    last_wd = os.getcwd()
    os.chdir(run_dir)
    if cal_time:
        r = execute_cmd("./main", data_path, "-t")
    else:
        r = execute_cmd("./main", data_path)
    os.chdir(last_wd)

    if r:
        raise RuntimeError(str(r))


def trivial_prep_run_cpp(build_dir, run_cpp_path):
    shutil.copyfile(run_cpp_path, os.path.join(build_dir, "run.cpp"))


def get_run_time(time_bin_path):
    with open(time_bin_path, 'r') as f:
        return float(f.readline())


def np_to_bin(np_data_path):
    bin_data_path = change_ext(np_data_path, "bin")
    data = np.load(np_data_path)
    data.flatten().tofile(bin_data_path)
    return bin_data_path
