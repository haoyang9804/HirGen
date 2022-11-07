import os
import shutil

import numpy as np


def select_manager(mode, has_input, has_two_output):
    if mode == "default":
        return DefaultManager()
    if has_input and not has_two_output:
        return DefaultManager()
    elif not has_input and not has_two_output:
        return NoInputManager()
    else:
        return None


class CompileManager:
    cc_file = None
    config_file = None

    @classmethod
    def before_compile(cls, compiler_path):
        shutil.copyfile(cls.config_file, os.path.join(compiler_path, "graph.config.pbtxt"))
        shutil.copyfile(cls.cc_file, os.path.join(compiler_path, "graph.cc"))

    @classmethod
    def predict(cls, lib, input_data):
        raise NotImplementedError


class DefaultManager(CompileManager):
    file_dir = os.path.dirname(__file__)
    config_file = os.path.join(file_dir, "default_run",
                               "default.config.pbtxt")
    cc_file = os.path.join(file_dir, "default_run",
                           "default_run.cc")

    @classmethod
    def predict(cls, lib, input_data):
        lib.run.argtypes = [
            np.ctypeslib.ndpointer(np.float32, ndim=4, shape=(4, 3, 32, 32), flags=('c', 'a')),
            np.ctypeslib.ndpointer(np.float32, ndim=2, shape=(4, 10), flags=('c', 'a', 'w')),
            np.ctypeslib.ctypes.c_int,
            np.ctypeslib.ctypes.c_int
        ]
        x = np.require(input_data, np.float32, ('c', 'a'))
        y = np.require(np.zeros((4, 10)), np.float32, ('c', 'a', 'w'))
        lib.run(x, y, x.size, y.size)
        return y


class NoInputManager(CompileManager):
    file_dir = os.path.dirname(__file__)
    config_file = os.path.join(file_dir, "no_in_one_out",
                               "no_in_one_out.config.txt")
    cc_file = os.path.join(file_dir, "no_in_one_out",
                           "no_in_one_out.cc")

    @classmethod
    def predict(cls, lib, input_data):
        lib.run.argtypes = [
            np.ctypeslib.ndpointer(np.float32, ndim=2, shape=(4, 10), flags=('c', 'a', 'w')),
            np.ctypeslib.ctypes.c_int
        ]
        y = np.require(np.zeros((4, 10)), np.float32, ('c', 'a', 'w'))
        lib.run(y, y.size)
        return y
