import os
import shutil
import numpy as np

import tqdm

from compile.output_diff import array_diff, write_output_diff
from compile.time_utils import time_iterator
from compile.compile_err import CompilationError
from utils.path_utils import clear_and_make_dir, file_name_no_ext, get_ext


class MetaCompile:
    def __init__(self, runner, result_dir, retain_build):
        self.runner = runner

        self.build_root_dir = os.path.join(result_dir, "build")
        self.output_dir = os.path.join(result_dir, "output")
        time_rec_dir = os.path.join(result_dir, "time_record")
        self.compile_time_file = os.path.join(time_rec_dir, "compile_time.txt")
        self.run_time_file = os.path.join(time_rec_dir, "run_time.txt")
        self.diff_file_path = os.path.join(result_dir, "output_diff.txt")
        self.err_summary_file = os.path.join(result_dir, "compilation_failure_models.txt")
        self.err_full_info_dir = os.path.join(result_dir, "error_info")

        # self.frac_compile = frac_compile

        self.retain_build = retain_build

        self.output = {}

        os.makedirs(self.build_root_dir, exist_ok=True)
        os.makedirs(time_rec_dir, exist_ok=True)

    def get_build_dir(self, model_path):
        model_name = file_name_no_ext(model_path)
        return os.path.join(self.build_root_dir, model_name) if self.retain_build \
            else self.build_root_dir

    def handle_compilation_error(self, e: CompilationError, model_name, build_dir):
        with open(self.err_summary_file, 'a') as f:
            f.write(f"{e.model_path} $$$ {e.err_code}\n")
        if not os.path.exists(self.err_full_info_dir):
            os.makedirs(self.err_full_info_dir)
        with open(os.path.join(self.err_full_info_dir, "%s.txt" % model_name), 'w') as f:
            f.write(e.err_info)
        shutil.rmtree(build_dir)

    def handle_run_error(self, e: RuntimeError, build_dir):
        with open(self.err_summary_file, 'a') as f:
            f.write(f"{build_dir} $$$ {str(e)}\n")
        shutil.rmtree(build_dir)

    def compile_run(self, compile_list, input_file):
        it = time_iterator(compile_list, [self.compile_time_file, self.run_time_file])

        self.runner.set_input(input_file)

        clear_and_make_dir(self.output_dir)

        for model_path in tqdm.tqdm(it):
            model_id = file_name_no_ext(model_path)

            build_dir = self.get_build_dir(model_path)
            clear_and_make_dir(build_dir)

            failed = self.cal_compile_time(it, model_path, build_dir)
            if failed:
                continue

            self.runner.prepare_run(build_dir)
            failed = self.cal_run_time(it, build_dir, input_file)
            if failed:
                continue

            out = self.get_output(build_dir)
            print(out)
            self.output.update({model_id: out})

            if not self.retain_build:
                np.save(os.path.join(self.output_dir, "%s.npy" % model_id), out)
                shutil.rmtree(build_dir)

    def run_only(self, compile_list, input_file):
        it = time_iterator(compile_list, [self.run_time_file])
        for model_path in tqdm.tqdm(it):
            run_dir = self.get_build_dir(model_path)
            self.runner.prepare_run(run_dir)
            self.cal_run_time(it, run_dir, input_file)

    def run_multiple(self, input_files, model_path):
        run_dir = self.build_root_dir

        # clear_and_make_dir(run_dir)
        # self.compile(model_path, run_dir)

        it = time_iterator(input_files, [self.run_time_file])

        self.runner.prepare_run(run_dir)
        for in_file in tqdm.tqdm(it):
            self.cal_run_time(it, run_dir, in_file)

    def compile_only(self, compile_list):
        it = time_iterator(compile_list, [self.compile_time_file])
        for model_path in tqdm.tqdm(it):
            build_dir = self.get_build_dir(model_path)
            clear_and_make_dir(build_dir)
            self.cal_compile_time(it, model_path, build_dir)

    def compile(self, model_path, build_dir):
        try:
            self.runner.build(model_path, build_dir)
        except CompilationError as e:
            self.handle_compilation_error(e, model_path, build_dir)
            return True
        else:
            return False

    def cal_run_time(self, iterator, build_dir, input_file):
        failed = self.run(build_dir, input_file)
        if failed:
            return True
        iterator.set_time(self.runner.get_run_time())
        return False

    def cal_compile_time(self, iterator, model_path, build_dir):
        return iterator.cal_time(lambda : self.compile(model_path, build_dir))

    def run(self, build_dir, input_file):
        try:
            self.runner.run_with_input(build_dir, input_file)
        except RuntimeError as e:
            self.handle_run_error(e, build_dir)
            return True
        else:
            return False

    def compare_output(self):
        name_list = [model_name for model_name in self.output.keys() if model_name != 'seed']
        name_list.sort(key=lambda x: int(x))

        seed_output = self.output['seed']

        diff_list = [array_diff(self.output[name], seed_output) for name in name_list]

        write_output_diff(self.diff_file_path, diff_list, name_list)

    def get_output(self, build_dir):
        return self.runner.get_output(build_dir)


def get_compile_list(compile_list, onnx_model_dir):
    if not compile_list:
        model_names = [os.path.splitext(file_name)[0]
                       for file_name in os.listdir(onnx_model_dir)
                       if file_name != 'seed.onnx' and file_name[-5:] == ".onnx"]
        model_names.sort(key=lambda x: int(x))
        # model_names = model_names[::self.frac_compile]
        model_names.append('seed')
    else:
        model_names = [str(name) for name in compile_list]
        if 'seed' not in model_names:
            model_names.append('seed')
    return [os.path.join(onnx_model_dir, "%s.onnx" % n) for n in model_names]


def get_run_multiple_list(run_list, input_dir):
    if not run_list:
        return [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                if get_ext(f) == ".npy"]
    else:
        return [os.path.join(input_dir, "%d.npy" % f) for f in run_list]


def compiler_run(runner, onnx_model_dir,
                 data_path, result_dir, retain_result, mode, compile_list):
    meta_compiler = MetaCompile(runner, result_dir, retain_result)
    if mode == "compile_run":
        meta_compiler.compile_run(get_compile_list(compile_list, onnx_model_dir), data_path)
        meta_compiler.compare_output()
    elif mode == "compile":
        meta_compiler.compile_only(get_compile_list(compile_list, onnx_model_dir))
    else:
        meta_compiler.run_only(get_compile_list(compile_list, onnx_model_dir), data_path)
