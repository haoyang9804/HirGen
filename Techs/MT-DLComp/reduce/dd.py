import onnx
import numpy as np

from reduce import reduce_utils
from compile.compile_err import CompilationError


class JudgeFail:
    def __init__(self, runner, compile_fail, save_dir,
                 input_file=None, err_code=None,
                 fault_output=None, ori_abs_diff=None, allowed_dev=0.1):
        self.save_dir = save_dir
        self.runner = runner
        self.id = 0

        self.runner.set_input(input_file)
        self.input_file = input_file

        self.check_failed = self.check_compile_failed if compile_fail else self.check_run_failed

        self.err_code = err_code

        self.fault_output = fault_output
        if ori_abs_diff is not None:
            self.threshold = ori_abs_diff * allowed_dev

    def compile(self, model_path, build_dir):
        self.runner.compile(model_path, build_dir)

    def run(self, run_dir):
        if hasattr(self.runner, "load_lib"):
            self.runner.load_lib(run_dir)
        self.runner.set_input(self.input_file)
        self.runner.run(run_dir)

    def check_compile_failed(self, model_path, build_dir):
        try:
            self.compile(model_path, build_dir)
        except CompilationError as e:
            return e.get_err_code() == self.err_code
        return False

    def check_run_failed(self, model_path, build_dir):
        try:
            self.compile(model_path, build_dir)
        except CompilationError:
            return False
        try:
            self.run(build_dir)
        except RuntimeError:
            return False
        return self.is_close(build_dir)

    def remain_failed(self, model):
        self.id += 1
        print("=================================")
        print("Running %s:" % self.id)
        model_path, build_dir = reduce_utils.prepare_run_dir(self.save_dir)

        onnx.save(model, model_path)

        return self.check_failed(model_path, build_dir)

    def is_close(self, run_dir):
        model_output = self.runner.get_output(run_dir)
        if True in np.isnan(self.fault_output):
            return True in np.isnan(model_output)
        max_abs_diff = np.max(np.abs(model_output - self.fault_output))
        print("Max absolute diff is %f" % max_abs_diff)
        return max_abs_diff < self.threshold


class DeltaDebugging:
    def __init__(self, applier, judge):
        self.applier = applier
        self.judge = judge

    def remain_failed(self, delta_ids):
        print("Applying deltas:", delta_ids)
        # Quick fix. Do not rely on it!
        try:
            model = self.applier.apply(delta_ids)
        except Exception:
            print("There's a bug in applier.apply(), let's just skip it hhh~~~")
            return False
        r = self.judge.remain_failed(model)
        if r:
            print("Model remain failed")
        else:
            print("Model passed")
        return r

    def resolved(self, delta_ids):
        return True
        print("Checking dependency of", delta_ids)
        r = self.ds.check_dep(delta_ids)
        if r:
            print("Check passed")
        else:
            print("Check failed")
        return r

    def apply(self, delta_ids, remaining):
        print("=============================")
        print("Delta_ids:", delta_ids)
        print("Remaining:", remaining)

        if len(delta_ids) <= 1:
            return delta_ids

        half_len = len(delta_ids) // 2
        left = delta_ids[:half_len]
        right = delta_ids[half_len:]

        print("Left:", left)
        print("Right:", right)

        left_r = reduce_utils.union_sort(left, remaining)
        right_r = reduce_utils.union_sort(right, remaining)

        if self.resolved(left_r):
            if self.remain_failed(left_r):  # Left failed
                return self.apply(left, remaining)  # Search left
            else:  # Left passed
                if self.resolved(right_r):
                    if self.remain_failed(right_r):  # Left passed, right failed
                        return self.apply(right, remaining)
                    else:  # Both passed
                        left_inducing = self.apply(left, right_r)
                        left_r = reduce_utils.union_sort(
                            left_inducing, remaining)
                        right_inducing = self.apply(right, left_r)
                        return reduce_utils.union_sort(left_inducing,
                                                       right_inducing)
                else:  # Left passed, right unresolved
                    return self.apply(right, left_r)
        else:  # Left unresolved
            # An impossible case, otherwise it requires re-partition
            raise Exception("Left cannot fail")

    def run(self):
        error_inducing = self.apply(list(self.applier.valid_range()), [])
        self.apply_err_inducing(error_inducing)

    def apply_err_inducing(self, error_inducing):
        print("Error-inducing deltas:", error_inducing)
        dep_chain = error_inducing
        # dep_chain = self.ds.get_dep_chain(error_inducing)
        # print("Error-inducing deltas with their dependencies", dep_chain)
        self.remain_failed(dep_chain)
