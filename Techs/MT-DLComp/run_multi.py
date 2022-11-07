import os

from arg import init_config, multiple_input_args

from compile.mass_compile import MetaCompile, get_run_multiple_list
from compile.make_runner import make_runner
from utils.path_utils import file_name_no_ext

args = init_config(multiple_input_args)

model_name = file_name_no_ext(args.model_path)
result_dir = os.path.join(args.result_saving_dir, args.compiler_name, model_name)
print("Result saving directory is", result_dir)

runner = make_runner(args.compiler_name, args.compiler_path, "model zoo", True)
compiler = MetaCompile(runner, result_dir, True)

compiler.run_multiple(get_run_multiple_list(args.run_list, args.input_dir), args.model_path)
