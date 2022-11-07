import os
import onnx
from onnx import shape_inference

from arg import init_config, delta_debugging_args
from reduce.dd import JudgeFail, DeltaDebugging
from reduce.graph_applier import GraphApplier
from reduce.node_applier import make_node_applier
from reduce.edges_diff import compare_onnx_compiler_edge_value
from compile.make_runner import make_runner

from utils.path_utils import clear_and_make_dir


args = init_config(delta_debugging_args)

mutant_models_dir = os.path.join(args.mutants_dir, args.model_name, str(args.seed_number),
                                 args.mutation_method, "models")
compiler_output_dir = os.path.join(args.compile_record_dir, args.compiler_name,
                                   args.model_name, str(args.seed_number),
                                   args.mutation_method)

result_dir = os.path.join(args.debug_dir, args.compiler_name, args.model_name,
                          str(args.seed_number), args.mutation_method, str(args.err_model_id))
print("Result saving directory is", result_dir)

reduced_model_path = os.path.join(result_dir, "model.onnx")
reduced_model_build_dir = os.path.join(result_dir, "build")
debug_info_file = os.path.join(result_dir, "debug_info.txt")
edge_diff_file = os.path.join(result_dir, "edge_diff.txt")
onnx_edge_value_dir = os.path.join(result_dir, "onnx_edges_value")

temp_dir = os.path.join(result_dir, "temp")


compile_fail, fault_output, ori_abs_diff, err_code = False, None, None, None
os.makedirs(temp_dir, exist_ok=True)
runner = make_runner(args.compiler_name, args.compiler_path,
                     'default', False)
output_diff_file = os.path.join(compiler_output_dir, "output_diff.txt")
with open(output_diff_file, 'r') as f:
    diff_line = [line for line in f.readlines()
                if line.split("$$$")[0] == str(args.err_model_id)]
if diff_line:
    compile_fail = False
    fault_output = runner.compile_run(
        os.path.join(mutant_models_dir, "%d.onnx" % args.err_model_id),
        temp_dir, args.input_data_path, view_edge=False
    )
    ori_abs_diff = float(diff_line[0].split("$$$")[1])
    print(f"Original abs diff is {ori_abs_diff}")
else:
    compile_fail = True
    compile_err_file = os.path.join(compiler_output_dir, "compilation_failure_models.txt")
    with open(compile_err_file, 'r') as f:
        err_line = [line for line in f.readlines()
                    if os.path.splitext(os.path.basename(line.split(" $$$ ")[0]))[0] ==
                    str(args.err_model_id)][0]
    err_code = err_line.strip().split(" $$$ ")[1]



def make_judge(runner, save_dir):
    judge = JudgeFail(runner, compile_fail, save_dir,
                      input_file=args.input_data_path, err_code=err_code,
                      fault_output=fault_output, ori_abs_diff=ori_abs_diff)
    return judge


def graph_reduce():
    print("============================")
    print("Running graph reduce")
    clear_and_make_dir(result_dir)
    mut_info_dir = os.path.join(args.mutants_dir, args.model_name, str(args.seed_number),
                                 args.mutation_method, "mut_info")

    runner = make_runner(args.compiler_name, args.compiler_path,
                         'default', False)

    judge = make_judge(runner, result_dir)

    applier = GraphApplier(mutant_models_dir, mut_info_dir, args.err_model_id)

    dd = DeltaDebugging(applier, judge)
    dd.run()


def node_reduce():
    print("============================")
    print("Running node reduce")

    applier = make_node_applier(reduced_model_path, args.input_data_path, temp_dir)

    runner = make_runner(args.compiler_name, args.compiler_path,
                         'node reduce', False)

    judge = make_judge(runner, result_dir)

    dd = DeltaDebugging(applier, judge)
    dd.run()

    model = onnx.load(reduced_model_path)
    model = shape_inference.infer_shapes(model)
    onnx.save(model, reduced_model_path)
    debug_compile(runner, reduced_model_path, reduced_model_build_dir,
                  debug_info_file)


def debug_compile(runner, model_path, build_dir, debug_output_file):
    method_list = [func for func in dir(runner) if callable(getattr(runner, func))]
    if "debug_compile" in method_list:
        runner.debug_compile(model_path, build_dir, debug_output_file)


def view_edges():
    print("============================")
    print("Running edge viewing")
    runner = make_runner(args.compiler_name, args.compiler_path,
                         'edge view', False)
    model = onnx.load(reduced_model_path)
    compare_onnx_compiler_edge_value(model, runner, fault_output,
                                     temp_dir, args.input_data_path,
                                     edge_diff_file, onnx_edge_value_dir)


graph_reduce()
node_reduce()
if not compile_fail:
    view_edges()
