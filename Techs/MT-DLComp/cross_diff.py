import os
import numpy as np
import tqdm

from arg import init_config, cross_compile_args
from cross_compile.glow_compile import select_compiler, batch_fixing


args = init_config(cross_compile_args)

glow_saving_dir = os.path.join(args.result_saving_dir, 'glow')
os.makedirs(glow_saving_dir, exist_ok=True)

others_save_dir = os.path.join(args.result_saving_dir, args.compared_name)
os.makedirs(others_save_dir, exist_ok=True)

fixed_batch_models_dir = os.path.join(args.result_saving_dir, 'batch_fixed_models')
os.makedirs(fixed_batch_models_dir, exist_ok=True)

diff_file = os.path.join(args.result_saving_dir, f"glow_{args.compared_name}_diff.txt")


output_diff = []

glow_compiler = select_compiler('glow', args.glow_compiler_path)
others_compiler = select_compiler(args.compared_name, None)

for model_file in os.listdir(args.model_zoo_dir):
    ori_model_path = os.path.join(args.model_zoo_dir, model_file)
    new_model_path = os.path.join(fixed_batch_models_dir, model_file)

    if not os.path.exists(new_model_path):
        batch_fixing(ori_model_path, new_model_path)

    model_name = os.path.splitext(model_file)[0]
    model_path = new_model_path

    if model_name != "inception-v1-9":
        continue

    print("===============================")
    print("Running %s" % model_name)

    glow_build_dir = os.path.join(glow_saving_dir, model_name)
    others_build_dir = os.path.join(others_save_dir, model_name)
    os.makedirs(others_build_dir, exist_ok=True)

    glow_compiler.compile(model_path, glow_build_dir)
    others_compiler.compile(model_path, others_build_dir)

    glow_compiler.load_lib(glow_build_dir)
    others_compiler.load_lib(others_build_dir)

    cnt_1, cnt_2, cnt_3, cnt_4 = 0, 0, 0, 0
    max_max_diff = -1
    for data_file in tqdm.tqdm(os.listdir(args.input_data_dir)):
        data_path = os.path.join(args.input_data_dir, data_file)

        glow_compiler.set_input(data_path)
        glow_compiler.run(glow_build_dir)
        glow_out = glow_compiler.get_output(glow_build_dir)

        others_compiler.set_input(data_path)
        others_compiler.run(others_build_dir)
        others_out = others_compiler.get_output(others_build_dir)

        max_diff = np.max(others_out.flatten() - glow_out)
        if max_diff > 1e-1:
            cnt_1 += 1
        if max_diff > 1e-2:
            cnt_2 += 1
        if max_diff > 1e-3:
            cnt_3 += 1
        if max_diff > 1e-4:
            cnt_4 += 1
        max_max_diff = max(max_max_diff, max_diff)
    print(f">1e-1: {cnt_1}, >1e-2: {cnt_2}, >1e-3: {cnt_3}, >1e-4: {cnt_4}, max_diff: {max_max_diff}")
