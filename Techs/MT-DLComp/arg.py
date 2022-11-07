import argparse
import random
import numpy as np


def mutation_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_name", choices=['resnet18', 'vgg11', 'mobilenet', 'inception'], default="resnet18", type=str)

    arg_parser.add_argument("--result_saving_dir", default="../mutants", type=str)

    arg_parser.add_argument("--seed_model_path", default="../data/resnet18.onnx", type=str)
    arg_parser.add_argument("--input_data_path", default="../data/data.npy", type=str)

    arg_parser.add_argument("--mutation_method", choices=['universal', 'per_input', 'hybrid'],
                            default="hybrid", type=str)

    arg_parser.add_argument("--mutation_times", default=1000, type=int)

    arg_parser.add_argument("--save_freq", default=10, type=int)

    arg_parser.add_argument("--seed_number", default=1, type=int)

    return arg_parser


def mutants_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--mutants_dir", default="../mutants", type=str)

    arg_parser.add_argument("--model_name", choices=['resnet18', 'vgg11', 'mobilenet', 'inception'], default="resnet18", type=str)
    arg_parser.add_argument("--seed_number", default=1, type=int)

    arg_parser.add_argument("--mutation_method", choices=['universal', 'per_input', 'hybrid'],
                            default="hybrid", type=str)

    arg_parser.add_argument("--compiler_name", choices=['glow', 'tvm', 'xla', 'tensorflow', 'glow-zoo', 'onnx'], default="glow", type=str)
    arg_parser.add_argument("--compiler_path", default="~/glow-build/bin/model-compiler", type=str)
    # arg_parser.add_argument("--compiler_path", default="~/tensorflow", type=str)

    arg_parser.add_argument("--compile_record_dir", default="../compile_record", type=str)
    arg_parser.add_argument("--input_data_path", default="../data/data.npy", type=str)

    return arg_parser


def compilation_args():
    arg_parser = mutants_args()

    arg_parser.add_argument("--compile_list", nargs="+", default=[], type=int)
    arg_parser.add_argument("--retain_build", default=False, type=bool)
    arg_parser.add_argument("--mode", default="compile_run", type=str)
    # arg_parser.add_argument("--frac_compile", default=1, type=int)

    return arg_parser


def delta_debugging_args():
    arg_parser = mutants_args()

    arg_parser.add_argument("--err_model_id", type=int)
    arg_parser.add_argument("--debug_dir", default="../debug")

    return arg_parser


def cross_compile_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--model_zoo_dir", default="../model_zoo/classification/models", type=str)
    arg_parser.add_argument("--result_saving_dir", default="../model_zoo/classification", type=str)

    arg_parser.add_argument("--glow_compiler_path", default="../glow-build/bin/model-compiler", type=str)

    arg_parser.add_argument("--compared_name", choices=['tvm', 'tensorflow'], default="tensorflow", type=str)

    arg_parser.add_argument("--input_data_dir", default="../data/processed_imagenet", type=str)

    arg_parser.add_argument("--seed_number", default=1, type=int)

    return arg_parser


def multiple_input_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--model_path", default="../model_zoo/classification/batch_fixed_models/vgg16-7.onnx", type=str)
    arg_parser.add_argument("--result_saving_dir", default="../time_record")

    arg_parser.add_argument("--input_dir", default="../data/processed_imagenet", type=str)
    arg_parser.add_argument("--run_list", nargs="+", default=[1], type=int)

    arg_parser.add_argument("--compiler_name", choices=['glow', 'tvm', 'xla', 'onnx'], default="glow",
                            type=str)
    arg_parser.add_argument("--compiler_path", default="~/glow-build/bin/model-compiler", type=str)

    arg_parser.add_argument("--seed_number", default=1, type=int)

    return arg_parser

def init_config(arg_init):
    arg_parser = arg_init()
    args = arg_parser.parse_args()
    random.seed(args.seed_number)
    np.random.seed(args.seed_number)
    return args
