import onnx
import os

from mutation.fcb_mut import FCBMutator
from arg import init_config, mutation_args

args = init_config(mutation_args)

result_dir = os.path.join(args.result_saving_dir, args.model_name,
                          str(args.seed_number), args.mutation_method)
print("Result saving directory is:", result_dir)

seed_model = onnx.load(args.seed_model_path)

mutator = FCBMutator(seed_model, args.mutation_method, result_dir, args.input_data_path)

mutator.mutate(args.mutation_times, args.save_freq)
