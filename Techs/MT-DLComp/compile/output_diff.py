import numpy as np


def array_diff(mutant_output, seed_output):
    mutant_output = mutant_output.flatten()
    seed_output = seed_output.flatten()
    diff = np.abs(mutant_output - seed_output)
    max_abs_diff = np.max(diff)
    max_abs_idx = np.argmax(diff)
    abs_ori, abs_cur = seed_output[max_abs_idx], mutant_output[max_abs_idx]
    rel_diff = diff / (np.abs(seed_output) + 1e-9)
    max_rel_diff = np.max(rel_diff)
    max_rel_idx = np.argmax(rel_diff)
    rel_ori, rel_cur = seed_output[max_rel_idx], mutant_output[max_rel_idx]
    return "%f" % max_abs_diff, "%d" % max_abs_idx, \
           "%f" % abs_ori, "%f" % abs_cur, \
           "%f" % max_rel_diff, "%d" % max_rel_idx, \
           "%f" % rel_ori, "%f" % rel_cur


def write_output_diff(diff_file, diff_list, name_list):
    with open(diff_file, 'w') as f:
        f.write("\n".join([n + "$$$" + "$$$".join(d)
                           for n, d in zip(name_list, diff_list)]))


def compare_output(seed_output, mutants_output):
    return [array_diff(o, seed_output) for o in mutants_output]
