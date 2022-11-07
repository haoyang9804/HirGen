import copy

import numpy as np


def conv_node(input_shape, kernel_shape):
    attr_dict = {
        'dilations': [1, 1],
        'kernel_shape': kernel_shape[2:],
        'strides': [1, 1]
    }
    out_shape = [input_shape[0], kernel_shape[0]]
    for in_dim, k_dim in zip(input_shape[2:], kernel_shape[2:]):
        out_dim = in_dim - k_dim + 1
        out_shape.append(out_dim)

    return attr_dict, out_shape


def pad_node(src_shape, tgt_shape, mode='constant'):
    pad_2d = []
    for s, t in zip(src_shape, tgt_shape):
        num_pads = t - s
        l_pads = num_pads // 2
        r_pads = num_pads - l_pads
        pad_2d.append((l_pads, r_pads))
    pads = list(zip(*pad_2d))
    pads = tuple(pads[0] + pads[1])
    return {'mode': mode, 'pads': pads}


def reduce_node(src_shape, keep_dims, tgt_rank=2):
    if keep_dims:
        new_shape = list(src_shape[:tgt_rank])
        new_shape.extend([1 for _ in range(tgt_rank, len(src_shape))])
    else:
        new_shape = src_shape[:tgt_rank]
    axes = [i for i in range(tgt_rank, len(src_shape))]
    return {'axes': np.array(axes, dtype=np.int32),
            'keepdims': int(keep_dims)}, tuple(new_shape)


def unsqueeze_node(ori_shape, r):
    new_shape = list(copy.copy(ori_shape)) + [1 for _ in range(len(ori_shape), r)]
    return {'axes': np.array([i for i in range(len(ori_shape), r)])}, tuple(new_shape)


def slice_node(ori_shape, new_shape):
    assert len(ori_shape) == len(new_shape)
    r = len(ori_shape)
    for i in range(0, r):
        if ori_shape[i] < new_shape[i]:
            raise Exception("Original shape should be greater than the new one")
    # steps = [(s - 1) // (t - 1) if t != 1 else 1
    #          for (s, t) in zip(ori_shape, new_shape)]
    steps = [1 for _ in range(0, r)]
    ends = [(new_shape[i] - 1) * steps[i] + 1 for i in range(0, r)]
    # return {'starts': np.array([0 for _ in range(0, r)], dtype=np.int32),
    #         'ends': np.array(ends, dtype=np.int32),
    #         'axes': np.array([i for i in range(0, r)], dtype=np.int32),
    #         'steps': np.array(steps, dtype=np.int32),
    #         }
    return {'axes': [i for i in range(0, r)],
            'ends': ends,
            'starts': [0 for _ in range(0, r)]
            }
