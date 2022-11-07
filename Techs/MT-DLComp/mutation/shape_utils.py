import onnx

from utils.onnx_utils import get_dim


def get_type(t):
    return t.type.tensor_type.elem_type


def is_float32(onnx_edge):
    return get_type(onnx_edge) == onnx.TensorProto.FLOAT


def shape_match(edge_list, start_idx, match_edge, shape_constrain):
    matched = []
    for edge in edge_list[start_idx + 1:]:
        if match_edge.type.tensor_type.elem_type != \
                edge.type.tensor_type.elem_type:
            continue
        out_dims = get_dim(edge)
        match_dim = get_dim(match_edge)
        if not shape_constrain(match_dim, out_dims):
            continue
        matched.append(edge.name)
    return matched


def broadcast_constrain(e1, e2):
    s1 = get_dim(e1)
    s2 = get_dim(e2)
    if len(s1) < len(s2):
        t = s1
        s1 = s2
        s2 = t
    for i in range(1, len(s2)):
        if s1[-i] != s2[-i] and s1[-i] != 1 and s2[-i] != 1:
            return False
    return True


def get_common_shape(shapes, broadcast=True):
    """

    :param shapes: shape list to be matched
    :param broadcast: whether broadcast is allowed
    :return: output shape after broadcasting if broadcast=True
    else minimum dim value but maximum rank
    """
    r = max(len(s) for s in shapes)
    common_s = []
    for i in range(0, r):
        if broadcast:
            dim_i = [s[i] for s in shapes if len(s) > i and s[i] != 1]
        else:
            dim_i = [s[i] for s in shapes if len(s) > i]
        if not dim_i:
            min_dim = 1
        else:
            min_dim = min(dim_i)
        common_s.append(min_dim)
    return tuple(common_s)


def get_slice_shape(src_shape, tgt_shape, broadcast=True):
    """
    The src_shape and tgt_shape must be of the same rank
    :param broadcast: whether there's broadcast constrain
    :param src_shape: original shape
    :param tgt_shape: shape to be matched to
    :return: slice shape
    """
    assert len(src_shape) == len(tgt_shape)
    if broadcast:
        return tuple([s if s == 1 or t == 1 else min(s, t)
                      for s, t in zip(src_shape, tgt_shape)])
    else:
        return tuple([min(s, t) for s, t in zip(src_shape, tgt_shape)])


def get_pad_shape(src_shape, tgt_shape, broadcast):
    assert len(src_shape) == len(tgt_shape)
    if broadcast:
        return tuple([s if s == 1 or t == 1 else max(s, t)
                      for s, t in zip(src_shape, tgt_shape)])
    else:
        return tuple([max(s, t) for s, t in zip(src_shape, tgt_shape)])
