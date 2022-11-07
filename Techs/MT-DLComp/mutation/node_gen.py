import onnx

import utils.onnx_utils
from mutation import mutate_utils, attr_gen, shape_utils
from mutation.edge_node import EdgeNode


class ElementGen:
    def __init__(self, next_node_idx, next_edge_idx):
        self.node_id = next_node_idx
        self.edge_id = next_edge_idx

    def new_node_name(self, node_type):
        self.node_id += 1
        return "%s_%d" % (node_type, self.node_id - 1)

    def new_edge_name(self):
        self.edge_id += 1
        return str(self.edge_id - 1)

    @staticmethod
    def new_tensor_name(node_name, attr_name):
        return "%s_%s" % (node_name, attr_name)

    def new_node(self, node_type, input_edges: list, **kwargs):
        node = onnx.helper.make_node(
            node_type,
            input_edges,
            [self.new_edge_name()],
            self.new_node_name(node_type),
            **kwargs
        )
        return node

    def new_node_specifying_output(self, node_type, input_edges: list,
                                   output_edge: str, **kwargs):
        node = onnx.helper.make_node(
            node_type,
            input_edges,
            [output_edge],
            self.new_node_name(node_type),
            **kwargs
        )
        return node

    @staticmethod
    def new_tensor(np_val, node_name, attr_name):
        data_type = mutate_utils.numpy_onnx_type_mapping(np_val.dtype)

        return onnx.helper.make_tensor(
            name=ElementGen.new_tensor_name(node_name, attr_name),
            data_type=data_type,
            dims=np_val.shape,
            vals=np_val.flatten()
        )


class NodeGen:
    def __init__(self, st_node_idx, st_edge_idx):
        self.elem_gen = ElementGen(st_node_idx, st_edge_idx)

    def gen_slice(self, input_name, src_shape, tgt_shape, broadcast=False):
        slice_shape = shape_utils.get_slice_shape(
            src_shape, tgt_shape, broadcast)
        if slice_shape != src_shape:
            op_type = 'Slice'
            slice_dict = attr_gen.slice_node(src_shape, slice_shape)
            slice_node = self.make_node(op_type, input_name, slice_dict)
            return slice_node, slice_shape, slice_node.output[0]
        else:
            return None, src_shape, input_name

    def gen_unsqueeze(self, input_name, src_shape, tgt_rank):
        if len(src_shape) < tgt_rank:
            axes, unsqueeze_shape = attr_gen.unsqueeze_node(src_shape, tgt_rank)
            node = self.make_node('Unsqueeze', input_name, axes)
            return node, unsqueeze_shape, node.output[0]
        else:
            return None, src_shape, input_name

    def gen_reduce(self, input_name, in_shape,
                   reduce='mean', keep_dims=False, rank=2):
        assert reduce.lower() in ['mean', 'max', 'min', 'l1', 'l2', 'sum']
        op_type = "Reduce%s%s" % (reduce[0].upper(), reduce[1:])
        if len(in_shape) > rank:
            attr, reduce_shape = attr_gen.reduce_node(in_shape, keep_dims, rank)
            node = self.make_node(op_type, input_name, attr)
            return node, reduce_shape, node.output[0]
        else:
            return None, in_shape, input_name

    def gen_pad(self, in_name, src_shape, tgt_shape,
                broadcast, mode='constant'):
        pad_shape = shape_utils.get_pad_shape(src_shape, tgt_shape, broadcast)
        if pad_shape != src_shape:
            pad_dict = attr_gen.pad_node(src_shape, pad_shape, mode)
            node = self.make_node('Pad', in_name, pad_dict)
            return node, pad_shape, node.output[0]
        else:
            return None, pad_shape, in_name

    def gen_constant(self, val):
        attr_dict = {'value': self.elem_gen.new_tensor(
            val, self.elem_gen.new_node_name('Constant'), 'tensor'
        )}
        return self.make_node('Constant', [], attr_dict)

    def gen_conv(self, in_name, in_shape, kernel_name, kernel_shape):
        attr, out_shape = attr_gen.conv_node(in_shape, kernel_shape)
        node = self.make_node('Conv', [in_name, kernel_name], attr)
        return node, out_shape, node.output[0]

    def make_node(self, op_type, input_edges_name, attr_dict=None):
        input_edges_name = mutate_utils.convert2iter(input_edges_name)
        if attr_dict:
            node = self.elem_gen.new_node(op_type, input_edges_name, **attr_dict)
        else:
            node = self.elem_gen.new_node(op_type, input_edges_name)
        return node


class NodeChainGen(NodeGen):
    def make_conv(self, new_edges, in_edge, np_kernel_val):
        weight = self.make_constant(np_kernel_val)
        new_edges.append(weight)

        conv_node, conv_out_shape, conv_out_name = self.gen_conv(
            in_edge.name, in_edge.shape, weight.name, weight.shape
        )
        conv_edge = EdgeNode(conv_out_name, conv_out_shape, conv_node,
                             in_edge.zero or mutate_utils.is_val_zero(np_kernel_val))
        new_edges.append(conv_edge)
        return conv_edge

    def make_unsqueeze(self, new_edges, in_edge, tgt_rank):
        if len(in_edge.shape) < tgt_rank:
            rank_node, edge_shape, edge_name = self.gen_unsqueeze(
                in_edge.name, in_edge.shape, tgt_rank)
            edge = EdgeNode(edge_name, edge_shape, rank_node)
            new_edges.append(edge)
            return edge
        return in_edge

    def make_reduce(self, new_edges: list, input_edge,
                    reduce='mean', keep_dims=False, rank=2):
        node, edge_shape, edge_name = self.gen_reduce(
            input_edge.name, input_edge.shape, reduce, keep_dims, rank
        )
        if node:
            new_edge = EdgeNode(edge_name, edge_shape, node, input_edge.zero)
            new_edges.append(new_edge)
            return new_edge
        else:
            return input_edge

    def make_constant(self, val):
        new_node = self.gen_constant(val)
        new_edge = EdgeNode(new_node.output[0], val.shape, new_node, False)
        return new_edge

    def make_subs_add(self, subs_edge, dead_edge):
        ori_output_name = subs_edge.name
        subs_edge = self.substitute_edge(subs_edge)
        add_node = self.elem_gen.new_node_specifying_output(
            'Add', [dead_edge.name, subs_edge.name],
            ori_output_name
        )
        add_edge = EdgeNode(ori_output_name, subs_edge.shape, add_node,
                            subs_edge.zero and dead_edge.zero)
        return subs_edge, add_edge

    def substitute_edge(self, substituted_edge):
        node = substituted_edge.def_node
        new_output_name = self.elem_gen.new_edge_name()
        mutate_utils.replace_node_output(node, new_output_name)
        new_output_edge = EdgeNode(
            new_output_name, substituted_edge.shape, node, substituted_edge.zero
        )
        substituted_edge.def_node = None
        return new_output_edge

    def make_edge_node(self, op_type, in_edges, out_shape, zero):
        in_edges = mutate_utils.convert2iter(in_edges)
        node = self.make_node(op_type, [e.name for e in in_edges], None)
        edge = EdgeNode(node.output[0], out_shape, node, zero)
        return edge

    def make_multi_input_node(self, op_type, in_edges, broadcast, out_zero):
        edges, node_in_edges, common_shape = self.bilateral_shape_matching(
            in_edges, broadcast)

        agg_edge = self.make_edge_node(
            op_type, node_in_edges, common_shape, out_zero)
        edges.append(agg_edge)
        return edges

    def unilateral_shape_matching(self, new_edges, in_edge,
                                  tgt_shape, broadcast):

        edge = self.match_rank(new_edges, in_edge, len(tgt_shape))
        edge_name, edge_shape = edge.name, edge.shape

        slice_node, edge_shape, edge_name = self.gen_slice(
            edge_name, edge_shape, tgt_shape, False)
        if slice_node:
            edge = EdgeNode(edge_name, edge_shape, slice_node)
            new_edges.append(edge)

        pad_node, edge_shape, edge_name = self.gen_pad(
            edge_name, edge_shape, tgt_shape, broadcast)
        if pad_node:
            edge = EdgeNode(edge_name, edge_shape, pad_node)
            new_edges.append(edge)

        return edge

    def match_rank(self, new_edges, in_edge, tgt_rank):
        in_name, src_shape = in_edge.name, in_edge.shape
        if len(src_shape) < tgt_rank:
            edge = self.make_unsqueeze(new_edges, in_edge, tgt_rank)
        elif len(src_shape) > tgt_rank:
            edge = self.make_reduce(new_edges, in_edge, 'max', rank=tgt_rank)
        else:
            edge = in_edge
        return edge

    def bilateral_shape_matching(self, in_edges: list, broadcast):
        shape_list = [e.shape for e in in_edges]
        common_shape = shape_utils.get_common_shape(shape_list, broadcast)
        new_edges, out_edges = [], []
        for in_edge in in_edges:
            out_edge = in_edge
            node, out_shape, out_name = self.gen_unsqueeze(
                in_edge.name, in_edge.shape, len(common_shape))
            if node:
                out_edge = EdgeNode(out_name, out_shape, node, in_edge.zero)
                new_edges.append(out_edge)

            node, out_shape, out_name = self.gen_slice(
                out_name, out_shape, common_shape, broadcast)
            if node:
                out_edge = EdgeNode(out_name, out_shape, node, in_edge.zero)
                new_edges.append(out_edge)

            out_edges.append(out_edge)

        return new_edges, out_edges, common_shape


def make_node_chain_generator(model):
    max_node_idx = utils.onnx_utils.get_max_node_idx(model.graph)
    max_edge_idx = utils.onnx_utils.get_max_edge_idx(model.graph)
    return NodeChainGen(max_node_idx + 1, max_edge_idx + 1)
