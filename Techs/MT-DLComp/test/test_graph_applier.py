import copy
from unittest import TestCase

from mutation.node_gen import NodeChainGen
from reduce.dd import DeltaDebugging
from reduce.reduce_utils import topological_sort
from reduce.graph_applier import GraphApplier
from utils.onnx_utils import print_onnx_graph


class JudgeFail:
    @staticmethod
    def remain_failed(model):
        try:
            topological_sort(model.graph.node)
        except Exception:
            return True
        return False


class WrongApplier(GraphApplier):
    def apply(self, delta_ids):
        model = copy.copy(self.seed_model)
        self.reset()
        gen = NodeChainGen(self.max_node_idx + 1, self.max_edge_idx + 1)
        for delta_id in delta_ids:
            delta = self.delta_list[delta_id]
            delta.apply(model.graph.node, self.name_info_mapping, gen)

        return model


def delta_debug():
    delta_ids = [
        1, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
        69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
        88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
        105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
        120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
        135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
        150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
        165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
        180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
        195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
        210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
        225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
        255, 256, 257, 258, 259, 260]

    applier = WrongApplier(
        "/export/d3/dwxiao/mutants/resnet18/99131411/hybrid/models",
        "/export/d3/dwxiao/mutants/resnet18/99131411/hybrid/mut_info",
        261
    )

    judge = JudgeFail()

    dd = DeltaDebugging(applier, judge)

    err_inducing_ids = dd.apply(delta_ids, [])
    print("=================")
    print(f"Error inducing ids: {err_inducing_ids}")
    # Error inducing ids: [1, 116, 245]
    # in: ['197'], out: ['3315'], name: Unsqueeze_3372
    # in: ['3315'], out: ['3316'], name: Slice_3373
    # in: ['3316', '3316'], out: ['3318'], name: Mul_3375
    # in: ['1390', '3316'], out: ['3319'], name: Mul_3376
    # in: ['3319', '260'], out: ['3320'], name: Mul_3377
    # in: ['3317', '3318'], out: ['3321'], name: Add_3378
    # in: ['3321', '3320'], out: ['3322'], name: Add_3379
    # in: ['3322', '259'], out: ['3323'], name: Add_3380
    # in: ['3323'], out: ['3324'], name: Neg_3381
    # in: ['3324'], out: ['3325'], name: Relu_3382
    # in: ['3326', '3325'], out: ['3327'], name: Mul_3384
    # in: ['3327'], out: ['3524'], name: Slice_3596
    # in: ['3524', '3328'], out: ['1779'], name: Add_3385
    # in: ['1779', '1304'], out: ['1780'], name: Sub_1693
    # in: ['1780', '1783'], out: ['1784'], name: Mul_1698
    # in: ['1784'], out: ['1785'], name: Pad_1699
    # in: ['1785'], out: ['3522'], name: ReduceMax_3594
    # in: ['3522'], out: ['3523'], name: Pad_3595
    # in: ['3523', '1786'], out: ['197'], name: Add_1700 (197 wrong)
    # in: ['197', 'fc.weight', 'fc.bias'], out: ['output'], name: Gemm_53


class TestGraphApplier(TestCase):
    def test_apply(self):
        err_ids = [1, 116, 245]
        applier = WrongApplier(
            "/export/d3/dwxiao/mutants/resnet18/99131411/hybrid/models",
            "/export/d3/dwxiao/mutants/resnet18/99131411/hybrid/mut_info",
            261
        )

        model = applier.apply(err_ids)
        print_onnx_graph(model)
