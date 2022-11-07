import onnx


class ModelDifferentError(Exception):
    def __init__(self, n_n1, n_n2):
        self.n_n1 = n_n1
        self.n_n2 = n_n2

    def __str__(self):
        return f"Number of nodes in model 1: {self.n_n1}; " \
               f"Number of nodes in model 2: {self.n_n2}"


class NodeDifferentError(Exception):
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    def __str__(self):
        return f"n1: in: {str(list(self.node1.input))}, out: {str(list(self.node1.output))}" \
               f", name: {self.node1.name}; " \
               f"n2: in: {str(list(self.node2.input))}, out: {str(list(self.node2.output))}" \
               f", name: {self.node2.name}"


def check_model_same(m1, m2):
    n_nodes_1 = len(m1.graph.node)
    n_nodes_2 = len(m2.graph.node)
    if n_nodes_1 != n_nodes_2:
        raise ModelDifferentError(n_nodes_1, n_nodes_2)

    for i in range(n_nodes_1):
        n1 = m1.graph.node[i]
        n2 = m2.graph.node[i]
        if n1.name != n2.name:
            raise NodeDifferentError(n1, n2)
        if list(n1.input) != list(n2.input):
            raise NodeDifferentError(n1, n2)
        if list(n1.output) != list(n2.output):
            raise NodeDifferentError(n1, n2)
    print("Success")


def test_model_same():
    m1_path = "/export/d3/dwxiao/temp/261.onnx"
    m2_path = "/export/d3/dwxiao/mutants/resnet18/99131411/hybrid/models/261.onnx"
    m1 = onnx.load(m1_path)
    m2 = onnx.load(m2_path)
    check_model_same(m1, m2)

test_model_same()
