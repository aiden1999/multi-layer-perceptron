import numpy as np


class Node:
    def __init__(self, index: int):
        self.index = index
        self.layer = "input"


class HiddenNode(Node):
    def __init__(self, index: int, num_nodes: int):
        super().__init__(index)
        self.bias = np.random.uniform(-2 / num_nodes, 2 / num_nodes)
        self.bias_old = None
        self.layer = "hidden"


class OutputNode(HiddenNode):
    def __init__(self, index: int, num_nodes: int):
        super().__init__(index, num_nodes)
        self.layer = "output"
