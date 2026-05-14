import numpy as np

from src.node import Node


class Weight:
    def __init__(self, left_node: Node, right_node: Node, node_count: int):
        """[TODO:description]

        Args:
            left_node: [TODO:description]
            right_node: [TODO:description]
            node_count: the number of input nodes if left_node is an input node, or the number of hidden nodes if left_node is a hidden node.
        """
        self.left_node = left_node
        self.right_node = right_node
        self.value = np.random.uniform(-2 / node_count, 2 / node_count)
        self.value_old = 0
        self.diff = 0

    def update_old_value(self):
        self.value_old = self.value

    def update_new_value(self, step_size, u, *, alpha):
        self.value += step_size * self.right_node.delta * u
        if alpha:
            self.value += alpha * self.diff

    def update_diff(self):
        self.diff = self.value - self.value_old
