"""Weights on nodes."""

import numpy as np

from src.classes.node import Node


class Weight:
    """Weight on the connection between two nodes.

    Attributes:
        left_node: Node the weight is "coming from".
        right_node: Node the weight is "going to".
        value: The weight value.
        value_old: The previous weight value.
        diff: Difference between `value` and `value_old`
    """

    def __init__(self, left_node: Node, right_node: Node, node_count: int):
        """Initialise weight.

        Args:
            left_node: Node the weight is "coming from"
            right_node: Node the weight is "going to"
            node_count: the number of input nodes if left_node is an input node,
                or the number of hidden nodes if left_node is a hidden node.
        """
        self.left_node = left_node
        self.right_node = right_node
        self.value = np.random.uniform(-2 / node_count, 2 / node_count)
        self.value_old = 0
        self.diff = 0

    def update_old_value(self):
        """Set the old weight value to be the current one."""
        self.value_old = self.value

    def update_new_value(self, step_size: float, u: float, alpha=0.0):
        """Calculate new weight.

        Args:
            alpha (float): Used with momentum, typically 0.9.
            step_size: Learning rate.
            u: Output of activation function applied to sum.
        """
        self.value += (step_size * self.right_node.delta * u) + (alpha * self.diff)

    def update_diff(self):
        """Calculate the difference between the current weight value and the old weight value."""
        self.diff = self.value - self.value_old

    def reset_values(self):
        """Reset weight values to previous."""
        self.value = self.value_old
        self.value_old = 0
