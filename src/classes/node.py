"""Node classes used in perceptron."""

import numpy as np


class Node:
    """Base node class

    Attributes:
        index: Index of the node in the layer. Starts at 0.
        bias: Bias applied to the node.
        bias_old: Previous bias applied to the node.
        bias_diff: Difference between `bias` and `bias_old`
        sum: Weighted sum.
        u: Output of activation function applied to sum.
        f_prime: f'(S_j) for node j. Used in backward pass.
        delta: Local gradient at the node.
        layer: Layer within the MLP.
    """

    def __init__(self, index: int):
        """Initialise node.

        Args:
            index: Index of the node in the layer. Starts at 0.
        """
        self.index = index
        self.bias = 0
        self.bias_old = 0
        self.bias_diff = 0
        self.sum = 0.0
        self.u = 0
        self.f_prime = 0
        self.delta = 0
        self.layer = "input"

    def reset_sum(self):
        """Set sum back to 0."""
        self.sum = 0

    def activation_function(self):
        """Transforms the calculated sum into the actual output."""
        self.u = 1 / (1 + np.exp(-self.sum))

    def calculate_f_prime(self):
        """Calculate the derivative of the activation function."""
        self.f_prime = self.u * (1 - self.u)

    def update_old_bias(self):
        """Set the old bias to be the current one."""
        self.bias_old = self.bias

    def update_new_bias(self, step_size: float, alpha=0.0):
        """Calculate new bias.

        Args:
            alpha (float): Used with momentum, typically 0.9.
            step_size: Learning rate.
        """
        self.bias += (step_size * self.delta) + (alpha * self.bias_diff)

    def update_bias_diff(self):
        """Calculate the difference between the current bias and the old bias."""
        self.bias_diff = self.bias - self.bias_old

    def reset_bias(self):
        """Reset bias to previous."""
        self.bias = self.bias_old
        self.bias_old = 0


class HiddenNode(Node):
    """Hidden node class.

    Attributes:
        layer: Layer within the MLP.
        bias: Bias applied to the node.
        delta: Local gradient at the node.
    """

    def __init__(self, index: int, num_nodes: int):
        """Initialise hidden node.

        Args:
            index: Index of the node in the layer. Starts at 0.
            num_nodes: Number of nodes in the hidden layer.
        """
        super().__init__(index)
        self.layer = "hidden"
        self.bias = np.random.uniform(-2 / num_nodes, 2 / num_nodes)

    def calculate_delta(self, weight: float, delta_output: float):
        """Calculate the delta - i.e. how much did this node contribute to the
        output errors.

        Args:
            weight: Weight value between the node and the output node.
            delta_output: Delta of the output node.
        """
        self.delta = weight * delta_output * self.f_prime


class OutputNode(Node):
    """Output node class.

    Attributes:
        layer: Layer within the MLP.
        bias: Bias applied to the node.
        delta: Local gradient at the node.
    """

    def __init__(self, index: int, num_nodes: int):
        """Initalise output node.

        Args:
            index: Index of the node in the layer. Starts at 0.
            num_nodes: Number of nodes in the hidden layer.
        """
        super().__init__(index)
        self.layer = "output"
        self.bias = np.random.uniform(-2 / num_nodes, 2 / num_nodes)

    def calculate_delta(self, correct_output: float):
        """Calculate the delta - i.e. how wrong was this prediction.

        Args:
            correct_output: Expected output.
        """
        self.delta = (correct_output - self.u) * self.f_prime
