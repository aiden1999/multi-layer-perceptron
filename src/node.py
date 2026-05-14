import numpy as np


class Node:
    def __init__(self, index: int):
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
        self.sum = 0

    def activation_function(self):
        self.u = 1 / (1 + np.exp(-self.sum))

    def calculate_f_prime(self):
        self.f_prime = self.u * (1 - self.u)

    def update_old_bias(self):
        self.bias_old = self.bias

    def update_new_bias(self, step_size: float, *, alpha: float):
        self.bias += step_size * self.delta
        if alpha:
            self.bias += alpha * self.bias_diff

    def update_bias_diff(self):
        self.bias_diff = self.bias - self.bias_old


class HiddenNode(Node):
    def __init__(self, index: int, num_nodes: int):
        super().__init__(index)
        self.layer = "hidden"
        self.bias = np.random.uniform(-2 / num_nodes, 2 / num_nodes)

    def calculate_delta(self, weight, delta_output):
        self.delta = weight * delta_output * self.f_prime


class OutputNode(Node):
    def __init__(self, index: int, num_nodes: int):
        super().__init__(index)
        self.layer = "output"
        self.bias = np.random.uniform(-2 / num_nodes, 2 / num_nodes)

    def calculate_delta(self, correct_output):
        self.delta = (correct_output - self.u) * self.f_prime
