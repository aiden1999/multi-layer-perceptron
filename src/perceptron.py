from src.node import HiddenNode, Node, OutputNode
from src.weight import Weight

import numpy as np


class Perceptron:
    def __init__(
        self,
        num_inputs: int,
        num_nodes: int,
        training_data: np.ndarray,
        use_momentum: bool,
        step_size: float,
        validation_data: np.ndarray,
    ):
        self.input_nodes = []
        for input in range(num_inputs):
            input_node = Node(index=input)
            self.input_nodes.append(input_node)

        self.hidden_nodes = []
        for node in range(num_nodes):
            hidden_node = HiddenNode(index=node, num_nodes=num_nodes)
            self.hidden_nodes.append(hidden_node)

        self.output_node = OutputNode(index=0, num_nodes=num_nodes)

        self.weights_ih = []
        for input_node in self.input_nodes:
            input_node_weights = []
            for hidden_node in self.hidden_nodes:
                weight = Weight(input_node, hidden_node, num_inputs)
                input_node_weights.append(weight)
            self.weights_ih.append(input_node_weights)

        self.weights_ho = []
        for hidden_node in self.hidden_nodes:
            weight = Weight(hidden_node, self.output_node, num_nodes)
            self.weights_ho.append(weight)

        self.epoch_count = 0
        self.training_data = training_data
        self.validation_data = validation_data
        self.use_momentum = use_momentum
        self.step_size = step_size
        self.alpha = 0.9
        self.correct_outputs = training_data[:, -1]
        self.predicted_outputs_training = []
        self.predicted_outputs_validation = []
        self.rmse_training = 0.0
        self.rmse_validation = 0.0
        self.rmse_validation_old = 0.0

    def train(self):
        while self.rmse_validation < self.rmse_validation_old:
            self.epoch_count += 1
            self.predicted_outputs_training = []
            print(self.epoch_count)  # TODO: replace with logging
            for row in self.training_data:
                correct_output = row[-1]
                self._forward_pass(row, self.predicted_outputs_training)
                self._backward_pass(correct_output)
                self._update_weights_and_biases(row)
            self.rmse_training = self._calculate_rmse(
                self.correct_outputs, self.predicted_outputs_training
            )
            if self.epoch_count % 5 == 0:
                self.predicted_outputs_validation = []
                self._validate()

    def _validate(self):
        if self.epoch_count != 5:
            self._set_rmse_validation_old()
        for row in self.validation_data:
            self._forward_pass(row, self.predicted_outputs_validation)
        self.rmse_validation = self._calculate_rmse(
            self.correct_outputs, self.predicted_outputs_validation
        )

    def _forward_pass(self, row, predicted_outputs: list):
        self.output_node.sum = self.output_node.bias
        for h in self.hidden_nodes:
            h.sum = h.bias
            for i in self.input_nodes:
                h.sum += row[i.index] * self.weights_ih[i.index][h.index].value
            h.activation_function()
            self.output_node.sum += h.u * self.weights_ho[h.index].value
        self.output_node.activation_function()
        predicted_outputs.append(self.output_node.u)

    def _backward_pass(self, correct_output):
        self.output_node.calculate_f_prime()
        self.output_node.calculate_delta(correct_output)
        for h in self.hidden_nodes:
            h.calculate_f_prime()
            h.calculate_delta(self.weights_ho[h.index].value, self.output_node.delta)

    def _update_weights_and_biases(self, row):
        if self.use_momentum and self.epoch_count == 1:
            for h in self.hidden_nodes:
                for i in self.input_nodes:
                    weight_ih = self.weights_ih[i.index][h.index]
                    weight_ih.update_old_value()
                    weight_ih.update_new_value(self.step_size, row[i.index])
                weight_ho = self.weights_ho[h.index]
                weight_ho.update_old_value()
                weight_ho.update_new_value(self.step_size, h.u)
                h.update_old_bias()
                h.update_new_bias(self.step_size)
            self.output_node.update_old_bias()
            self.output_node.update_new_bias(self.step_size)

        elif self.use_momentum:
            for h in self.hidden_nodes:
                for i in self.input_nodes:
                    weight_ih = self.weights_ih[i.index][h.index]
                    weight_ih.update_diff()
                    weight_ih.update_old_value()
                    weight_ih.update_new_value(
                        self.step_size, row[i.index], alpha=self.alpha
                    )
                weight_ho = self.weights_ho[h.index]
                weight_ho.update_diff()
                weight_ho.update_old_value()
                weight_ho.update_new_value(self.step_size, h.u, alpha=self.alpha)
                h.update_bias_diff()
                h.update_old_bias()
                h.update_new_bias(self.step_size, alpha=self.alpha)
            self.output_node.update_bias_diff()
            self.output_node.update_old_bias()
            self.output_node.update_new_bias(self.step_size, alpha=self.alpha)

        else:
            for h in self.hidden_nodes:
                for i in self.input_nodes:
                    weight_ih = self.weights_ih[i.index][h.index]
                    weight_ih.update_new_value(self.step_size, row[i.index])
                self.weights_ho[h.index].update_new_value(self.step_size, h.u)
                h.update_new_bias(self.step_size)
            self.output_node.update_new_bias(self.step_size)

    def _calculate_rmse(self, correct_outputs, predicted_outputs):
        correct_outputs = np.array(correct_outputs)
        predicted_outputs = np.array(predicted_outputs)
        return np.sqrt(np.mean(correct_outputs - predicted_outputs) ** 2)

    def _set_rmse_validation_old(self):
        self.rmse_validation_old = self.rmse_validation
