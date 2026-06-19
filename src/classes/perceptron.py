"""Multi-layer perceptron"""

import numpy as np
from numpy.typing import NDArray

from src.classes.node import HiddenNode, Node, OutputNode
from src.classes.weight import Weight
from src.logger import setup_logger

logger = setup_logger(__name__, __name__ + ".log")


class Perceptron:
    """Multi-layer perceptron class.

    Attributes:
        training_data: Dataset to use when training the perceptron.
        validation_data: Datasest to use when validating the perceptron.
        testing_data: Dataset to use when testing the perceptron.
        input_nodes: List of nodes in the input layer.
        hidden_nodes: List of nodes in the hidden layer.
        output_node: Node in the output layer.
        weights_ih: Weights between the input layer and the hidden layer.
        weights_ho: Weights between the hidden layer and the output layer.
        use_annealing: Whether to use simulated annealing improvement.
        epoch_count: Number of epochs the perceptron has been through.
        use_momentum: Whether to use momentum improvement.
        use_bold_driver: Whether to use bold driver improvement.
        step_size: Learning rate.
        alpha: Used with momentum, typically 0.9.
        correct_outputs_training: Known outputs of the training dataset.
        predicted_outputs_training: Predicted outputs of the training dataset.
        correct_outputs_validation: Known outputs of the validation dataset.
        predicted_outputs_validation: Predicted outputs of the validation dataset.
        correct_outputs_testing: Known outputs of the testing dataset.
        predicted_outputs_testing: Predicted outputs of the testing dataset.
        rmse_training: Root mean squared error of the training outputs.
        rmse_training_old: Previous RMSE of the training outputs.
        rmse_validation: Root mean squared error of the validation outputs.
        rmse_validation_old: Previous RMSE of the validation outputs.
        rmse_testing: Root mean squared error of the testing outputs.
        rmse_testing: Previous RMSE of the testing outputs.
    """

    def __init__(
        self,
        datasets: list[NDArray[np.float64]],
        num_nodes: int,
        use_momentum: bool,
        step_size: float,
        use_bold_driver: bool,
        use_annealing: bool,
    ):
        """Initalise perceptron.

        Args:
            datasets: Datasets to be used by the perceptron.
            num_nodes: Number of nodes in the hidden layer.
            use_momentum: Whether to use the momentum improvement.
            step_size: Learning rate.
            use_bold_driver: Whether to use bold driver improvement.
            use_annealing: Whether to use simulated annealing improvement.
        """
        self.training_data = datasets[0]
        self.validation_data = datasets[1]
        self.testing_data = datasets[2]

        num_inputs = np.shape(self.training_data)[1] - 1

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

        self.use_annealing = use_annealing
        if self.use_annealing:
            self.inital_step_size = step_size
            self.decay_rate = 0.01
            self.min_step_size = 1e-5

        self.epoch_count = 0
        self.use_momentum = use_momentum
        self.use_bold_driver = use_bold_driver
        self.step_size = step_size
        self.alpha = 0.9
        self.correct_outputs_training = self.training_data[:, -1]
        self.predicted_outputs_training = []
        self.correct_outputs_validation = self.validation_data[:, -1]
        self.predicted_outputs_validation = []
        self.correct_outputs_testing = self.testing_data[:, 1]
        self.predicted_outputs_testing = []
        self.rmse_training = 0.0
        self.rmse_training_old = 1000.0
        self.rmse_validation = 0.0
        self.rmse_validation_old = 1000.0
        self.rmse_testing = 0.0

    def train(self):
        """Train the perceptron."""
        logger.info("Starting training")
        while self.rmse_validation < self.rmse_validation_old:
            self.epoch_count += 1
            self.predicted_outputs_training = []
            if self.use_annealing:
                self._set_step_size()
            logger.debug(f"Epoch {self.epoch_count} | RMSE {self.rmse_training}")
            for row in self.training_data:
                correct_output = row[-1]
                self._forward_pass(row, self.predicted_outputs_training)
                self._backward_pass(correct_output)
                self._update_weights_and_biases(row)
            if self.use_bold_driver:
                self._set_rmse_training_old()
            self.rmse_training = self._calculate_rmse(
                self.correct_outputs_training, self.predicted_outputs_training
            )
            if self.use_bold_driver:
                self._bold_driver()
            if self.epoch_count % 5 == 0:
                self.predicted_outputs_validation = []
                self._validate()
                logger.debug(
                    f"Old validation RMSE {self.rmse_validation_old} | New validation RMSE {self.rmse_validation}"
                )

    def test(self):
        """Test the perceptron."""
        for row in self.testing_data:
            self._forward_pass(row, self.predicted_outputs_testing)
        self.rmse_testing = self._calculate_rmse(
            self.correct_outputs_testing, self.predicted_outputs_testing
        )
        logger.info(f"Testing RMSE {self.rmse_testing}")

    def _validate(self):
        """Validate the perceptron."""
        if self.epoch_count != 5:
            self._set_rmse_validation_old()
        for row in self.validation_data:
            self._forward_pass(row, self.predicted_outputs_validation)
        self.rmse_validation = self._calculate_rmse(
            self.correct_outputs_validation, self.predicted_outputs_validation
        )

    def _set_step_size(self):
        """Set the step size, only used with simulated annealing."""
        step_size = self.inital_step_size * np.exp(-self.decay_rate * self.epoch_count)
        self.step_size = max(step_size, self.min_step_size)

    def _forward_pass(self, row: NDArray[np.float64], predicted_outputs: list[float]):
        """Forward pass through the perceptron.

        Args:
            row: Row of dataset that is being passed.
            predicted_outputs: Predicted outputs from the dataset that is being
                passed through.
        """
        self.output_node.sum = self.output_node.bias
        for h in self.hidden_nodes:
            h.sum = h.bias
            for i in self.input_nodes:
                h.sum += row[i.index] * self.weights_ih[i.index][h.index].value
            h.activation_function()
            self.output_node.sum += h.u * self.weights_ho[h.index].value
        self.output_node.activation_function()
        predicted_outputs.append(self.output_node.u)

    def _backward_pass(self, correct_output: float):
        """Backward pass through the perceptron.

        Args:
            correct_output: Known output value of the row of the dataset that is
                being passed through.
        """
        self.output_node.calculate_f_prime()
        self.output_node.calculate_delta(correct_output)
        for h in self.hidden_nodes:
            h.calculate_f_prime()
            h.calculate_delta(self.weights_ho[h.index].value, self.output_node.delta)

    def _update_weights_and_biases(self, row: NDArray[np.float64]):
        """Update old and current values of biases on nodes, and the weights between them.

        Args:
            row: Row of dataset that is being passed.

        """
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

    def _revert_weights_and_biases(self):
        """Reset weights and biases to previous values."""
        for h in self.hidden_nodes:
            for i in self.input_nodes:
                self.weights_ih[i.index][h.index].reset_values()
            self.weights_ho[h.index].reset_values()
            h.reset_bias()
        self.output_node.reset_bias()

    def _calculate_rmse(
        self,
        correct_outputs_raw: NDArray[np.float64],
        predicted_outputs_raw: list[float],
    ) -> float:
        """Calculate root mean square error between known and predicted values.

        Args:
            correct_outputs_raw: Known outputs of the dataset.
            predicted_outputs_raw: Predicted values of the dataset.

        Returns:
            Root mean square error.
        """
        correct_outputs = np.array(correct_outputs_raw)
        predicted_outputs = np.array(predicted_outputs_raw)
        return np.sqrt(np.mean(correct_outputs - predicted_outputs) ** 2)

    def _set_rmse_training_old(self):
        """Set the old RMSE for the training dataset to be the current one."""
        self.rmse_training_old = self.rmse_training

    def _set_rmse_validation_old(self):
        """Set the old RMSE for the validation dataset to be the current one."""
        self.rmse_validation_old = self.rmse_validation

    def _bold_driver(self):
        """Bold driver improvement."""
        if self.epoch_count % 10 == 0:
            if self.rmse_training > self.rmse_training_old:
                self._revert_weights_and_biases()
                self.step_size *= 0.5
                if self.step_size < 1e-5:
                    self.step_size = 1e-5
            else:
                self.step_size *= 1.1
                if self.step_size > 0.5:
                    self.step_size = 0.5
