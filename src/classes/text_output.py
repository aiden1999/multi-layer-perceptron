"""Output to JSON file."""

import datetime
import json

from src.classes.perceptron import Perceptron


class TextOutput:
    """Text output class.

    Attributes:
        num_nodes: Number of nodes in the hidden layer.
        step_size: Learning rate.
        used_momentum: Whether the momentum improvement was used.
        used_bold_driver: Whether the bold driver improvement was used.
        used_annealing: Whether the simulated annealing improvement was used.
        epoch_count: Number of epochs the perceptron went through.
        rmse_training: Root mean squared error of the training outputs.
        rmse_validation: Root mean squared error of the validation outputs.
        rmse_testing: Root mean squared error of the testing outputs.
        input_nodes: List of nodes in the input layer.
        hidden_nodes: List of nodes in the hidden layer.
        output_node: Node in the output layer.
        weights_ih: Weights between the input layer and the hidden layer.
        weights_ho: Weights between the hidden layer and the output layer.
        training_data: Dataset used when training the perceptron.
        validation_data: Dataset used when validating the perceptron.
        testing_data: Dataset used when testing the perceptron.
    """

    def __init__(self, num_nodes: int, perceptron: Perceptron):
        """Initialise text output.

        Args:
            num_nodes: Number of nodes in the hidden layer.
            perceptron: Multi layer perceptron that was used.
        """
        self.num_nodes = num_nodes
        self.step_size = perceptron.step_size
        self.used_momentum = perceptron.use_momentum
        self.used_bold_driver = perceptron.use_bold_driver
        self.used_annealing = perceptron.use_annealing
        self.epoch_count = perceptron.epoch_count
        self.rmse_training = perceptron.rmse_training
        self.rmse_validation = perceptron.rmse_validation
        self.rmse_testing = perceptron.rmse_testing
        self.input_nodes = perceptron.input_nodes
        self.hidden_nodes = perceptron.hidden_nodes
        self.output_node = perceptron.output_node
        self.weights_ih = perceptron.weights_ih
        self.weights_ho = perceptron.weights_ho
        self.training_data = perceptron.training_data.tolist()
        self.validation_data = perceptron.validation_data.tolist()
        self.testing_data = perceptron.testing_data.tolist()

    def write_to_json(self):
        """Write perceptron information to JSON file."""
        output_dict = self._create_output()
        current_time = datetime.datetime.now()
        file_name = f"data/output/{current_time.strftime("%Y%m%d_%H%M%S")}.json"
        with open(file_name, "w") as file:
            json.dump(output_dict, file, indent=4)

    def _create_output(self) -> dict:
        """Create output dictionary.

        Returns:
            Dictionary with perceptron information.
        """
        output = {
            "parameters": {
                "number_of_hidden_nodes": self.num_nodes,
                "step_size": self.step_size,
                "used_momentum": self.used_momentum,
                "used_bold_driver": self.used_bold_driver,
                "used_annealing": self.used_annealing,
            },
            "epoch_count": self.epoch_count,
            "rmse": {
                "training": self.rmse_training,
                "validation": self.rmse_validation,
                "testing": self.rmse_testing,
            },
            "biases": {
                "hidden_nodes": {},
                "output_node": self.output_node.bias,
            },
            "weights": {"input_to_hidden": {}, "hidden_to_output": {}},
            "data": {
                "training": self.training_data,
                "validation": self.validation_data,
                "testing": self.testing_data,
            },
        }

        for h in self.hidden_nodes:
            output["biases"]["hidden_nodes"].update({h.index: h.bias})
            weight_ho = self.weights_ho[h.index].value
            output["weights"]["hidden_to_output"].update({h.index: weight_ho})

        for i in self.input_nodes:
            o_w_ih = output["weights"]["input_to_hidden"]
            o_w_ih.update({i.index: {}})
            for h in self.hidden_nodes:
                weight_ih = self.weights_ih[i.index][h.index].value
                o_w_ih[i.index].update({h.index: weight_ih})

        return output
