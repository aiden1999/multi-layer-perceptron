import datetime
import json

from src.classes.perceptron import Perceptron


class TextOutput:
    def __init__(self, num_nodes, perceptron: Perceptron):
        self.num_nodes = num_nodes
        self.step_size = perceptron.step_size
        self.used_momentum = perceptron.use_momentum
        self.used_bold_driver = perceptron.use_bold_driver
        self.used_annealing = perceptron.use_annealing
        # epoch count
        # rmses
        # nodes with biases
        # weights and the nodes between
        # data used in each set.

    def write_to_json(self):
        output_dict = self._create_output()
        current_time = datetime.datetime.now()
        file_name = current_time.strftime("%Y%m%d_%H%M%S")
        with open(f"{file_name}.json", "w") as file:
            json.dump(output_dict, file)

    def _create_output(self):
        output = {
            "number_of_hidden_nodes": self.num_nodes,
            "step_size": self.step_size,
            "used_momentum": self.used_momentum,
            "used_bold_driver": self.used_bold_driver,
            "used_annealing": self.used_annealing,
        }

        return output
