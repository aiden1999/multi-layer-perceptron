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
        self.epoch_count = perceptron.epoch_count
        self.rmse_training = perceptron.rmse_training
        self.rmse_validation = perceptron.rmse_validation
        self.rmse_testing = perceptron.rmse_testing
        self.input_nodes = perceptron.input_nodes
        self.hidden_nodes = perceptron.hidden_nodes
        self.output_node = perceptron.output_node
        self.weights_ih = perceptron.weights_ih
        self.weights_ho = perceptron.weights_ho
        # data used in each set.

    def write_to_json(self):
        output_dict = self._create_output()
        current_time = datetime.datetime.now()
        file_name = f"data/output/{current_time.strftime("%Y%m%d_%H%M%S")}.json"
        with open(file_name, "w") as file:
            json.dump(output_dict, file)

    def _create_output(self):
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
        }
        for h in self.hidden_nodes:
            output["biases"]["hidden_nodes"].update({h.index: h.bias})
            output["weights"]["hidden_to_output"].update(
                {h.index: self.weights_ho[h.index].value}
            )

        for i in self.input_nodes:
            output["weights"]["input_to_hidden"].update({i.index: {}})
            for h in self.hidden_nodes:
                output["weights"]["input_to_hidden"][i.index].update(
                    {h.index: self.weights_ih[i.index][h.index].value}
                )

        return output
