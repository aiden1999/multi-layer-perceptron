from src.node import HiddenNode, Node, OutputNode
from src.weight import Weight


class Perceptron:
    def __init__(self, num_inputs: int, num_nodes: int):
        self.input_nodes = []
        for input in range(num_inputs):
            input_node = Node(index=input)
            self.input_nodes.append(input_node)

        self.hidden_nodes = []
        for node in range(num_nodes):
            hidden_node = HiddenNode(index=node, num_nodes=num_nodes)
            self.hidden_nodes.append(hidden_node)

        self.output_node = OutputNode(index=0, num_nodes=num_nodes)

        self.weights_input_hidden = []
        for input_node in self.input_nodes:
            for hidden_node in self.hidden_nodes:
                weight = Weight(input_node, hidden_node, num_inputs)
                self.weights_input_hidden.append(weight)

        self.weights_hidden_output = []
        for hidden_node in self.hidden_nodes:
            weight = Weight(hidden_node, self.output_node, num_nodes)
            self.weights_hidden_output.append(weight)
