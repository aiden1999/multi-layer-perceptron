"""Main program"""

import argparse

from src.classes.perceptron import Perceptron
from src.classes.text_output import TextOutput
from src.ingest_data import get_datasets
from src.logger import setup_logger


def main(
    *,
    num_nodes=6,
    step_size=0.1,
    use_momentum=True,
    use_bold_driver=False,
    use_annealing=True,
):
    """Loads in data, creates a percepton, trains and tests it.

    Args:
        num_nodes (int): Number of nodes to use in the hidden layer.
        step_size (float): Step size used when calculating new values.
        use_momentum (bool): Whether to use momentum improvement.
        use_bold_driver (bool): Whether to use bold driver improvement.
        use_annealing (bool): Whether to use simulated annealing improvement.
    """
    logger = setup_logger(__name__, __name__ + ".log")

    logger.info("Getting data sets")
    datasets = get_datasets()

    logger.info("Creating perceptron")
    perceptron = Perceptron(
        datasets,
        num_nodes,
        use_momentum,
        step_size,
        use_bold_driver,
        use_annealing,
    )

    logger.info("Training perceptron")
    perceptron.train()
    logger.info("Testing perceptron")
    perceptron.test()

    text_output = TextOutput(num_nodes, perceptron)
    logger.info("Writing results to output")
    text_output.write_to_json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--momentum", type=bool, default=True)
    parser.add_argument("--bold_driver", type=bool, default=False)
    parser.add_argument("--annealing", type=bool, default=True)

    args = parser.parse_args()
    main(
        num_nodes=args.nodes,
        step_size=args.step_size,
        use_momentum=args.momentum,
        use_bold_driver=args.bold_driver,
        use_annealing=args.annealing,
    )
