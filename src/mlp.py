import argparse
import numpy as np

from src.ingest_data import get_datasets
from src.logger import setup_logger
from src.perceptron import Perceptron

# from src.testing import test


def main(*, num_nodes=6, step_size=0.1, use_momentum=True):
    logger = setup_logger(__name__, __name__ + ".log")

    logger.info("Getting data sets")
    training_data, validation_data, testing_data = get_datasets()
    num_inputs = np.shape(training_data)[1] - 1

    logger.info("Creating perceptron")
    perceptron = Perceptron(
        num_inputs, num_nodes, training_data, use_momentum, step_size, validation_data
    )

    logger.info("Training perceptron")
    perceptron.train()
    logger.info("Testing perceptron")
    # test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--momentum", type=bool, default=True)

    args = parser.parse_args()
    main(num_nodes=args.nodes, step_size=args.step_size, use_momentum=args.momentum)
