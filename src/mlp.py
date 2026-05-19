import argparse

from src.ingest_data import get_datasets
from src.logger import setup_logger
from src.perceptron import Perceptron


def main(*, num_nodes=6, step_size=0.1, use_momentum=True):
    logger = setup_logger(__name__, __name__ + ".log")

    logger.info("Getting data sets")
    datasets = get_datasets()

    logger.info("Creating perceptron")
    perceptron = Perceptron(
        datasets,
        num_nodes,
        use_momentum,
        step_size,
    )

    logger.info("Training perceptron")
    perceptron.train()
    logger.info("Testing perceptron")
    perceptron.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--momentum", type=bool, default=True)

    args = parser.parse_args()
    main(num_nodes=args.nodes, step_size=args.step_size, use_momentum=args.momentum)
