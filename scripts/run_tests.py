"""Script to run the linter and tests."""

import subprocess
from src.logger import setup_logger

logger = setup_logger("Testing", "tests.log")


def main():
    """Runs linting and tests."""
    logger.info("Started linting")
    subprocess.run(["ruff", "check"])
    logger.info("Running tests")
    subprocess.run(["pytest", "--verbose", "--cov"])


if __name__ == "__main__":
    main()
