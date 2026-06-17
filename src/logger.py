"""Utilities for logging"""

import coloredlogs
import logging
import os
import sys
import pathlib


def setup_logger(
    name: str, log_file: str, level: int = logging.DEBUG
) -> logging.Logger:
    """Sets up the logger for use elsewhere.

    Args:
        name: String representing the name of the logger.
        log_file: String representing the file for the logger's logs to be recorded.
        level: Level of logging needed.

    Returns:
        Logger.
    """
    log_directory = ensure_log_directory("logs")
    if log_directory is None:
        log_directory = pathlib.Path("logs")
    else:
        log_directory = pathlib.Path("logs")
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        file_handler, console_handler = create_handlers(log_directory, log_file, level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    coloredlogs.install(level=level, logger=logger, stream=console_handler.stream)
    return logger


def ensure_log_directory(path: str):
    """Create log directory.

    Args:
        path: Directory file path.

    Returns:
        Log directory.
    """
    log_directory = os.makedirs(path, exist_ok=True)
    return log_directory


def create_handlers(
    log_directory: pathlib.Path, log_file: str, level: int
) -> tuple[logging.FileHandler, logging.StreamHandler]:
    """Create handlers for logging to files and to the console.

    Args:
        log_directory: Log directory path.
        log_file: Log file name.
        level: Level of logging to be used.

    Returns:
        File logging handler and console logging handler.
    """
    log_format = "%(asctime)s %(name)s %(levelname)s %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)
    file_handler = logging.FileHandler(log_directory / log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    return file_handler, console_handler
