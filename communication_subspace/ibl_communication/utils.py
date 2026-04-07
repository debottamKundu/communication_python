from pathlib import Path
import logging
import sys
import yaml


def check_config():
    """Load config yaml and perform some basic checks"""
    # Get config
    with open(Path(__file__).parent.parent.joinpath("config.yaml"), "r") as config_yml:
        config = yaml.safe_load(config_yml)
    return config


def setup_logger(name="CrossPrediction", log_file="pipeline.log", level=logging.INFO):
    """
    Sets up and returns a customized logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # SAFETY CHECK: Only add handlers if the logger doesn't already have them.
    # This prevents duplicate log lines if the function is accidentally called twice.
    if not logger.handlers:
        # Create formatting
        log_format = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 1. Console Handler (Prints to your terminal)
        c_handler = logging.StreamHandler(sys.stdout)
        c_handler.setFormatter(log_format)
        logger.addHandler(c_handler)

        # 2. File Handler (Saves to a text file)
        f_handler = logging.FileHandler(log_file)
        f_handler.setFormatter(log_format)
        logger.addHandler(f_handler)

    return logger
