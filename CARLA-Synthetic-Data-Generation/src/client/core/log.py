import logging
import logging.config

import yaml


def setup_logging(cfg_file: str) -> None:
    """Setup logging configuration from a file.

    Args:
        cfg_file: Logging configuration yaml file.
    """
    assert cfg_file.lower().split(".")[-1] in ["yml", "yaml"]
    with open(cfg_file, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
