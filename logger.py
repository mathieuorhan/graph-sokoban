import logging
import os
import time


def setup_logger(base_path="logs", training_id=None, log_fname="training.log"):
    """Setup a basic logger with stdout and file handlers"""
    if training_id is None:
        timestamp = str(int(time.time()))
        log_path = os.path.join(base_path, timestamp)
    else:
        log_path = os.path.join(base_path, training_id)
    os.makedirs(log_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-5.5s]  %(message)s",
        # disabled StreamHandler because I cannot find an easy way to strip out
        # colors, and I would like to keep them for screen display.
        handlers=[
            logging.FileHandler(os.path.join(log_path, log_fname)),
            # logging.StreamHandler(),
        ],
    )

