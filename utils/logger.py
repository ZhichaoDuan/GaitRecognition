import logging
import os
import sys

def setup_logger(cfg):
    logger = logging.getLogger(cfg.LOGGER.NAME)
    level = getattr(logging, cfg.LOGGER.LEVEL)
    logger.setLevel(level)
    formatter = logging.Formatter(cfg.LOGGER.FORMAT)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if cfg.LOGGER.WRITE_TO_FILE:
        fh = logging.FileHandler(os.path.join(
            cfg.PATH.OUTPUT_DIR,
            cfg.PATH.EXPERIMENT_DIR,
            cfg.PATH.LOG_STORE_DIR,
            '{}.log'.format(cfg.LOGGER.NAME),
        ))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger


