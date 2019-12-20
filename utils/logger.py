import logging
import os
import sys

def setup_logger(cfg, phase):
    logger = logging.getLogger(cfg.LOGGER.NAME)
    level = getattr(logging, cfg.LOGGER.LEVEL)
    logger.setLevel(level)
    formatter = logging.Formatter(cfg.LOGGER.FORMAT)

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if cfg.LOGGER.WRITE_TO_FILE:
        if phase == 'train':
            fh = logging.FileHandler(os.path.join(
                cfg.OUTPUT_DIR,
                cfg.EXPERIMENT,
                cfg.LOGGER.STORE_DIR,
                '{}.log'.format(cfg.LOGGER.NAME),
            ))
        else:
            fh = logging.FileHandler(os.path.join(
                cfg.OUTPUT_DIR,
                cfg.EXPERIMENT,
                'tests',
                '{}.log'.format(cfg.LOGGER.NAME),
            ))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


