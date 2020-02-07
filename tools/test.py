import os
import argparse
import torch
import torch.nn as nn
from torch.backends import cudnn
import sys
sys.path.append('.')

from config import cfg
from data import make_data_loader
from utils import Models, setup_logger
from engine.inference import do_test
import logging

def test(cfg):
    # get data loader
    _, val_loader, _ = make_data_loader(cfg)
    # retrive model
    models = Models(cfg)
    logger = logging.getLogger(cfg.LOGGER.NAME)
    logger.info('Testing starts.')
    for model, name in models:
        logger.info('====> Testing with model {} <===='.format(os.path.basename(name)))
        do_test(model, cfg, val_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default=None, help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    logger = setup_logger(cfg)
    logger.info('Testing logger setup finished')

    if cfg.MODEL.DEVICE == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        cudnn.benchmark = True 

    test(cfg)

if __name__ == "__main__":
    main()