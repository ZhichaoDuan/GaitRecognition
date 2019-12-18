import os
import argparse
import sys
sys.path.append('.')

from config import cfg
from utils.logger import setup_logger
from data import make_data_loader
from modeling import build_model

def test(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    train_loader = make_data_loader(cfg, 'test')
    model = build_model(cfg)
    # do_train(model, optimizer, cfg, train_loader, loss, cfg.TRAIN.RESTORE_FROM_ITER)

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
    logger.info('Logger setup finished')

    if args.config_file is not None:
        logger.info('Merged settings from file %s', args.config_file)
    else:
        logger.info('Using default settings to operate.')

    test(cfg)

if __name__ == "__main__":
    main()