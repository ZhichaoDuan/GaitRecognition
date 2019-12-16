import os
import argparse
import sys
sys.path.append('.')

from config import cfg
from utils.logger import setup_logger
from data import make_data_loader
from modeling import build_model
from solver import build_optimizer
from losses import build_loss
from engine.trainer import do_train

def train(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    train_loader = make_data_loader(cfg, 'train')
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)
    loss = build_loss(cfg)
    do_train(model, optimizer, cfg, train_loader, loss, cfg.TRAIN.RESTORE_FROM_ITER)

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

    os.makedirs(os.path.join(cfg.OUTPUT_DIR, cfg.EXPERIMENT), exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, cfg.EXPERIMENT, cfg.LOGGER.STORE_DIR), exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, cfg.EXPERIMENT, cfg.CHECKPOINT_DIR), exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, cfg.EXPERIMENT, cfg.RECORD), exist_ok=True)

    logger = setup_logger(cfg)
    logger.info('Logger setup finished')

    if args.config_file is not None:
        logger.info('Merged settings from file %s', args.config_file)
    else:
        logger.info('Using default settings to operate.')
    
    train(cfg)
    
if __name__ == "__main__":
    main()