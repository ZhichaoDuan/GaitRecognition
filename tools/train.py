import os
import argparse
import sys
sys.path.append('.')

import torch
from torch.backends import cudnn

from config import cfg
from utils.logger import setup_logger
from data import make_data_loader
from modeling import build_model
from solver import make_optimizer, make_just_optimizer, WarmupMultiStepLR
from layers import make_loss
from engine.trainer import do_train

def train(cfg, rank):
    # data loader
    train_loader, val_loader, num_classes = make_data_loader(cfg)
    # setup model
    model = build_model(cfg, num_classes)
    # make optimizer
    optimizer = make_optimizer(cfg, model)
    # make loss
    loss_fn = make_loss(cfg, num_classes)
    # change settings can simulate no scheduler mode.
    scheduler = WarmupMultiStepLR(
        optimizer=optimizer, 
        milestones=cfg.SOLVER.MILESTONES, 
        gamma=cfg.SOLVER.GAMMA,
        warmup_iters=cfg.SOLVER.WARMUP_ITERS, 
        last_epoch=cfg.TRAIN.RESTORE_FROM_ITER,
    )
    # call core function
    do_train(
        cfg,
        rank,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", default=None, help="path to config file", type=str
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    os.makedirs(cfg.PATH.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.PATH.OUTPUT_DIR, cfg.PATH.EXPERIMENT_DIR), exist_ok=True)
    os.makedirs(os.path.join(cfg.PATH.OUTPUT_DIR, cfg.PATH.EXPERIMENT_DIR, cfg.PATH.LOG_STORE_DIR), exist_ok=True)
    os.makedirs(os.path.join(cfg.PATH.OUTPUT_DIR, cfg.PATH.EXPERIMENT_DIR, cfg.PATH.CHECKPOINT_DIR ), exist_ok=True)

    logger = setup_logger(cfg)
    logger.info('Training logger setup finished')

    logger.info(args)

    if args.config_file is not None:
        logger.info('Merged settings from file %s', args.config_file)
        with open(args.config_file,'r') as cf:
            logger.info('\n'+cf.read())
    logger.info("Running with config:\n{}".format(cfg))
    
    if cfg.MODEL.DEVICE == 'cuda':
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
        cudnn.benchmark = True

    world_sz = torch.cuda.device_count()
    if world_sz > 1:
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://',
            world_size=world_sz,
            rank=args.local_rank,
        )
        torch.cuda.set_device(args.local_rank)

    train(cfg, args.local_rank)
    
if __name__ == "__main__":
    main()