import os
import argparse
import sys
sys.path.append('.')

from config import cfg
from utils.logger import setup_logger
from data import make_data_loader
from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR
from layers import build_loss
from engine.trainer import do_train

def train(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    train_loader = make_data_loader(cfg, 'train')
    model = build_model(cfg, nm_cls=len(list(set(train_loader.dataset.ids))))
    if cfg.SOLVER.OPTIMIZER_MANNER == 'layer-wise':
        optimizer = make_optimizer(cfg, model)
    else:
        import torch.optim as optim
        optimizer = getattr(optim, cfg.SOLVER.OPTIMIZER_NAME)(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    if not cfg.TRAIN.USE_SCHEDULER:
        scheduler = None
    elif cfg.TRAIN.RESTORE_FROM_ITER == 0:
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.MILESTONES, warmup_iters=cfg.SOLVER.WARMUP_ITERS)
    else:
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.MILESTONES, warmup_iters=cfg.SOLVER.WARMUP_ITERS, last_epoch=cfg.TRAIN.RESTORE_FROM_ITER)
    loss = build_loss(cfg)
    # do_train(model, optimizer, cfg, train_loader, loss, cfg.TRAIN.RESTORE_FROM_ITER)
    do_train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss=loss,
        iteration=cfg.TRAIN.RESTORE_FROM_ITER,
    )

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

    os.makedirs(cfg.PATH.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.PATH.OUTPUT_DIR, cfg.PATH.EXPERIMENT_DIR), exist_ok=True)
    os.makedirs(os.path.join(cfg.PATH.OUTPUT_DIR, cfg.PATH.EXPERIMENT_DIR, cfg.PATH.LOG_STORE_DIR), exist_ok=True)
    os.makedirs(os.path.join(cfg.PATH.OUTPUT_DIR, cfg.PATH.EXPERIMENT_DIR, cfg.PATH.CHECKPOINT_DIR ), exist_ok=True)
    os.makedirs(os.path.join(cfg.PATH.OUTPUT_DIR, cfg.PATH.EXPERIMENT_DIR, cfg.PATH.SPLIT_RECORD_DIR), exist_ok=True)

    logger = setup_logger(cfg)
    logger.info('Training logger setup finished')

    logger.info(args)

    if args.config_file is not None:
        logger.info('Merged settings from file %s', args.config_file)
        with open(args.config_file,'r') as cf:
            logger.info('\n'+cf.read())
    logger.info("Running with config:\n{}".format(cfg))
    
    train(cfg)
    
if __name__ == "__main__":
    main()