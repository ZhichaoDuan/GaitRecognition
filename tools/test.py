import os
import argparse
import torch
import torch.nn as nn
import sys
sys.path.append('.')

from config import cfg
from utils.logger import setup_logger
from data import make_data_loader
from engine.inference import do_test

def test(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.CUDA_VISIBLE_DEVICES
    test_loader = make_data_loader(cfg, 'test')
    f_structure = os.path.join(cfg.OUTPUT_DIR, cfg.EXPERIMENT, cfg.CHECKPOINT_DIR, 'structure.pt')
    model_structure, _ = torch.load(f_structure)
    model = nn.DataParallel(model_structure.float()).cuda()
    f_weight = os.path.join(cfg.OUTPUT_DIR,cfg.EXPERIMENT,cfg.CHECKPOINT_DIR,'{}_{}.pt'.format(cfg.MODEL.NAME,cfg.TEST.TEST_ITER))
    model_weight, _ = torch.load(f_weight)
    model.load_state_dict(model_weight)
    do_test(model, cfg, test_loader)

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
    cfg.CUDA_VISIBLE_DEVICES = ','.join(list(map(lambda x: str(x), cfg.CUDA_VISIBLE_DEVICES)))
    cfg.freeze()

    os.makedirs(os.path.join(cfg.OUTPUT_DIR,cfg.EXPERIMENT,'tests'), exist_ok=True)

    logger = setup_logger(cfg, 'test')
    logger.info('Logger setup finished')

    if args.config_file is not None:
        logger.info('Merged settings from file %s', args.config_file)
    else:
        logger.info('Using default settings to test.')
    logger.info('About to access test method.')
    logger.info('Be advised we used model of iteration %d', cfg.TEST.TEST_ITER)
    test(cfg)

if __name__ == "__main__":
    main()