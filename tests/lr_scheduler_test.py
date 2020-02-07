import sys
import unittest

import torch
from torch import nn

sys.path.append('.')
from solver import make_optimizer, WarmupMultiStepLR
from config import cfg

class MyTestCase(unittest.TestCase):
    def test_something(self):
        net = nn.Linear(10, 10)
        optimizer = make_optimizer(cfg, net)
        lr_scheduler = WarmupMultiStepLR(
            optimizer=optimizer, 
            milestones=cfg.SOLVER.MILESTONES, 
            gamma=cfg.SOLVER.GAMMA,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS, 
            last_epoch=cfg.TRAIN.RESTORE_FROM_ITER,
        )
        for i in range(100):
            optimizer.step()
            lr_scheduler.step()
            print(lr_scheduler.get_lr()[0])


if __name__ == '__main__':
    unittest.main()