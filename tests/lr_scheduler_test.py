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
        lr_scheduler = WarmupMultiStepLR(optimizer, [10,20], warmup_iters=5)
        for i in range(21):
            optimizer.step()
            lr_scheduler.step()
            print(lr_scheduler.get_lr()[0])


if __name__ == '__main__':
    unittest.main()