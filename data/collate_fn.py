import random
import torch

class Collate:
    def __init__(self, cfg):
        self.cfg = cfg
        
    def gait_collate_fn(self, batch):
        batch_size = len(batch)
        views = [batch[i][1] for i in range(batch_size)]
        status = [batch[i][2] for i in range(batch_size)]
        ids = [batch[i][3] for i in range(batch_size)]
        data = []
        for i in range(batch_size):
            temp = random.choices(batch[i][0], k=self.cfg.TRAIN.FRAME_NUM)
            temp = torch.cat(temp, dim=0).unsqueeze(0)
            data.append(temp)
        data = torch.cat(data, dim=0)
        return [data, views, status, ids]
