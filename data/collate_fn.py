import random
import torch
import torch.nn.functional as F
import numpy as np
import math

class CollateFn:
    def __init__(self, cfg, phase):
        self.cfg = cfg
        if phase == 'train':
            self.gait_collate_fn = self.gait_collate_fn_train
        else:
            self.gait_collate_fn = self.gait_collate_fn_test
        
    def gait_collate_fn_train(self, batch):
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

    def gait_collate_fn_test(self, batch):
        batch_size = len(batch)
        views = [batch[i][1] for i in range(batch_size)]
        status = [batch[i][2] for i in range(batch_size)]
        ids = [batch[i][3] for i in range(batch_size)]
        data = []
        for i in range(batch_size):
            data.append(torch.stack(batch[i][0]))
        gpu_num = min(torch.cuda.device_count(), batch_size)
        batch_per_gpu = math.ceil(batch_size / gpu_num)
        batch_frames = [[
            data[i].size(0) for i in range(batch_per_gpu*_, batch_per_gpu*(1+_)) if i < batch_size
        ] for _ in range(gpu_num)]

        if len(batch_frames[-1]) != batch_per_gpu:
            for i in range(batch_per_gpu-len(batch_frames[-1])):
                batch_frames[-1].append(0)

        max_frame_sum = np.max([np.sum(batch_frames[i]) for i in range(gpu_num)])
        data = [
            torch.cat([data[i] for i in range(batch_per_gpu*_, batch_per_gpu*(_+1))  if i < batch_size], 0) for _ in range(gpu_num)
        ]

        data = [F.pad(data[i], (0,0,0,0,0,0,0, max_frame_sum-data[i].size(0))) for i in range(gpu_num)]
        data = torch.stack(data).squeeze(2) # remove the channel dim
        return [data, views, status, ids, batch_frames]
        

