import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self, cfg):
        super(TripletLoss, self).__init__()
        self.margin = cfg.TRIPLET_LOSS.MARGIN

    def forward(self, ftr, label):
        n, m, d = ftr.size()
        pm = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        nm = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)
        
        dist = self._cal_dist(ftr)
        mean_dist = dist.mean(1).mean(1)
        dist = dist.view(-1)

        # hard triplet loss
        hard_pm_dist = torch.max(torch.masked_select(dist, pm).view(n, m, -1), 2)[0]
        hard_nm_dist = torch.min(torch.masked_select(dist, nm).view(n, m, -1), 2)[0]
        hard_loss = F.relu(self.margin + hard_pm_dist - hard_nm_dist).view(n, -1)
        hard_loss_mean = torch.mean(hard_loss, 1)
        # full triplet loss
        full_pm_dist = torch.masked_select(dist, pm).view(n, m, -1, 1)
        full_nm_dist = torch.masked_select(dist, nm).view(n, m, 1, -1)
        full_loss = F.relu(self.margin + full_pm_dist - full_nm_dist).view(n, -1)

        full_loss_sum = full_loss.sum(1)
        full_loss_num = (full_loss != 0).sum(1).float()
        full_loss_mean = full_loss_sum / full_loss_num
        full_loss_mean[full_loss_num == 0] = 0 # unlikely happen, though
        return full_loss_mean, hard_loss_mean, mean_dist, full_loss_num

    def _cal_dist(self, x):
        x2 = torch.sum(x**2, 2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1, 2) - 2 * torch.matmul(x, x.transpose(1, 2))
        dist = torch.sqrt(F.relu(dist))
        return dist
    


