import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = ['SetNet']

def clip(arr, tar):
    len_ = len(arr)
    for i in range(len(arr)):
        if arr[-(i+1)] != tar:
            break
        else:
            len_ -= 1
    return arr[:len_]

class ActivatedCNN(nn.Module):
    def __init__(self, in_channels, out_channels, ks, activation, **kwargs):
        super(ActivatedCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ks, **kwargs)
        self.activation = getattr(F, activation)

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x, inplace=True)

class SetNet(nn.Module):
    def __init__(self, cfg, nm_cls):
        super(SetNet, self).__init__()
        self.num_features = cfg.MODEL.NUM_FEATURES
        self.activation = cfg.MODEL.ACTIVATION
        self.batch_frame = None
        self.use_bnneck = cfg.MODEL.BNNECK
        kwargs = dict(
            activation = self.activation,
            bias = False
        )
        self.local_layer1 = ActivatedCNN(1, 32, 5, padding=2, **kwargs)
        self.local_layer2 = ActivatedCNN(32, 32, 3, padding=1, **kwargs)
        self.local_layer3 = ActivatedCNN(32, 64, 3, padding=1, **kwargs)
        self.local_layer4 = ActivatedCNN(64, 64, 3, padding=1, **kwargs)
        self.local_layer5 = ActivatedCNN(64, 128, 3, padding=1, **kwargs)
        self.local_layer6 = ActivatedCNN(128, 128, 3, padding=1, **kwargs)

        self.global_layer1 = ActivatedCNN(32, 64, 3, padding=1, **kwargs)
        self.global_layer2 = ActivatedCNN(64, 64, 3, padding=1, **kwargs)
        self.global_layer3 = ActivatedCNN(64, 128, 3, padding=1, **kwargs)
        self.global_layer4 = ActivatedCNN(128, 128, 3, padding=1, **kwargs)
        
        if self.use_bnneck:
            self.bottleneck = nn.BatchNorm1d(self.num_features)
            self.bottleneck.bias.requires_grad_(False)
            

        self.bin_num = [1, 2, 4, 8, 16]
        self.fc1 = nn.Parameter(
            nn.init.normal_(torch.zeros(sum(self.bin_num)*2, 128, self.num_features), std=0.001)
        )
        self.fc2 = nn.Parameter(
            nn.init.normal_(torch.zeros(sum(self.bin_num)*2, self.num_features, nm_cls), std=0.001)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm1d):
                if m.affine:
                    nn.init.constant_(m.weight, 1.0)
                    nn.init.constant_(m.bias, 0.0)

    def _max(self, x, n, m):
        _, c, h, w = x.size()
        x = x.view(n, m, c, h, w)
        if self.batch_frame is None:
            return torch.max(x, 1)[0]
        else: 
            tmp = [
                torch.max(x[:, self.batch_frame[i]:self.batch_frame[i+1], ...], 1)[0]
                for i in range(len(self.batch_frame) - 1)
            ]
            return torch.cat(tmp, 0)

    def forward(self, x, batch_frame=None):
        if batch_frame is not None:
            batch_frame = batch_frame[0].data.cpu().numpy().tolist()
            batch_frame = clip(batch_frame, 0)
            frame_sum = np.sum(batch_frame)
            if frame_sum < x.size(1):
                x = x[:, :frame_sum, ...]
            self.batch_frame = [0] + np.cumsum(batch_frame).tolist()
        x = x.unsqueeze(2) # added image channel
        n, m, c, h, w = x.size()

        x = self.local_layer1(x.view(-1, c, h, w))
        x = self.local_layer2(x)
        x = F.max_pool2d(x, 2)

        global_ = self.global_layer1(self._max(x, n, m))
        global_ = self.global_layer2(global_)
        global_ = F.max_pool2d(global_, 2)

        x = self.local_layer3(x)
        x = self.local_layer4(x)
        x = F.max_pool2d(x, 2)

        global_ = self.global_layer3(global_ + self._max(x, n, m))
        global_ = self.global_layer4(global_)

        x = self.local_layer5(x)
        x = self.local_layer6(x)
        x = self._max(x, n, m)

        global_ = global_ + x

        feature = list()
        n, c, h, w = global_.size()
        for bin_ in self.bin_num:
            z = x.view(n, c, bin_, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
            z = global_.view(n, c, bin_, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
    
        feature = torch.cat(feature, 2).permute(2, 0, 1) # num of bins, batch size, num of channels
        feature = feature.matmul(self.fc1)
        feature = feature.permute(1, 0, 2) # 128 62 256

        if self.use_bnneck:
            bs, nm_cnns, ftrs = feature.size()
            neck = self.bottleneck(feature.contiguous().view(-1, ftrs))
            neck_for_cls = neck.view(bs, nm_cnns, ftrs)
            neck = neck_for_cls.permute(1, 0, 2).matmul(self.fc2).permute(1, 0, 2)
            return feature.contiguous(), neck.contiguous(), neck_for_cls.contiguous()
        else:
            return feature.contiguous()
