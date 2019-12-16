import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SetNet']

class ActivatedCNN(nn.Module):
    def __init__(self, in_channels, out_channels, ks, activation, **kwargs):
        super(ActivatedCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ks, **kwargs)
        self.activation = getattr(F, activation)

    def forward(self, x):
        x = self.conv(x)
        return self.activation(x, inplace=True)

class SetNet(nn.Module):
    def __init__(self, cfg):
        super(SetNet, self).__init__()
        self.num_features = cfg.MODEL.NUM_FEATURES
        self.activation = cfg.MODEL.ACTIVATION
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
        
        self.bin_num = [1, 2, 4, 8, 16]
        self.fc_bin = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(sum(self.bin_num)*2, 128, self.num_features))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def _max(self, x, n, m):
        _, c, h, w = x.size()
        x = x.view(n, m, c, h, w)
        mx = torch.max(x, 1)
        return mx[0]

    def forward(self, x):
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
        
        feature = torch.cat(feature, 2).permute(2, 0, 1)
        feature = feature.matmul(self.fc_bin)
        feature = feature.permute(1, 0, 2)
        return feature
