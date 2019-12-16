import torch.optim as optim

def build_optimizer(model, cfg):
    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    return optimizer