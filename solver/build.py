import torch.optim as optim

def make_optimizer(cfg, model):
    params = []
    for k, v in model.named_parameters():
        if not v.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in k:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [v], "lr": lr, "weight_decay": weight_decay}]
    optimizer = getattr(optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    return optimizer

def make_just_optimizer(cfg, model):
    optimizer = getattr(optim, cfg.SOLVER.OPTIMIZER_NAME)(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    return optimizer