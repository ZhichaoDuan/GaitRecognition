from .triplet_loss import TripletLoss

def build_loss(cfg):
    return TripletLoss(cfg)