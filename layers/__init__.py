from .triplet_loss import TripletLoss
from .cross_entropy_loss import CrossEntropyLabelSmooth
import torch
import torch.nn.functional as F

def make_loss(cfg, num_classes):
    if cfg.TRAIN.CLS_LOSS_TYPE == 'none':
        trip = TripletLoss(cfg.TRIPLET_LOSS.MARGIN).to(cfg.MODEL.DEVICE)
        def loss_func(feature, cls_score, target, cls_target):
            full_loss, hard_loss, _, _ = trip(feature, target)
            if cfg.TRIPLET_LOSS.TYPE == 'full':
                loss_chosen = full_loss.mean()
            elif cfg.TRIPLET_LOSS.TYPE == 'hard':
                loss_chosen = hard_loss.mean()
            return loss_chosen
        return loss_func

    elif cfg.TRAIN.CLS_LOSS_TYPE == 'normal':
        trip = TripletLoss(cfg.TRIPLET_LOSS.MARGIN).to(cfg.MODEL.DEVICE)
        def loss_func(feature, cls_score, target, cls_target):
            full_loss, hard_loss, _, _ = trip(feature, target)
            if cfg.TRIPLET_LOSS.TYPE == 'full':
                loss_chosen = full_loss.mean()
            elif cfg.TRIPLET_LOSS.TYPE == 'hard':
                loss_chosen = hard_loss.mean()
            return loss_chosen + F.cross_entropy(cls_score, cls_target) / cfg.TRAIN.CE_DIVIDED
        return loss_func

    elif cfg.TRAIN.CLS_LOSS_TYPE == 'smooth':
        trip = TripletLoss(cfg.TRIPLET_LOSS.MARGIN).to(cfg.MODEL.DEVICE)
        sce = CrossEntropyLabelSmooth(num_classes, cfg.TRAIN.SMOOTH_EPSILON, cfg.MODEL.DEVICE).to(cfg.MODEL.DEVICE)
        def loss_func(feature, cls_score, target, cls_target):
            full_loss, hard_loss, _, _ = trip(feature, target)
            if cfg.TRIPLET_LOSS.TYPE == 'full':
                loss_chosen = full_loss.mean()
            elif cfg.TRIPLET_LOSS.TYPE == 'hard':
                loss_chosen = hard_loss.mean()
            return loss_chosen + sce(cls_score, cls_target) / cfg.TRAIN.CE_DIVIDED
        return loss_func

