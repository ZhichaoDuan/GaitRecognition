from .backbones import SetNet

def build_model(cfg, num_classes):
    if cfg.MODEL.NAME == 'SetNet':
        return SetNet(cfg.MODEL.NUM_FEATURES, cfg.MODEL.BNNECK, num_classes)