from .models import SetNet

def build_model(cfg, nm_cls):
    if cfg.MODEL.NAME == 'SetNet':
        return SetNet(cfg, nm_cls=nm_cls)