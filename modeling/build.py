from .models import get_model
def build_model(cfg):
    model = get_model(cfg.MODEL.NAME)
    return model(cfg)