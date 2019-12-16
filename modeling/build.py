import importlib

def build_model(cfg):
    model_holder = importlib.import_module(cfg.MODEL.NAME)
    model = getattr(model_holder, cfg.MODEL.NAME)
    return model