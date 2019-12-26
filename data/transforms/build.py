from torchvision import transforms

def build_transforms(cfg, phase):
    hc = (cfg.DATASET.RESOLUTION // 64) * 10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda im: im[..., hc:-hc])
    ])
    return transform