import torch
import glob
import os.path as osp

class Models(object):
    def __init__(self, cfg):
        experiment_dir = osp.join(cfg.PATH.OUTPUT_DIR, cfg.PATH.EXPERIMENT_DIR, cfg.PATH.CHECKPOINT_DIR)
        self.options = self.retrieve_ops(experiment_dir)
    
    def retrieve_ops(self, path):
        files = sorted(glob.glob(osp.join(path, 'iter*.pt')))
        files.reverse()
        return files

    def __len__(self):
        return len(self.options)
    
    def __getitem__(self, index):
        model = torch.load(self.options[index])
        return model, self.options[index]