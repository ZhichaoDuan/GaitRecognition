import torch
import numpy as np
import xarray as xr
import os
from PIL import Image

class CASIADataset(torch.utils.data.Dataset):
    def __init__(self, seqs_dir, ids, status, views, cfg, transform=None):
        self.seqs_dir = seqs_dir
        self.ids = ids
        self.status = status
        self.views = views
        self.use_cache = cfg.TRAIN.CACHE
        self.data = [None] * len(self.ids)
        self.transform = transform
        self.index_dict = self._init_index_dict()
        if self.use_cache:
            self.cache()
    
    def _init_index_dict(self):
        _ = np.zeros((len(set(self.ids)),
                      len(set(self.status)),
                      len(set(self.views))), dtype=int)
        _ -= 1
        index_dict = xr.DataArray(
            _,
            dims=['id', 'status', 'view'],
            coords={
                'id':sorted(list(set(self.ids))),
                'status':sorted(list(set(self.status))),
                'view':sorted(list(set(self.views))),
            }
        )

        for i in range(self.__len__()):
            _id = self.ids[i]
            _status = self.status[i]
            _view = self.views[i]
            index_dict.loc[_id, _status, _view] = i
        return index_dict

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if not self.use_cache:
            data = self._load_imgs(self.seqs_dir[idx])
        elif self.data[idx] is None:
            data = self._load_imgs(self.seqs_dir[idx])
            self.data[idx] = data
        else:
            data = self.data[idx]
        return data, self.views[idx], self.status[idx], self.ids[idx]

    def _load_imgs(self, dir_):
        files = sorted(os.listdir(dir_))
        ims = []
        for file_ in files:
            pil_im = Image.open(os.path.join(dir_, file_))
            if self.transform is not None:
                im = self.transform(pil_im)
            ims.append(im)
        return ims

    def cache(self):
        for i in range(self.__len__()):
            self.__getitem__(i)