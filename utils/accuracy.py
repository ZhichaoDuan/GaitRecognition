import numpy as np
import torch
from ignite.metrics import Metric
from .distance import compute_euclidean_dist, compute_cosine_dist

def exclude_identical(acc, flattened_view=True):
    res = np.sum(acc - np.diag(np.diag(acc)), 1) / 10
    if not flattened_view:
        res = np.mean(res)
    return res

def list_to_str(arr):
    arr = list(map(lambda x: round(x, 1), arr))
    return str(arr)

class Top_K_Acc(Metric):
    def __init__(self, k):
        super(Top_K_Acc, self).__init__()
        self.k = k

    def reset(self):
        self.features = []
        self.views = []
        self.ids = []
        self.status = []

    def update(self, output):
        neck, view_, status_, id_ = output
        n, _, _ = neck.size()
        
        self.features.append(neck.view(n, -1).data.cpu().numpy())
        self.views += view_
        self.ids += id_
        self.status += status_

    def compute(self):
        features = np.concatenate(self.features, axis=0)
        ids = np.array(self.ids)
        view_set = list(set(self.views))
        view_set.sort()

        view_num = len(view_set)

        probe_seq = [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']]
        gallery_seq = ['nm-01', 'nm-02', 'nm-03', 'nm-04']

        acc = np.zeros(shape=(len(probe_seq), view_num, view_num, self.k))
        for ps_idx, ps_ in enumerate(probe_seq):
            for v1, probe_view in enumerate(view_set):
                for v2, gallery_view in enumerate(view_set):
                    g_mask = np.isin(self.status, gallery_seq) & np.isin(self.views, [gallery_view])
                    gallery_x = features[g_mask, :]
                    gallery_y = ids[g_mask]

                    p_mask = np.isin(self.status, ps_) & np.isin(self.views, [probe_view])
                    probe_x = features[p_mask, :]
                    probe_y = ids[p_mask]

                    dist = compute_cosine_dist(probe_x, gallery_x)
                    idx = np.argsort(dist, axis=1)
                    acc[ps_idx, v1, v2, :] = np.round(np.sum(np.cumsum(np.expand_dims(probe_y, -1) == gallery_y[idx[:, :self.k]], 1) > 0, 0) * 100 / dist.shape[0], 2)

        iiv = dict(
            NM=np.mean(acc[0, :, :, self.k-1]), 
            BG=np.mean(acc[1, :, :, self.k-1]), 
            CL=np.mean(acc[2, :, :, self.k-1])
        )
        eiv_flatten = dict(
            NM=exclude_identical(acc[0, :, :, self.k-1], False), 
            BG=exclude_identical(acc[1, :, :, self.k-1], False), 
            CL=exclude_identical(acc[2, :, :, self.k-1], False)
        )
        eiv_array = dict(
            NM=list_to_str(exclude_identical(acc[0, :, :, self.k-1], True).tolist()), 
            BG=list_to_str(exclude_identical(acc[1, :, :, self.k-1], True).tolist()), 
            CL=list_to_str(exclude_identical(acc[2, :, :, self.k-1], True).tolist())
        )

        return (iiv, eiv_flatten, eiv_array)
