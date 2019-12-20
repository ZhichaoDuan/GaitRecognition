import torch
import numpy as np


def do_test(model, cfg, test_loader):
    model.eval()

    features = []
    views = []
    ids = []
    status = []

    for items in test_loader:
        data, view_, status_, id_, batch_frames = items
        batch_frames = torch.tensor(batch_frames).cuda()
        feature_ = model(data, batch_frames)
        n, num_bins, _ = feature_.size()
        features.append(feature_.view(n, -1).data.cpu().numpy())
        views += view_
        ids += id_
        status += status_

    features = np.array(features)
    ids = np.array(ids)

    view_set = list(set(views))
    view_set.sort()

    view_num = len(view_set)
    sample_num = features.shape[0]

    probe_seq = [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']]
    gallery_seq = ['nm-01', 'nm-02', 'nm-03', 'nm-04']

    acc = np.zeros(shape=(len(probe_seq), view_num, view_num, cfg.NUM_RANKS))
    for idx, ps_ in enumerate(probe_seq):
        for v1, probe_view in enumerate(view_set):
            for v2, gallery_view in enumerate(view_set):
                g_mask = np.isin(status, gallery_seq) & np.isin(views, [gallery_view])
                gallery_x = 