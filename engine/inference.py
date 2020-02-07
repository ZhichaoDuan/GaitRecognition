import torch
import numpy as np
import logging
from utils import compute_cosine_dist, compute_euclidean_dist
from utils.accuracy import exclude_identical, list_to_str

def do_test(model, cfg, val_loader):
    model.eval().to(cfg.MODEL.DEVICE)
    logger = logging.getLogger(cfg.LOGGER.NAME)

    features = []
    views = []
    ids = []
    status = []

    for items in val_loader:
        data, view_, status_, id_, batch_frames = items
        batch_frames = torch.tensor(batch_frames).to(cfg.MODEL.DEVICE)
        data = data.to(cfg.MODEL.DEVICE)

        _, feature_, _ = model(data, batch_frames)

        n, num_bins, _ = feature_.size()
        features.append(feature_.view(n, -1).data.cpu().numpy())
        views += view_
        ids += id_
        status += status_

    logger.info('Feature retrival finished, about to compute acc.')

    features = np.concatenate(features, axis=0)
    ids = np.array(ids)
    
    view_set = list(set(views))
    view_set.sort()

    view_num = len(view_set)

    probe_seq = [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']]
    gallery_seq = ['nm-01', 'nm-02', 'nm-03', 'nm-04']

    acc = np.zeros(shape=(len(probe_seq), view_num, view_num, cfg.VAL.K))
    for ps_idx, ps_ in enumerate(probe_seq):
        for v1, probe_view in enumerate(view_set):
            for v2, gallery_view in enumerate(view_set):
                g_mask = np.isin(status, gallery_seq) & np.isin(views, [gallery_view])
                gallery_x = features[g_mask, :]
                gallery_y = ids[g_mask]

                p_mask = np.isin(status, ps_) & np.isin(views, [probe_view])
                probe_x = features[p_mask, :]
                probe_y = ids[p_mask]
                
                if not cfg.MODEL.BNNECK:
                    dist = compute_euclidean_dist(probe_x, gallery_x)
                else:
                    dist = compute_cosine_dist(probe_x, gallery_x)
                idx = np.argsort(dist, axis=1)
                acc[ps_idx, v1, v2, :] = np.round(np.sum(np.cumsum(np.expand_dims(probe_y, -1) == gallery_y[idx[:, :cfg.VAL.K]], 1) > 0, 0) * 100 / dist.shape[0], 2)
    
    logger.info('Acc compute finishied.')

    for i in range(cfg.VAL.K):
        logger.info('===Rank-%d (Include identical-view cases)===', i + 1)
        logger.info('NM: %.3f,\tBG: %.3f,\tCL: %.3f', 
            np.mean(acc[0, :, :, i]),
            np.mean(acc[1, :, :, i]),
            np.mean(acc[2, :, :, i]))

    for i in range(cfg.VAL.K):
        logger.info('===Rank-%d (Exclude identical-view cases)===', i + 1)
        logger.info('NM: %.3f,\tBG: %.3f,\tCL: %.3f', 
            exclude_identical(acc[0, :, :, i], False),
            exclude_identical(acc[1, :, :, i], False),
            exclude_identical(acc[2, :, :, i], False))

    for i in range(cfg.VAL.K):
        logger.info('===Rank-%d of each angle (Exclude identical-view cases)===', i + 1)
        logger.info('NM:%s', list_to_str(exclude_identical(acc[0, :, :, i], True).tolist()))
        logger.info('BG:%s', list_to_str(exclude_identical(acc[1, :, :, i], True).tolist()))
        logger.info('CL:%s', list_to_str(exclude_identical(acc[2, :, :, i], True).tolist()))

    