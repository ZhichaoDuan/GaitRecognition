import torch
import numpy as np
import logging

def compute_euclidean_dist(ndarr1, ndarr2):
    dist = np.expand_dims(np.sum(ndarr1 ** 2, axis=1), 1) + np.expand_dims(np.sum(ndarr2 ** 2, 1), 0) - 2 * np.matmul(ndarr1, np.transpose(ndarr2))
    dist = np.sqrt(np.maximum(0, dist))
    return dist

def compute_cosine_dist(ndarr1, ndarr2):
    dot = np.matmul(ndarr1, np.transpose(ndarr2))
    norm_arr1 = np.linalg.norm(ndarr1, axis=1).reshape(-1, 1)
    norm_arr2 = np.linalg.norm(ndarr2, axis=1).reshape(1, -1)
    norm_mat = np.matmul(norm_arr1, norm_arr2)
    return 1. - dot / norm_mat

def exclude_identical(acc, flattened_view=True):
    res = np.sum(acc - np.diag(np.diag(acc)), 1) / 10
    if not flattened_view:
        res = np.mean(res)
    return res

def list_to_str(arr):
    arr = list(map(lambda x: round(x, 1), arr))
    return str(arr)

def do_test(model, cfg, test_loader):
    model.eval()
    logger = logging.getLogger(cfg.LOGGER.NAME)
    logger.info('Testing starts.')

    features = []
    views = []
    ids = []
    status = []

    for items in test_loader:
        data, view_, status_, id_, batch_frames = items
        batch_frames = torch.tensor(batch_frames).cuda()
        data = data.cuda()
        if not cfg.MODEL.BNNECK:
            feature_ = model(data, batch_frames)
        else:
            feature_, _, _ = model(data, batch_frames)

        n, num_bins, _ = feature_.size()
        features.append(feature_.view(n, -1).data.cpu().numpy())
        views += view_
        ids += id_
        status += status_

    features = np.concatenate(features, axis=0)
    ids = np.array(ids)
    logger.info('Feature retrival finished, about to compute acc.')
    view_set = list(set(views))
    view_set.sort()

    view_num = len(view_set)

    probe_seq = [['nm-05', 'nm-06'], ['bg-01', 'bg-02'], ['cl-01', 'cl-02']]
    gallery_seq = ['nm-01', 'nm-02', 'nm-03', 'nm-04']

    acc = np.zeros(shape=(len(probe_seq), view_num, view_num, cfg.TEST.NUM_RANKS))
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
                    dist = compute_euclidean_dist(probe_x, gallery_x)
                idx = np.argsort(dist, axis=1)
                acc[ps_idx, v1, v2, :] = np.round(np.sum(np.cumsum(np.expand_dims(probe_y, -1) == gallery_y[idx[:, :cfg.TEST.NUM_RANKS]], 1) > 0, 0) * 100 / dist.shape[0], 2)
    
    logger.info('Acc compute finishied.')

    for i in range(1):
        logger.info('===Rank-%d (Include identical-view cases)===', i + 1)
        logger.info('NM: %.3f,\tBG: %.3f,\tCL: %.3f', 
            np.mean(acc[0, :, :, i]),
            np.mean(acc[1, :, :, i]),
            np.mean(acc[2, :, :, i]))

    for i in range(1):
        logger.info('===Rank-%d (Exclude identical-view cases)===', i + 1)
        logger.info('NM: %.3f,\tBG: %.3f,\tCL: %.3f', 
            exclude_identical(acc[0, :, :, i], False),
            exclude_identical(acc[1, :, :, i], False),
            exclude_identical(acc[2, :, :, i], False))

    for i in range(1):
        logger.info('===Rank-%d of each angle (Exclude identical-view cases)===', i + 1)
        logger.info('NM:%s', list_to_str(exclude_identical(acc[0, :, :, i], True).tolist()))
        logger.info('BG:%s', list_to_str(exclude_identical(acc[1, :, :, i], True).tolist()))
        logger.info('CL:%s', list_to_str(exclude_identical(acc[2, :, :, i], True).tolist()))

    