import os
import torch
from torch.utils.data.sampler import SequentialSampler
import numpy as np
from .datasets.casia import CASIADataset
from .samplers import RandomIdentitySampler
from .collate_batch import collate_fn_train, collate_fn_val
from .transforms import build_transforms


def build_dataset(cfg, train_transform, val_transform):
    seqs_dir = list()
    views = list()
    status = list()
    subject_ids = list()
    for _subject_id in sorted(os.listdir(cfg.DATASET.DATASET_DIR)):
        if _subject_id == '005':
            continue
        _subject_dir = os.path.join(cfg.DATASET.DATASET_DIR, _subject_id)
        for _status in sorted(os.listdir(_subject_dir)):
            _status_dir = os.path.join(_subject_dir, _status)
            for _view in sorted(os.listdir(_status_dir)):
                _seq_dir = os.path.join(_status_dir, _view)
                seqs = os.listdir(_seq_dir)
                if len(seqs) > 0:
                    seqs_dir.append(_seq_dir)
                    subject_ids.append(_subject_id)
                    status.append(_status)
                    views.append(_view)

    ids = sorted(list(set(subject_ids)))
    partition = [ids[:cfg.DATASET.BOUNDARY], ids[cfg.DATASET.BOUNDARY:]]

    ids_for_train, ids_for_val = partition
    num_classes = len(ids_for_train)

    train_dataset = CASIADataset(
        [seqs_dir[i] for i, l in enumerate(subject_ids) if l in ids_for_train],
        [subject_ids[i] for i, l in enumerate(subject_ids) if l in ids_for_train],
        [status[i] for i, l in enumerate(subject_ids) if l in ids_for_train],
        [views[i] for i, l in enumerate(subject_ids) if l in ids_for_train],
        cfg.TRAIN.CACHE, train_transform
    )

    val_dataset = CASIADataset(
        [seqs_dir[i] for i, l in enumerate(subject_ids) if l in ids_for_val],
        [subject_ids[i] for i, l in enumerate(subject_ids) if l in ids_for_val],
        [status[i] for i, l in enumerate(subject_ids) if l in ids_for_val],
        [views[i] for i, l in enumerate(subject_ids) if l in ids_for_val],
        cfg.VAL.CACHE, val_transform
    )

    return train_dataset, val_dataset, num_classes

def make_data_loader(cfg):
    train_transform = build_transforms(cfg, is_train=True)
    val_transform = build_transforms(cfg, is_train=False)

    train_ds, val_ds, num_classes = build_dataset(cfg, train_transform, val_transform)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_ds,
        batch_sampler=RandomIdentitySampler(train_ds, cfg.TRAIN.BATCH_SIZE),
        collate_fn=collate_fn_train(cfg.TRAIN.FRAME_NUM),
        num_workers=cfg.DATASET.NUM_WORKERS,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_ds,
        sampler=SequentialSampler(val_ds),
        collate_fn=collate_fn_val,
        num_workers=cfg.DATASET.NUM_WORKERS,
        batch_size=cfg.VAL.BATCH_SIZE,
    )
    return train_loader, val_loader, num_classes