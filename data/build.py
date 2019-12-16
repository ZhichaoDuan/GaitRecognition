import os
import torch
import numpy as np
from .datasets.casia import CASIADataset
from .batch_sampler import GaitSampler
from .collate_fn import Collate
from .transforms import build_transforms


def build_dataset(cfg, phase, transform):
    seqs_dir = list()
    views = list()
    status = list()
    subject_ids = list()
    for _subject_id in sorted(os.listdir(cfg.DATASET_DIR)):
        _subject_dir = os.path.join(cfg.DATASET_DIR, _subject_id)
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
    
    split_record_dir = os.path.join(
        cfg.OUTPUT_DIR,
        cfg.EXPERIMENT,
        cfg.RECORD,
        '{}_{}.npy'.format(cfg.INPUT.BOUNDARY, cfg.INPUT.SHUFFLE),
    )

    if not os.path.exists(split_record_dir):
        ids = sorted(list(set(subject_ids)))
        if cfg.INPUT.SHUFFLE:
            np.random.shuffle(ids)
        partition = [ids[:cfg.INPUT.BOUNDARY], ids[cfg.INPUT.BOUNDARY:]]
        np.save(split_record_dir, partition)

    partition = np.load(split_record_dir)
    ids_for_train, ids_for_test = partition
    if phase == 'train':
        ids_chosen = ids_for_train
    else:
        ids_chosen = ids_for_test

    dataset = CASIADataset(
        [seqs_dir[i] for i, l in enumerate(subject_ids) if l in ids_chosen],
        [subject_ids[i] for i, l in enumerate(subject_ids) if l in ids_chosen],
        [status[i] for i, l in enumerate(subject_ids) if l in ids_chosen],
        [views[i] for i, l in enumerate(subject_ids) if l in ids_chosen],
        cfg, transform
    )
    return dataset

def make_data_loader(cfg, phase):
    assert phase in ('train', 'test')
    transforms = build_transforms(cfg, phase)
    dataset = build_dataset(cfg, phase, transforms)
    bs = GaitSampler(dataset, cfg.TRAIN.BATCH_SIZE)
    cf = Collate(cfg).gait_collate_fn
    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=bs,
        collate_fn=cf,
        num_workers=cfg.NUM_WORKERS,
    )
    return loader