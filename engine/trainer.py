import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Engine, Events
from ignite.contrib.handlers import CustomPeriodicEvent
import os.path as osp
import numpy as np
import time
from utils import fmt_secs, Top_K_Acc

_start = time.time()

def create_supervised_trainer(model, optimizer, loss_fn, train_ids, device, rank):
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[rank],
            output_device=rank,
        )

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        data, _, _, ids = batch
        data = data.to(device)

        feature, _, cls_score = model(data)
        
        feature = feature.permute(1, 0, 2).contiguous()
        ids_gt = torch.tensor([train_ids.index(i) for i in ids], dtype=torch.long, device=device)
        ids_gt_for_cls = ids_gt.unsqueeze(1).repeat(1, cls_score.size(1))
        cls_score = cls_score.reshape(-1, cls_score.size(2))
        ids_gt_for_dis = ids_gt.unsqueeze(0).repeat(feature.size(0), 1)

        loss = loss_fn(feature, cls_score, ids_gt_for_dis, ids_gt_for_cls.reshape(-1))
        loss.backward()

        optimizer.step()
        return loss.item()

    return Engine(_update)

def do_train(
    cfg,
    rank,
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    scheduler, 
    loss_fn, 
):
    logger = logging.getLogger(cfg.LOGGER.NAME)

    train_ids = sorted(list(set(train_loader.dataset.ids)))

    trainer = create_supervised_trainer(model, optimizer, loss_fn, train_ids, cfg.MODEL.DEVICE, rank)

    @trainer.on(Events.STARTED)
    def signal_start(engine):
        if rank == 0:
            logger.info('Training starts.')

    @trainer.on(Events.ITERATION_COMPLETED)
    def step_scheduler(engine):
        scheduler.step()

    cpe1 = CustomPeriodicEvent(n_iterations=cfg.TRAIN.DISPLAY_INFO_STEP)
    cpe1.attach(trainer)
    @trainer.on(getattr(cpe1.Events, 'ITERATIONS_{}_COMPLETED'.format(cfg.TRAIN.DISPLAY_INFO_STEP)))
    def log_info(engine):
        if rank == 0:
            global _start
            _end = time.time()
            time_cost = fmt_secs(_end - _start, 1)
            time_cost = '{} hours {} mins {} secs'.format(*time_cost)

            time_expected = fmt_secs((_end - _start) / engine.state.iteration, cfg.TRAIN.MAX_ITERS - engine.state.iteration)
            time_expected = '{} hours {} mins {} secs'.format(*time_expected)
            logger.info('Iteration: {}, Loss: {:.4f}, already cost: {}, ETA: {}'.format(
                engine.state.iteration,
                engine.state.output,
                time_cost,
                time_expected,
            ))

    cpe2 = CustomPeriodicEvent(n_iterations=cfg.TRAIN.RECORD_STEP)
    cpe2.attach(trainer)
    @trainer.on(getattr(cpe2.Events, 'ITERATIONS_{}_COMPLETED'.format(cfg.TRAIN.RECORD_STEP)))
    def save_checkpoint(engine):
        if rank == 0:
            store_path = osp.join(cfg.PATH.OUTPUT_DIR, cfg.PATH.EXPERIMENT_DIR, cfg.PATH.CHECKPOINT_DIR)
            file_name = 'iter_{:0>6d}.pt'.format(engine.state.iteration)
            file_path = osp.join(store_path, file_name)
            torch.save(model, file_path)

    cpe3 = CustomPeriodicEvent(n_iterations=cfg.TRAIN.MAX_ITERS)
    cpe3.attach(trainer)
    @trainer.on(getattr(cpe3.Events, 'ITERATIONS_{}_COMPLETED'.format(cfg.TRAIN.MAX_ITERS)))
    def signal_terminate(engine):
        logger.info('Reaching maximum {} epochs, break training loop now.'.format(cfg.TRAIN.MAX_ITERS))
        engine.terminate()
    
    trainer.run(train_loader)