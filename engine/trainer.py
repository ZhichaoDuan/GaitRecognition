import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Engine, Events
from ignite.handlers import Timer, ModelCheckpoint
import os
import numpy as np
import time
from utils import fmt_secs, Top_K_Acc

def score_eval(engine):
    iiv, eiv_flatten, eiv_array = engine.state.metrics['acc']
    if eiv_flatten['NM'] < 95 or eiv_flatten['BG'] < 87.2 or eiv_flatten['CL'] < 70.4:
        return 0.0
    return (eiv_flatten['NM'] + eiv_flatten['BG'] + eiv_flatten['CL']) / 3.

def create_supervised_trainer(model, optimizer, loss_fn, train_ids, device):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

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

def create_supervised_evaluator(model, metrics, device):
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    def _inference(engine, batch):
        model.eval()
        data, view_, status_, id_, batch_frames = batch
        batch_frames = torch.tensor(batch_frames, device=device)
        data = data.to(device)

        _, neck, _ = model(data, batch_frames)
        
        return (neck, view_, status_, id_)
    
    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine

def do_train(
    cfg,
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    scheduler, 
    loss_fn, 
):
    logger = logging.getLogger(cfg.LOGGER.NAME)

    # save model structure first
    model_path = os.path.join(cfg.PATH.OUTPUT_DIR, cfg.PATH.EXPERIMENT_DIR, cfg.PATH.CHECKPOINT_DIR, 'structure.pt')
    torch.save(model, model_path)

    train_ids = sorted(list(set(train_loader.dataset.ids)))

    trainer = create_supervised_trainer(model, optimizer, loss_fn, train_ids, cfg.MODEL.DEVICE)
    evaluator = create_supervised_evaluator(
        model,
        metrics=dict(acc=Top_K_Acc(cfg.VAL.K)),
        device=cfg.MODEL.DEVICE,
    )
    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED
    )

    model_saver = ModelCheckpoint(
        os.path.join(cfg.PATH.OUTPUT_DIR, cfg.PATH.EXPERIMENT_DIR, cfg.PATH.CHECKPOINT_DIR),
        cfg.MODEL.NAME,
        require_empty=False,
        score_function=score_eval,
        score_name='val_acc',
        n_saved=cfg.MODEL.N_SAVED,
    )

    evaluator.add_event_handler(Events.EPOCH_COMPLETED, model_saver, {cfg.MODEL.FILE_MIDDLE:model})

    @trainer.on(Events.STARTED)
    def signal_start(engine):
        logger.info('Training starts.')

    @trainer.on(Events.ITERATION_COMPLETED)
    def step_scheduler(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_info(engine):
        if engine.state.iteration % cfg.TRAIN.DISPLAY_INFO_STEP == 0:
            time_cost = fmt_secs(timer.value(), engine.state.iteration)
            time_cost = '{} hours {} mins {} secs'.format(*time_cost)

            time_expected = fmt_secs(timer.value(), cfg.TRAIN.MAX_ITERS - engine.state.iteration)
            time_expected = '{} hours {} mins {} secs'.format(*time_expected)
            logger.info('Iteration: {}, Loss: {:.4f}, already cost: {}, ETA: {}'.format(
                engine.state.iteration,
                engine.state.output,
                time_cost,
                time_expected,
            ))

    @trainer.on(Events.ITERATION_COMPLETED)
    def validate(engine):
        if engine.state.iteration % cfg.TRAIN.RECORD_STEP == 0:
            evaluator.run(val_loader)
            iiv, eiv_flatten, eiv_array = evaluator.state.metrics['acc']
            logger.info('Iteration: {}, Include identical-view cases, Got Acc of NM: {:.3f}, BG: {:.3f}, CL:{:.3f}'.format(
                engine.state.iteration,
                iiv['NM'],
                iiv['BG'],
                iiv['CL']))

            logger.info('Iteration: {}, Exclude identical-view cases, Got Acc of NM: {:.3f}, BG: {:.3f}, CL:{:.3f}'.format(
                engine.state.iteration,
                eiv_flatten['NM'],
                eiv_flatten['BG'],
                eiv_flatten['CL']))

            logger.info('Iteration: {}, Exclude identical-view cases (detailed), Got Acc of \nNM: {}, \nBG: {}, \nCL: {}'.format(
                engine.state.iteration,
                eiv_array['NM'],
                eiv_array['BG'],
                eiv_array['CL']))

    @trainer.on(Events.ITERATION_COMPLETED)
    def signal_terminate(engine):
        if engine.state.iteration == cfg.TRAIN.MAX_ITERS:
            logger.info('Reaching maximum {} epochs, break training loop now.'.format(cfg.TRAIN.MAX_ITERS))
            engine.terminate()
    
    trainer.run(train_loader)