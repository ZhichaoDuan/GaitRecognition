import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import time
from utils.format_seconds import fmt_secs

def do_train(model, optimizer, cfg, train_loader, loss, iteration=0):
    logger = logging.getLogger(cfg.LOGGER.NAME)
    logger.info('Training starts.')
    # save the model structure for one time
    f = os.path.join(cfg.OUTPUT_DIR, cfg.EXPERIMENT, cfg.CHECKPOINT_DIR, 'structure.pt')
    if not os.path.exists(f):
        logger.info('No structure file detected, serialize structure of model and optimizer into %s', f)
        torch.save(
            [model,optimizer], 
            f
        )
    # retrive the weight from serielized file
    if iteration != 0:
        logger.info('Continuing training process from %d', iteration)
        logger.info('Loading weight file now')
        f = os.path.join(cfg.OUTPUT_DIR,cfg.EXPERIMENT,cfg.CHECKPOINT_DIR,'{}_{}.pt'.format(cfg.MODEL.NAME,iteration))
        (model_weight, opt_weight) = torch.load(f)
        model.load_state_dict(model_weight)
        optimizer.load_state_dict(opt_weight)
    
    full_loss_record = []
    hard_loss_record = []
    mean_dist_record = []
    full_loss_nm_record = []

    model = nn.DataParallel(model.float()).cuda()
    loss = nn.DataParallel(loss.float()).cuda()

    model.train()
    train_ids = list(set(train_loader.dataset.ids))
    train_ids = sorted(train_ids)

    logger.info('Entering training loop now.')
    for data, views, status, ids in train_loader:
        _start = time.time()
        iteration += 1

        optimizer.zero_grad()

        data = data.cuda()

        ids_gt = [train_ids.index(i) for i in ids]
        ids_gt = torch.Tensor(ids_gt)
        
        if cfg.MODEL.BNNECK:
            feature, neck, _ = model(data)
            cls_ids_gt = ids_gt.unsqueeze(1).repeat(1, neck.size(1))
            cls_ids_gt_flatten = cls_ids_gt.reshape(-1).long().cuda()
            neck = neck.reshape(-1, neck.size(2))
            cls_loss = F.cross_entropy(neck, cls_ids_gt_flatten)         
        else:
            feature = model(data)
        feature = feature.permute(1, 0, 2).contiguous()
        ids_gt = ids_gt.unsqueeze(0).repeat(feature.size(0), 1).cuda()

        _container = loss(feature, ids_gt)
        full_loss, hard_loss, mean_dist, full_loss_nm = _container
        
        if cfg.TRAIN.TRIPLET_LOSS.TYPE == 'full':
            loss_chosen = full_loss.mean()
        elif cfg.TRAIN.TRIPLET_LOSS.TYPE == 'hard':
            loss_chosen = hard_loss.mean()

        if cfg.MODEL.BNNECK:
            loss_chosen = loss_chosen + cls_loss / 10.

        full_loss_record.append(full_loss.mean().data.cpu().numpy())
        hard_loss_record.append(hard_loss.mean().data.cpu().numpy())
        mean_dist_record.append(mean_dist.mean().data.cpu().numpy())
        full_loss_nm_record.append(full_loss_nm.mean().data.cpu().numpy())
        
        loss_chosen.backward()
        optimizer.step()

        _end = time.time()

        if iteration % cfg.TRAIN.RECORD_STEP == 0:
            # save weight first
            f = os.path.join(
                    cfg.OUTPUT_DIR,
                    cfg.EXPERIMENT,
                    cfg.CHECKPOINT_DIR,
                    '{}_{}.pt'.format(
                        cfg.MODEL.NAME,
                        iteration
                    )
                )
            logger.info('saving weight file of iteration %d into %s', iteration, f)

            torch.save(
                [model.state_dict(), optimizer.state_dict()], 
                f
            )

        if iteration % cfg.TRAIN.DISPLAY_INFO_STEP == 0:
            logger.info('Iteration %d', iteration)
            logger.info('Hard triplet loss value is '+cfg.DISPLAY_FLOAT_FORMAT, np.mean(hard_loss_record))
            logger.info('Full triplet loss value is '+cfg.DISPLAY_FLOAT_FORMAT, np.mean(full_loss_record))
            logger.info('Number of positive full loss entries is '+cfg.DISPLAY_FLOAT_FORMAT, np.mean(full_loss_nm_record))
            logger.info('Mean dist is '+cfg.DISPLAY_FLOAT_FORMAT, np.mean(mean_dist_record))
            logger.info('Current learning rate is '+cfg.DISPLAY_FLOAT_FORMAT, optimizer.param_groups[0]['lr'])
            logger.info('Current triplet loss type is %s', cfg.TRAIN.TRIPLET_LOSS.TYPE)
            if cfg.TRAIN.TRIPLET_LOSS.TYPE == 'full':
                logger.info('Current full triplet loss value is '+cfg.DISPLAY_FLOAT_FORMAT, full_loss.mean().data.cpu().numpy().tolist())
            elif cfg.TRAIN.TRIPLET_LOSS.TYPE == 'hard':
                logger.info('Current hard triplet loss value is '+cfg.DISPLAY_FLOAT_FORMAT, hard_loss.mean().data.cpu().numpy().tolist())
            if cfg.MODEL.BNNECK:
                logger.info('Current cross-entropy loss is '+cfg.DISPLAY_FLOAT_FORMAT, cls_loss.data.cpu().numpy().tolist()) 
            hard_loss_record = []
            full_loss_nm_record = []
            full_loss_record = []
            mean_dist_record = []

            period = _end - _start
            h,m,s = fmt_secs(period, cfg.TRAIN.MAX_ITERS-iteration)
            logger.info('Estimated time of finish is {} hours {} mins {} secs.'.format(h, m, s))
        
        if iteration == cfg.TRAIN.MAX_ITERS:
            logger.info('Reaching maximum epochs, break training loop now.')
            break