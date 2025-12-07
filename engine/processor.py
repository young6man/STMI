import logging
import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP
from torch.cuda import amp
import torch.distributed as dist


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS
    logging.getLogger().setLevel(logging.INFO)
    logger = logging.getLogger("STMI.train")
    logger.info('start training')
    writer = SummaryWriter('../runs/{}'.format(cfg.OUTPUT_DIR.split('/')[-1]))
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, img_path, text) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device),
                   'mask': img['mask'].to(device)}
            text = {'rgb_text': text['rgb_text'].to(device),
                    'ni_text': text['ni_text'].to(device),
                    'ti_text': text['ti_text'].to(device)}
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                output = model(image=img, text=text, label=target, cam_label=target_cam, view_label=target_view,
                               writer=writer, epoch=epoch, img_path=img_path)
                loss = 0
                if len(output) % 2 == 1:
                    index = len(output) - 1
                    for i in range(0, index, 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp
                    if not isinstance(output[-1], dict):
                        loss = loss + output[-1]
                    else:
                        num_region = output[-1]['num']
                        for i in range(num_region):
                            loss = loss + (1 / num_region) * loss_fn(score=output[-1][f'score_{i}'],
                                                                     feat=output[-1][f'feat_{i}'],
                                                                     target=target, target_cam=target_cam)
                else:
                    for i in range(0, len(output), 2):
                        loss_tmp = loss_fn(score=output[i], feat=output[i + 1], target=target, target_cam=target_cam)
                        loss = loss + loss_tmp
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(output, list):
                acc = (output[0][0].max(1)[1] == target).float().mean()
            else:
                acc = (output[0].max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))


        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                        .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger)
            else:
                mAP, cmc = training_neat_eval(cfg, model, val_loader, device, evaluator, epoch, logger, writer=None)
                writer.add_scalar('RGBNT201/mAP', mAP, epoch)
                writer.add_scalar('RGBNT201/Rank-1', cmc[0], epoch)
                writer.add_scalar('RGBNT201/Rank-5', cmc[4], epoch)
                writer.add_scalar('RGBNT201/Rank-10', cmc[9], epoch)
                if mAP >= best_index['mAP']:
                    best_index['mAP'] = mAP
                    best_index['Rank-1'] = cmc[0]
                    best_index['Rank-5'] = cmc[4]
                    best_index['Rank-10'] = cmc[9]
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'best.pth'))
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                logger.info("~" * 50)



def do_inference(cfg,
                 model,
                 val_loader,
                 num_query, logger):
    device = "cuda"
    logger.info("Enter inferencing")

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    for n_iter, (img, pid, camid, camids, target_view, imgpath, text) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device),
                   'mask': img['mask'].to(device)}
            text = {'rgb_text': text['rgb_text'].to(device),
                    'ni_text': text['ni_text'].to(device),
                    'ti_text': text['ti_text'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(image=img, text=text, cam_label=camids, view_label=target_view, img_path=imgpath)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feat, pid, camid, imgpath))

    if cfg.DATASETS.NAMES == "RGBNT201":
        mAP, cmc = compute_log(evaluator=evaluator, logger=logger, query=['LOCAL'], gallery=['LOCAL'])
    else:
        mAP, cmc = compute_log(evaluator=evaluator, logger=logger,
                        query=['V_RGB', 'V_NIR', 'V_TIR', 'T', 'T', 'T'],
                        gallery=['V_RGB', 'V_NIR', 'V_TIR', 'T', 'T', 'T'])

    return mAP, cmc


def compute_log(evaluator, logger, query, gallery, epoch=0):
    cmc, mAP, _, _, _, _, _ = evaluator.compute(query=query, gallery=gallery)
    logger.info("Validation Results - Epoch: {}".format(epoch))
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("~" * 50)
    torch.cuda.empty_cache()
    return mAP, cmc


def training_neat_eval(cfg,
                       model,
                       val_loader,
                       device,
                       evaluator, epoch, logger, return_pattern=1, writer=None):
    evaluator.reset()
    model.eval()
    for n_iter, (img, pid, camid, camids, target_view, imgpath, text) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device),
                   'mask': img['mask'].to(device)}
            text = {'rgb_text': text['rgb_text'].to(device),
                    'ni_text': text['ni_text'].to(device),
                    'ti_text': text['ti_text'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(image=img, text=text, cam_label=camids, view_label=target_view, return_pattern=return_pattern,
                         img_path=imgpath, writer=writer, epoch=epoch)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, pid, camid, scenceids, imgpath))
            else:
                evaluator.update((feat, pid, camid, imgpath))

    if cfg.DATASETS.NAMES == "RGBNT201":
        mAP, cmc = compute_log(evaluator=evaluator, logger=logger, query=['LOCAL'], gallery=['LOCAL'], epoch=epoch)
    else:
        mAP, cmc = compute_log(evaluator=evaluator, logger=logger,
                        query=['V_RGB', 'V_NIR', 'V_TIR', 'T', 'T', 'T'],
                        gallery=['V_RGB', 'V_NIR', 'V_TIR', 'T', 'T', 'T'], epoch=epoch)
    return mAP, cmc

