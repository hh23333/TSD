import logging
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import w_R1_mAP_eval
from utils.metrics_new import w_R1_mAP_eval_new
from torch.cuda import amp
import torch.distributed as dist
from .vis_attn import visualize_attn_decoder, GradCAM, visualize_cam, rollout_attn, mean_attn, visualize_attn_max
from loss.parsing_loss import calculate_mask_loss
from torchvision.ops.focal_loss import sigmoid_focal_loss
from loss.diverse_loss import diverse_loss_w

def do_train_tsd(cfg,
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

    # device = "cpu"
    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    gcls_loss_meter = AverageMeter()
    ccls_loss_meter = AverageMeter()
    cmcls_loss_meter = AverageMeter()
    acls_loss_meter = AverageMeter()
    gtri_loss_meter = AverageMeter()
    ptri_loss_meter = AverageMeter()
    pmtri_loss_meter = AverageMeter()
    atri_loss_meter = AverageMeter()
    parsing_loss_meter = AverageMeter()
    mask_loss_meter = AverageMeter()
    mimic_loss_meter = AverageMeter()
    div_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = w_R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        gcls_loss_meter.reset()
        ccls_loss_meter.reset()
        cmcls_loss_meter.reset()
        acls_loss_meter.reset()
        gtri_loss_meter.reset()
        ptri_loss_meter.reset()
        pmtri_loss_meter.reset()
        atri_loss_meter.reset()
        parsing_loss_meter.reset()
        mask_loss_meter.reset()
        mimic_loss_meter.reset()
        div_loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view, mask) in enumerate(train_loader):  # mask: (B, 4, H, W)
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            mask = mask.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score_dict, feat, part_visb_logits, visb_gt, part_logits, mask_gt = model(img, target, mask, cam_label=target_cam, view_label=target_view)
                mask_part_feat_d = feat.pop('part_md')
                loss_dict = loss_fn(score_dict, feat, target)
                loss = 0
                for _, v_loss in loss_dict.items():
                    loss += v_loss
                loss_parsing = sigmoid_focal_loss(part_visb_logits, visb_gt, reduction='mean')
                loss += loss_parsing
                # mimic loss
                cat_feat = feat['part'].flatten(1,2)
                aux_feat = feat['aux'].flatten(1,2)  # (B, P*C)
                cat_feat_norm = F.normalize(cat_feat, dim=1)
                aux_feat_norm = F.normalize(aux_feat, dim=1).detach()
                loss_mimic = (1 - (cat_feat_norm * aux_feat_norm).sum(dim=1)).mean()
                loss += loss_mimic

                mask_loss_weight = cfg.MODEL.MASK_LOSS_WEIGHT
                loss_mask = mask_loss_weight*calculate_mask_loss(part_logits, mask_gt, label_smoothing=0.1)
                loss += loss_mask
                if cfg.MODEL.WITH_DIVERSE_LOSS:
                    aux_part_feat = feat['aux']
                    div_loss = diverse_loss_w(aux_part_feat, part_type=cfg.INPUT.MASK_PREPROCESS)
                    div_loss += diverse_loss_w(mask_part_feat_d, part_type=cfg.INPUT.MASK_PREPROCESS)
                    loss += div_loss

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            score = score_dict['global']

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            gcls_loss_meter.update(loss_dict['cls_global'].item(), img.shape[0])
            ccls_loss_meter.update(loss_dict['cls_cat'].item(), img.shape[0])
            cmcls_loss_meter.update(loss_dict['cls_cat_m'].item(), img.shape[0])
            acls_loss_meter.update(loss_dict['cls_aux'].item(), img.shape[0])
            gtri_loss_meter.update(loss_dict['tri_global'].item(), img.shape[0])
            ptri_loss_meter.update(loss_dict['tri_part'].item(), img.shape[0])
            pmtri_loss_meter.update(loss_dict['tri_part_m'].item(), img.shape[0])
            atri_loss_meter.update(loss_dict['tri_aux'].item(), img.shape[0])
            parsing_loss_meter.update(loss_parsing.item(), img.shape[0])
            mask_loss_meter.update(loss_mask.item(), img.shape[0])
            mimic_loss_meter.update(loss_mimic.item(), img.shape[0])
            div_loss_meter.update(div_loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, gcls_Loss: {:.3f}, ccls_Loss: {:.3f}, cmcls_Loss: {:.3f}, acls_Loss: {:.3f}, gtri_Loss: {:.3f}, ptri_Loss: {:.3f}, pmtri_Loss: {:.3f}, atri_Loss: {:.3f}, parsing_Loss: {:.3f}, mimic_Loss: {:.3f}, mask: {:.3f}, div_Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, gcls_loss_meter.avg, ccls_loss_meter.avg, cmcls_loss_meter.avg, acls_loss_meter.avg, gtri_loss_meter.avg, ptri_loss_meter.avg, pmtri_loss_meter.avg, atri_loss_meter.avg, parsing_loss_meter.avg,
                                            mimic_loss_meter.avg,
                                            mask_loss_meter.avg,
                                            # vs_loss_meter.avg,
                                            div_loss_meter.avg, 
                                            acc_meter.avg, base_lr))
            else:
                if (n_iter + 1) % log_period == 0:
                    base_lr = scheduler._get_lr(epoch)[0]
                    # logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, gcls_Loss: {:.3f}, ccls_Loss: {:.3f}, gtri_Loss: {:.3f}, ptri_Loss: {:.3f}, vs_Loss: {:.3f}, pc_Loss: {:.3f} Acc: {:.3f}, Base Lr: {:.2e}"
                    logger.info("Epoch[{}] Iter[{}/{}] Loss: {:.3f}, gcls_Loss: {:.3f}, ccls_Loss: {:.3f}, cmcls_Loss: {:.3f}, acls_Loss: {:.3f}, gtri_Loss: {:.3f}, ptri_Loss: {:.3f}, pmtri_Loss: {:.3f}, atri_Loss: {:.3f}, parsing_Loss: {:.3f}, mimic_Loss: {:.3f}, mask: {:.3f}, div_Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader), loss_meter.avg, gcls_loss_meter.avg, ccls_loss_meter.avg, cmcls_loss_meter.avg, acls_loss_meter.avg, gtri_loss_meter.avg, ptri_loss_meter.avg, pmtri_loss_meter.avg, atri_loss_meter.avg, parsing_loss_meter.avg,
                                            mimic_loss_meter.avg,
                                            mask_loss_meter.avg,
                                            # vs_loss_meter.avg,
                                            div_loss_meter.avg, 
                                            acc_meter.avg, base_lr))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

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
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat, vs = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vs, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat, vs = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vs, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

def do_inference_part(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = w_R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, occ_l, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            # occ_l = occ_l.to(device)
            feat, visb = model(img, cam_label=camids)
            # evaluator.update((feat[:, :1], visb[:, :1], pid, camid))
            evaluator.update((feat, visb, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

def do_inference_part_new(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = w_R1_mAP_eval_new(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, occ_l, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            # occ_l = occ_l.to(device)
            feat, visb = model(img, cam_label=camids)
            evaluator.update((feat, visb, pid, camid, occ_l))
            img_path_list.extend(imgpath)

    cmc, mAP, cmc_NPO, mAP_NPO, cmc_NTP, mAP_NTP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    logger.info("NPO_mAP: {:.1%}".format(mAP_NPO))
    for r in [1, 5, 10]:
        logger.info("NPO CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_NPO[r - 1]))
    logger.info("NTP_mAP: {:.1%}".format(mAP_NTP))
    for r in [1, 5, 10]:
        logger.info("NTP CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_NTP[r - 1]))
    return cmc[0], cmc[4]

def do_vis_attn(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = w_R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    output_dir = os.path.join(cfg.OUTPUT_DIR, 'vis_attn_')
    os.makedirs(output_dir, exist_ok=True)

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat, visb, attn_out = model(img, cam_label=camids, view_label=target_view, return_attn=True)

            # visualize_attn(attn_out[-1:], imgpath, vis_dir=output_dir)
            # joint_attn = rollout_attn(attn_out)
            joint_attn = mean_attn([attn_out])
            # print(imgpath)
            visualize_attn_decoder(joint_attn, imgpath, num_attr=cfg.MODEL.NUM_PARTS+1, vis_dir=output_dir, score=visb)
            # visualize_attn_max(joint_attn, imgpath, num_attr=4, vis_dir=output_dir)
            # evaluator.update((feat, visb, pid, camid))
            evaluator.update((feat[:, :2], visb[:, :2], pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]