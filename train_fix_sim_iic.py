from __future__ import print_function
import random

import time
import argparse
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import WideResnet
from datasets.cifar import get_train_loader, get_train_loader_mix, get_val_loader
from label_guessor import LabelGuessor
from lr_scheduler import WarmupCosineLrScheduler
from models.ema import EMA

from utils import accuracy, setup_default_logging, interleave, de_interleave

from utils import AverageMeter

from modules.nt_xent import NT_Xent

from modules.IIC_losses import IIC_loss, compute_joint

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def mi(P):
    eps = torch.finfo(P.dtype).eps
    P[P < eps] = eps
    m0 = torch.sum(P, dim=0, keepdim=True)
    m1 = torch.sum(P, dim=1, keepdim=True)
    return torch.sum((P.log2() - m0.log2() - m1.log2()) * P)


def mi_loss(x, x_t):
    """make a joint distribution from the batch """
    x = F.softmax(x, dim=1)
    x_t = F.softmax(x_t, dim=1)
    P = torch.matmul(x.T, x_t) / x.size(0)

    """symmetrical"""
    P = (P + P.T) / 2.0

    """return negative of mutual information as loss"""
    return - mi(P), P

def set_model(args):
    model = WideResnet(n_classes=10 if args.dataset == 'CIFAR10' else 100,
                       k=args.wresnet_k, n=args.wresnet_n)  # wresnet-28-2

    # name = 'simclr_trained_good_h2.pt'
    # model.load_state_dict(torch.load(name))
    # print('model loaded')

    model.train()
    model.cuda()

    criteria_x = nn.CrossEntropyLoss().cuda()
    criteria_u = nn.CrossEntropyLoss(reduction='none').cuda()
    criteria_z = NT_Xent(args.batchsize, args.temperature, args.mu)
    return model, criteria_x, criteria_u, criteria_z


def train_one_epoch(epoch,
                    model,
                    criteria_x,
                    criteria_u,
                    criteria_z,
                    optim,
                    lr_schdlr,
                    ema,
                    dltrain_x,
                    dltrain_u,
                    dltrain_f,
                    lb_guessor,
                    lambda_u,
                    lambda_s,
                    n_iters,
                    logger,
                    bt,
                    mu
                    ):
    model.train()
    # loss_meter, loss_x_meter, loss_u_meter, loss_u_real_meter = [], [], [], []
    loss_meter = AverageMeter()
    loss_x_meter = AverageMeter()
    loss_u_meter = AverageMeter()
    loss_u_real_meter = AverageMeter()
    loss_simclr_meter = AverageMeter()
    # the number of correctly-predicted and gradient-considered unlabeled data
    n_correct_u_lbs_meter = AverageMeter()
    # the number of gradient-considered strong augmentation (logits above threshold) of unlabeled samples
    n_strong_aug_meter = AverageMeter()
    mask_meter = AverageMeter()

    epoch_start = time.time()  # start time
    # dl_x, dl_u, dl_f = iter(dltrain_x), iter(dltrain_u), iter(dltrain_f)
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)
    for it in range(n_iters):
        ims_x_weak, ims_x_strong, lbs_x = next(dl_x)
        ims_u_weak, ims_u_strong, lbs_u_real = next(dl_u) # transformaciones de fixmatch
        # ims_s_weak, ims_s_strong, lbs_s_real = next(dl_f) # con transformaciones de simclr

        lbs_x = lbs_x.cuda()
        lbs_u_real = lbs_u_real.cuda()

        # --------------------------------------
        # imgs = torch.cat([ims_x_weak, ims_u_weak, ims_u_strong, ims_s_weak, ims_s_strong], dim=0).cuda()
        imgs = torch.cat([ims_x_weak, ims_u_weak, ims_u_strong], dim=0).cuda()
        # imgs = interleave(imgs, 4 * mu + 1)
        imgs = interleave(imgs, 2 * mu + 1)
        logits, logit_z, _ = model(imgs)
        # logits = de_interleave(logits, 4 * mu + 1)
        logits = de_interleave(logits, 2 * mu + 1)

        # SEPARACION DE LOGITS PARA ETAPA SUPERVISADA DE FIXMATCH
        logits_x = logits[:bt]
        # SEPARACION DE LOGITS PARA ETAPA NO SUPERVISADA DE FIXMATCH
        logits_u_w, logits_u_s = torch.split(logits[bt:], bt * mu)
        # _, _, logits_s_w, logits_s_s = torch.split(logit_z[bt:], bt * mu)

        # calculo de la mascara con transformacion debil de fixmatch
        with torch.no_grad():
            probs = torch.softmax(logits_u_w, dim=1)
            scores, lbs_u_guess = torch.max(probs, dim=1)
            mask = scores.ge(0.95).float()

        # calcular perdida de fixmatch
        # loss_s = criteria_z(logits_s_w, logits_s_s)
        loss_u = (criteria_u(logits_u_s, lbs_u_guess) * mask).mean()
        loss_x = criteria_x(logits_x, lbs_x)

        loss = loss_x + loss_u * lambda_u# + loss_s * lambda_s

        loss_u_real = (F.cross_entropy(logits_u_s, lbs_u_real) * mask).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        ema.update_params()
        lr_schdlr.step()

        loss_meter.update(loss.item())
        loss_x_meter.update(loss_x.item())
        loss_u_meter.update(loss_u.item())
        loss_u_real_meter.update(loss_u_real.item())
        # loss_simclr_meter.update(loss_s.item())
        mask_meter.update(mask.mean().item())

        corr_u_lb = (lbs_u_guess == lbs_u_real).float() * mask
        n_correct_u_lbs_meter.update(corr_u_lb.sum().item())
        n_strong_aug_meter.update(mask.sum().item())

        if (it + 1) % 512 == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)

            # logger.info("epoch:{}, iter: {}. loss: {:.4f}. loss_u: {:.4f}. loss_x: {:.4f}. loss_u_real: {:.4f}. "
            #             "n_correct_u: {:.2f}/{:.2f}. loss_s: {:.4f}"
            #             "Mask:{:.4f} . LR: {:.4f}. Time: {:.2f}".format(
            #     epoch, it + 1, loss_meter.avg, loss_u_meter.avg, loss_x_meter.avg, loss_u_real_meter.avg,
            #     n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, loss_simclr_meter.avg, mask_meter.avg, lr_log, t))

            logger.info("epoch:{}, iter: {}. loss: {:.4f}. loss_u: {:.4f}. loss_x: {:.4f}. loss_u_real: {:.4f}. "
                        "n_correct_u: {:.2f}/{:.2f}."
                        "Mask:{:.4f} . LR: {:.4f}. Time: {:.2f}".format(
                epoch, it + 1, loss_meter.avg, loss_u_meter.avg, loss_x_meter.avg, loss_u_real_meter.avg,
                n_correct_u_lbs_meter.avg, n_strong_aug_meter.avg, mask_meter.avg, lr_log, t))

            epoch_start = time.time()

    ema.update_buffer()
    return loss_meter.avg, loss_x_meter.avg, loss_u_meter.avg,\
           loss_u_real_meter.avg, mask_meter.avg


def train_one_epoch_simclr(epoch,
                           model,
                           criteria_z,
                           optim,
                           lr_schdlr,
                           ema,
                           dltrain_f,
                           lambda_s,
                           n_iters,
                           logger,
                           bt,
                           mu
                           ):
    model.train()

    loss_meter = AverageMeter()
    loss_simclr_meter = AverageMeter()

    epoch_start = time.time()  # start time
    dl_f = iter(dltrain_f)
    for it in range(n_iters):
        ims_s_weak, ims_s_strong, lbs_s_real = next(dl_f) # con transformaciones de simclr

        imgs = torch.cat([ims_s_weak, ims_s_strong], dim=0).cuda()
        imgs = interleave(imgs, 2 * mu)
        logits, logit_z, _ = model(imgs)
        logits_z = de_interleave(logit_z, 2 * mu)

        # SEPARACION DE ULTIMAS REPRESENTACIONES PARA SIMCLR
        logits_s_w_z, logits_s_s_z = torch.split(logits_z, bt * mu)

        loss_s = criteria_z(logits_s_w_z, logits_s_s_z)

        loss = loss_s

        optim.zero_grad()
        loss.backward()
        optim.step()
        ema.update_params()
        lr_schdlr.step()

        loss_meter.update(loss.item())
        loss_simclr_meter.update(loss_s.item())

        if (it + 1) % 512 == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)

            logger.info("epoch:{}, iter: {}. loss: {:.4f}. "
                        " loss_simclr: {:.4f}. LR: {:.4f}. Time: {:.2f}".format(
                epoch, it + 1, loss_meter.avg,
                loss_simclr_meter.avg, lr_log, t))

            epoch_start = time.time()

    ema.update_buffer()
    return loss_meter.avg, loss_simclr_meter.avg, model


def train_one_epoch_iic(epoch,
                        model,
                        optim,
                        lr_schdlr,
                        ema,
                        dltrain_f,
                        n_iters,
                        logger,
                        bt,
                        mu
                        ):
    model.train()
    loss_meter = AverageMeter()
    loss_iic_meter = AverageMeter()

    epoch_start = time.time()  # start time
    dl_f = iter(dltrain_f)
    for it in range(n_iters):
        ims_s_weak, ims_s_strong, lbs_s_real = next(dl_f) # con transformaciones de simclr

        imgs = torch.cat([ims_s_weak, ims_s_strong], dim=0).cuda()
        imgs = interleave(imgs, 2 * mu)
        _, _, logits_iic = model(imgs)
        logits_iic = de_interleave(logits_iic, 2 * mu)

        # SEPARACION DE ULTIMAS REPRESENTACIONES PARA SIMCLR
        logits_iic_w, logits_iic_s = torch.split(logits_iic, bt * mu)

        # loss_iic = IIC_loss(logits_s_w_h, logits_s_s_h)
        loss_iic, P = mi_loss(logits_iic_w, logits_iic_s)

        loss = loss_iic

        optim.zero_grad()
        loss.backward()
        optim.step()
        ema.update_params()
        lr_schdlr.step()

        loss_meter.update(loss.item())
        loss_iic_meter.update(loss_iic.item())

        if (it + 1) % 512 == 0:
            t = time.time() - epoch_start

            lr_log = [pg['lr'] for pg in optim.param_groups]
            lr_log = sum(lr_log) / len(lr_log)

            logger.info("epoch:{}, iter: {}. loss: {:.4f}. "
                        " loss_iic: {:.4f}. LR: {:.4f}. Time: {:.2f}".format(
                epoch, it + 1, loss_meter.avg,
                loss_iic_meter.avg, lr_log, t))

            epoch_start = time.time()

    ema.update_buffer()
    return loss_meter.avg, loss_iic_meter.avg, model


def evaluate(ema, dataloader, criterion):
    # using EMA params to evaluate performance
    ema.apply_shadow()
    ema.model.eval()
    ema.model.cuda()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    # matches = []
    with torch.no_grad():
        for ims, lbs in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            logits, _, _ = ema.model(ims)
            # logits = ema.model(ims)
            loss = criterion(logits, lbs)
            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
            loss_meter.update(loss.item())
            top1_meter.update(top1.item())
            top5_meter.update(top5.item())

    # note roll back model current params to continue training
    ema.restore()
    return top1_meter.avg, top5_meter.avg, loss_meter.avg


from sklearn.linear_model import LogisticRegression


def evaluate_linear_Clf(ema, trainloader, dataloader, criterion):
    # using EMA params to evaluate performance
    ema.apply_shadow()
    ema.model.eval()
    ema.model.cuda()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    cls_model = LogisticRegression(random_state=0, max_iter=10000)

    FEATURES = []
    LABS = []

    trainloader = iter(trainloader)

    with torch.no_grad():
        print('training classifier: ', end='')
        for it in range(100):
            ims, _, labs = next(trainloader)
            ims = ims.cuda()
            _, _, features = ema.model(ims)
            features = features.cpu().numpy()
            labs = labs.cpu().numpy()

            FEATURES.append(features)
            LABS.append(labs)

        FEATURES = np.vstack(FEATURES)
        LABS = np.array(LABS).ravel()

        cls_model = cls_model.fit(FEATURES, LABS)
        print(cls_model.score(FEATURES, LABS), FEATURES.shape)

        for ims, lbs in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            _, _, features = ema.model(ims)

            predictions = cls_model.predict(features.cpu().numpy())

            logits = torch.from_numpy(np.eye(10, dtype='float')[predictions]).cuda()

            loss = criterion(logits, lbs)

            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
            loss_meter.update(loss.item())
            top1_meter.update(top1.item())
            top5_meter.update(top5.item())

    # note roll back model current params to continue training
    ema.restore()
    return top1_meter.avg, top5_meter.avg, loss_meter.avg


def evaluate_Clf(model, trainloader, dataloader, criterion):
    # using EMA params to evaluate performance
    # ema.apply_shadow()
    # ema.model.eval()
    # ema.model.cuda()

    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    cls_model = LogisticRegression(random_state=0, max_iter=10000)

    FEATURES = []
    LABS = []

    trainloader = iter(trainloader)

    with torch.no_grad():
        print('training classifier: ')
        for it in range(100):
            ims, _, labs = next(trainloader)
            ims = ims.cuda()
            _, _, features = model(ims)
            features = features.cpu().numpy()
            labs = labs.cpu().numpy()

            FEATURES.append(features)
            LABS.append(labs)

        FEATURES = np.vstack(FEATURES)
        LABS = np.array(LABS).ravel()

        print(FEATURES.shape)
        cls_model = cls_model.fit(FEATURES, LABS)
        print(cls_model.score(FEATURES, LABS))

        for ims, lbs in dataloader:
            ims = ims.cuda()
            lbs = lbs.cuda()
            _, _, features = model(ims)

            predictions = cls_model.predict(features.cpu().numpy())

            logits = torch.from_numpy(np.eye(10, dtype='float')[predictions]).cuda()

            loss = criterion(logits, lbs)

            scores = torch.softmax(logits, dim=1)
            top1, top5 = accuracy(scores, lbs, (1, 5))
            loss_meter.update(loss.item())
            top1_meter.update(top1.item())
            top5_meter.update(top5.item())

    # ema.restore()

    return


def main():
    parser = argparse.ArgumentParser(description=' FixMatch Training')
    parser.add_argument('--wresnet-k', default=2, type=int,
                        help='width factor of wide resnet')
    parser.add_argument('--wresnet-n', default=28, type=int,
                        help='depth of wide resnet')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='number of classes in dataset')
    # parser.add_argument('--n-classes', type=int, default=100,
    #                     help='number of classes in dataset')
    parser.add_argument('--n-labeled', type=int, default=40,
                        help='number of labeled samples for training')
    parser.add_argument('--n-epoches', type=int, default=1024,
                        help='number of training epoches')
    parser.add_argument('--batchsize', type=int, default=40,
                        help='train batch size of labeled samples')
    parser.add_argument('--mu', type=int, default=7,
                        help='factor of train batch size of unlabeled samples')
    parser.add_argument('--thr', type=float, default=0.95,
                        help='pseudo label threshold')
    parser.add_argument('--n-imgs-per-epoch', type=int, default=64 * 1024,
                        help='number of training images for each epoch')
    parser.add_argument('--lam-u', type=float, default=1.,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--lam-s', type=float, default=0.2,
                        help='coefficient of unlabeled loss SimCLR')
    parser.add_argument('--ema-alpha', type=float, default=0.999,
                        help='decay rate for ema module')
    parser.add_argument('--lr', type=float, default=0.03,
                        help='learning rate for training')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for optimizer')
    parser.add_argument('--seed', type=int, default=-1,
                        help='seed for random behaviors, no seed if negtive')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='temperature for loss function')
    args = parser.parse_args()

    # args.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    logger, writer = setup_default_logging(args)
    logger.info(dict(args._get_kwargs()))

    # global settings
    #  torch.multiprocessing.set_sharing_strategy('file_system')
    if args.seed > 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        # torch.backends.cudnn.deterministic = True

    n_iters_per_epoch = args.n_imgs_per_epoch // args.batchsize  # 1024
    n_iters_all = n_iters_per_epoch * args.n_epoches  # 1024 * 1024

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.n_labeled}")
    logger.info(f"  Num Epochs = {n_iters_per_epoch}")
    logger.info(f"  Batch size per GPU = {args.batchsize}")
    # logger.info(f"  Total train batch size = {args.batch_size * args.world_size}")
    logger.info(f"  Total optimization steps = {n_iters_all}")

    model, criteria_x, criteria_u, criteria_z = set_model(args)

    logger.info("Total params: {:.2f}M".format(
        sum(p.numel() for p in model.parameters()) / 1e6))

    dltrain_x, dltrain_u, dltrain_f = get_train_loader_mix(
        args.dataset, args.batchsize, args.mu, n_iters_per_epoch, L=args.n_labeled)
    dlval = get_val_loader(dataset=args.dataset, batch_size=64, num_workers=2)

    lb_guessor = LabelGuessor(thresh=args.thr)

    ema = EMA(model, args.ema_alpha)


    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        # if len(param.size()) == 1:
        if 'bn' in name:
            non_wd_params.append(param)  # bn.weight, bn.bias and classifier.bias
            # print(name)
        else:
            wd_params.append(param)


    param_list = [
        {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]

    optim_fix = torch.optim.SGD(param_list, lr=args.lr, weight_decay=args.weight_decay,
                            momentum=args.momentum, nesterov=True)
    lr_schdlr_fix = WarmupCosineLrScheduler(
        optim_fix, max_iter=n_iters_all, warmup_iter=0
    )

    train_args = dict(
        model=model,
        criteria_x=criteria_x,
        criteria_u=criteria_u,
        criteria_z=criteria_z,
        optim=optim_fix,
        lr_schdlr=lr_schdlr_fix,
        ema=ema,
        dltrain_x=dltrain_x,
        dltrain_u=dltrain_u,
        dltrain_f=dltrain_f,
        lb_guessor=lb_guessor,
        lambda_u=args.lam_u,
        lambda_s=args.lam_s,
        n_iters=n_iters_per_epoch,
        logger=logger,
        bt=args.batchsize,
        mu=args.mu
    )

    # param_list = [
    #     {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    #
    # optim_simclr = torch.optim.SGD(param_list, lr=0.5, weight_decay=args.weight_decay,
    #                                momentum=args.momentum, nesterov=False)
    #
    # lr_schdlr_simclr = WarmupCosineLrScheduler(
    #     optim_simclr, max_iter=n_iters_all, warmup_iter=0
    # )
    #
    # train_args_simclr = dict(
    #     model=model,
    #     criteria_z=criteria_z,
    #     optim=optim_simclr,
    #     lr_schdlr=lr_schdlr_simclr,
    #     ema=ema,
    #     dltrain_f=dltrain_f,
    #     lambda_s=args.lam_s,
    #     n_iters=n_iters_per_epoch,
    #     logger=logger,
    #     bt=args.batchsize,
    #     mu=args.mu
    # )

    # param_list = [
    #     {'params': wd_params}, {'params': non_wd_params, 'weight_decay': 0}]
    #
    # optim_iic = torch.optim.Adam(param_list, lr=1e-4, weight_decay=args.weight_decay)
    #
    # lr_schdlr_iic = WarmupCosineLrScheduler(
    #     optim_iic, max_iter=n_iters_all, warmup_iter=0
    # )
    #
    # train_args_iic = dict(
    #     model=model,
    #     optim=optim_iic,
    #     lr_schdlr=lr_schdlr_iic,
    #     ema=ema,
    #     dltrain_f=dltrain_f,
    #     n_iters=n_iters_per_epoch,
    #     logger=logger,
    #     bt=args.batchsize,
    #     mu=args.mu
    # )
    #
    best_acc = -1
    best_epoch = 0
    logger.info('-----------start training--------------')

    for epoch in range(args.n_epoches):
        # guardar accuracy de modelo preentrenado hasta espacio h
        top1, top5, valid_loss = evaluate_linear_Clf(ema, dltrain_x, dlval, criteria_x)
        writer.add_scalars('test/1.test_linear_acc', {'top1': top1, 'top5': top5}, epoch)

        logger.info("Epoch {}. on h space Top1: {:.4f}. Top5: {:.4f}.".
                    format(epoch, top1, top5))

        if epoch < -500:
            # entrenar feature representation simclr
            train_loss, loss_simclr, model_ = train_one_epoch_simclr(epoch, **train_args_simclr)
            writer.add_scalar('train/4.train_loss_simclr', loss_simclr, epoch)

            # entrenar iic
            # train_loss, loss_iic, model_ = train_one_epoch_iic(epoch, **train_args_iic)
            # writer.add_scalar('train/4.train_loss_iic', loss_iic, epoch)
            # evaluate_Clf(model_, dltrain_f, dlval, criteria_x)

            top1, top5, valid_loss = evaluate_linear_Clf(ema, dltrain_x, dlval, criteria_x)
            if epoch == 497:
                # save model
                name = 'simclr_trained_good_h2.pt'
                torch.save(model_.state_dict(), name)
                logger.info('model saved')

        else:
            train_loss, loss_x, loss_u, loss_u_real, mask_mean = train_one_epoch(epoch, **train_args)
            top1, top5, valid_loss = evaluate(ema, dlval, criteria_x)

            # writer.add_scalar('train/4.train_loss_simclr', loss_simclr, epoch)
            writer.add_scalar('train/2.train_loss_x', loss_x, epoch)
            writer.add_scalar('train/3.train_loss_u', loss_u, epoch)
            writer.add_scalar('train/4.train_loss_u_real', loss_u_real, epoch)
            writer.add_scalar('train/5.mask_mean', mask_mean, epoch)

        writer.add_scalars('train/1.loss', {'train': train_loss,
                                            'test': valid_loss}, epoch)
        writer.add_scalars('test/1.test_acc', {'top1': top1, 'top5': top5}, epoch)
        # writer.add_scalar('test/2.test_loss', loss, epoch)

        # best_acc = top1 if best_acc < top1 else best_acc
        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch

        logger.info("Epoch {}. Top1: {:.4f}. Top5: {:.4f}. best_acc: {:.4f} in epoch{}".
                    format(epoch, top1, top5, best_acc, best_epoch))

    writer.close()


if __name__ == '__main__':
    main()

