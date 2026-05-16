import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from core.networks import CLSNet
from core.loss import SP_CAM_Loss2
from core.datasets import Dataset_with_CAM
from tools.general.Q_util import poolfeat, tile_features
from tools.general.io_utils import create_directory

import meta_infer_seg
import core.models as fcnmodel

from tools.ai.log_utils import log_print
from tools.ai.demo_utils import Timer, Average_Meter, Iterator
from tools.ai.optim_utils import PolyOptimizer
from tools.ai.torch_utils import set_seed, calculate_parameters, save_model, str2bool, get_learning_rate_from_optimizer
from tools.ai.evaluate_utils import make_cam
from tools.ai.augment_utils import (
    RandomResize_For_Segmentation,
    RandomHorizontalFlip_For_Segmentation,
    Normalize_For_Segmentation,
    RandomCrop_For_Segmentation,
    Transpose_For_Segmentation,
)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--image_size', default=480, type=int)
    parser.add_argument('--min_image_size', default=320, type=int)
    parser.add_argument('--max_image_size', default=640, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--wd', default=4e-5, type=float)
    parser.add_argument('--nesterov', default=True, type=str2bool)
    parser.add_argument('--print_ratio', default=0.1, type=float)
    parser.add_argument('--clamp_rate', default=0.001, type=float)
    parser.add_argument('--ig_th', default=0.1, type=float)
    parser.add_argument('--th', default=0.6, type=float)
    parser.add_argument(
        '--Qmodel_path',
        default='/media/ders/sdd1/XS/pipeline/weights/SCN/Q_img_2025_03_12_21_18_51.pth',
        type=str,
    )
    parser.add_argument('--SSTB', default=False, type=str2bool)
    parser.add_argument('--beta', default=16, type=float)
    parser.add_argument('--afflossPara', default=0.3, type=float)
    parser.add_argument('--gpu', default='1', type=str)
    parser.add_argument(
        '--dataset',
        default='PriMETA',
        type=str,
        choices=['PubBUSI', 'PriBUTS', 'PubDB', 'PriMETA'],
    )
    parser.add_argument('--domain', default='fold_1_train', type=str)
    parser.add_argument(
        '--expName',
        default='CAM_SOTA/CAM_IPC/META/CAM_train_PriMETA',
        type=str,
    )
    parser.add_argument('--compName', default='CAM_IPC', type=str)
    parser.add_argument('--dist_hw', default=10, type=int)
    parser.add_argument('--dist_the', default=10, type=int)
    parser.add_argument('--patch_number', default=9, type=int)
    args, _ = parser.parse_known_args()
    return args


def main(args):
    set_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    time_string = time.strftime('%Y_%m_%d_%H_%M_%S')

    print(args.expName)
    log_dir = create_directory(f'./experiments/logs/{args.expName}/')
    model_dir = create_directory(f'./experiments/models/{args.expName}/')

    log_path = f'{log_dir}/{args.compName}_{time_string}.txt'
    model_path = f'{model_dir}/{args.compName}_{time_string}.pth'

    log_func = lambda string='': log_print(string, log_path)

    log_func(f'afflossPara: {args.afflossPara}')
    log_func(f'dist_the: {args.dist_the}')
    log_func(f'patch_number: {args.patch_number}')
    log_func(f'dist_hw: {args.dist_hw}')
    log_func(f'[i] {args.expName}')
    log_func(str(args))

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        RandomResize_For_Segmentation(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip_For_Segmentation(),
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        RandomCrop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation(),
    ])

    data_dir = '/media/ders/sda1/XS/dataset/Meta/process_data/'
    saliency_dir = '/media/ders/sda1/XS/dataset/Meta/process_data/SegmentationClass/'

    train_dataset = Dataset_with_CAM(
        data_dir,
        saliency_dir,
        args.domain,
        train_transform,
        _dataset=args.dataset,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )

    log_func(f'[i] mean values is {imagenet_mean}')
    log_func(f'[i] std values is {imagenet_std}')
    log_func(f'[i] train_transform is {train_transform}')

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration

    log_func(f'[i] log_iteration : {log_iteration:,}')
    log_func(f'[i] val_iteration : {val_iteration:,}')
    log_func(f'[i] max_iteration : {max_iteration:,}')

    model = CLSNet(args.backbone, num_classes=3).cuda()
    model.train()
    log_func(f'[i] Total Params: {calculate_parameters(model):.2f}M')
    log_func()
    log_func(args.gpu)

    gpu_count = len(args.gpu.split(','))
    save_model_fn = lambda: save_model(model, model_path, parallel=gpu_count > 1)

    evaluator = meta_infer_seg.evaluator(
        args.dataset,
        domain=args.domain,
        SSTB=args.SSTB,
        refine_list=[0],
    )

    param_groups = model.get_parameter_groups()
    params = [
        {'params': param_groups[0], 'lr': 1 * args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0},
    ]
    optimizer = PolyOptimizer(
        params,
        lr=args.lr,
        momentum=0.5,
        weight_decay=args.wd,
        max_step=max_iteration,
        nesterov=args.nesterov,
    )

    if gpu_count > 1:
        log_func(f'[i] the number of gpu : {gpu_count}')
        model = nn.DataParallel(model)

    if args.SSTB:
        q_model = fcnmodel.SpixelNet1l_bn().cuda()
        q_model.load_state_dict(torch.load(args.Qmodel_path))
        q_model.eval()
    else:
        q_model = None

    lossfn_ipc = torch.nn.DataParallel(SP_CAM_Loss2(args=args)).cuda()

    train_timer = Timer()
    eval_timer = Timer()
    train_meter = Average_Meter(['loss', 'cls_loss', 'gc_loss'])
    train_iterator = Iterator(train_loader)

    best_valid_miou = -1
    for iteration in range(max_iteration):
        images, _, labels, _ = train_iterator.get()
        images = images.cuda()
        labels = labels.cuda()

        prob = q_model(images) if args.SSTB else None
        logits, logitsmin, _ = model(images, pcm=0)

        _, _, h, w = logits.shape
        img_mask = F.interpolate(images.float(), size=(h, w))
        img_mask = img_mask.float().sum(dim=1, keepdim=True) != 0

        tagpred = logitsmin
        cls_loss = F.multilabel_soft_margin_loss(
            tagpred[:, 1:].view(tagpred.size(0), -1),
            labels[:, 1:],
        )
        mask = labels.unsqueeze(2).unsqueeze(3).cuda()
        fg_cam = make_cam(logits[:, 1:]) * mask[:, 1:]

        if args.SSTB:
            target_feat = poolfeat(images, prob, 16, 16).cuda()
        else:
            target_feat = F.interpolate(
                images.float(),
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            )

        target_feat = target_feat.detach() * img_mask
        target_feat_tile = tile_features(target_feat, args.patch_number)
        fg_cam_tile = tile_features(fg_cam, args.patch_number)

        gc_loss = lossfn_ipc(fg_cam_tile, target_feat_tile).mean() * args.afflossPara
        loss = cls_loss + gc_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        if args.dataset != 'PubDB' and (iteration + 1) % log_iteration == 0:
            train_meter.add({
                'loss': loss.item(),
                'gc_loss': gc_loss.item(),
                'cls_loss': cls_loss.item(),
            })
            avg_loss, avg_cls_loss, avg_gc_loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))

            log_data = {
                'iteration': iteration + 1,
                'learning_rate': learning_rate,
                'loss': avg_loss,
                'cls_loss': avg_cls_loss,
                'gc_loss': avg_gc_loss,
                'time': train_timer.tok(clear=True),
            }

            log_func(
                '[i] iteration={iteration:,}, learning_rate={learning_rate:.4f}, '
                'loss={loss:.4f}, cls_loss={cls_loss:.4f}, gc_loss={gc_loss:.4f}, '
                'time={time:.0f}sec'.format(**log_data)
            )

        if (iteration + 1) % val_iteration == 0:
            miou, para = evaluator.evaluate(
                model,
                q_model,
                beta=args.beta,
                ite=args.th,
                dCRF_iter=0,
            )[0]

            if miou < 35:
                log_func(f'miou is too low{miou}')

            refine_num, threshold = para
            if best_valid_miou == -1 or best_valid_miou < miou:
                best_valid_miou = miou
                if miou > 22:
                    save_model_fn()
                    log_func('[i] save model')

            eval_data = {
                'iteration': iteration + 1,
                'threshold': threshold,
                'refine_num': refine_num,
                'mIoU': miou,
                'best_valid_mIoU': best_valid_miou,
                'time': eval_timer.tok(clear=True),
            }

            log_func(
                '[i] iteration={iteration:,}, mIoU={mIoU:.2f}%, '
                'best_valid_mIoU={best_valid_mIoU:.2f}%, threshold={threshold:.2f}%, '
                'refine_num={refine_num:.0f}, time={time:.0f}sec'.format(**eval_data)
            )


if __name__ == '__main__':
    args = get_params()
    print(args.domain)
    print(str(args))
    main(args)
