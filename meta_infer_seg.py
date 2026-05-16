import os
import sys
import time
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader

from core.networks import CLSNet
from core.datasets import Dataset_For_Evaluation_PUBBUSI
from tools.general.io_utils import create_directory
from tools.general.Q_util import refine_with_affmat, upfeat
from tools.dataset.voc_utils import VOClabel2colormap
from tools.ai.log_utils import log_print
from tools.ai.torch_utils import str2bool, get_numpy_from_tensor
from tools.ai.evaluate_utils import Calculator_For_mIoU, make_cam
from tools.ai.augment_utils import Normalize_For_Segmentation, Transpose_For_Segmentation
from tools.general.visualization import convert_to_tf, get_colored_mask, generate_vis, save_colored_mask

import core.models as fcnmodel
import dataset_root


parser = argparse.ArgumentParser()


def get_params():
    parser.add_argument('--dataset', default='PriMETA', type=str, choices=['PubBUSI', 'PriBUTS', 'PubDB', 'PriMETA'])
    parser.add_argument('--domain', default='fold_1_test', type=str)
    parser.add_argument('--Qmodel_path', default='/media/ders/sdd1/XS/pipeline/weights/SCN/Q_img_2025_03_12_21_18_51.pth', type=str)
    parser.add_argument('--Cmodel_path', default='/media/ders/sda1/XS/SPCAM_META/experiments/models/CAM_SOTA/CAM_IPC/META/CAM_train_PriMETA/CAM_IPC_2026_03_08_21_37_08.pth', type=str)
    parser.add_argument('--tag', default='TTTTTT', type=str)
    parser.add_argument('--tagA', default='fold1/test/CAM_CLS_IPC_GAM_Q_2026_03_08_21_37_08', type=str)
    parser.add_argument('--savepng', default=True, type=str2bool)
    parser.add_argument('--savenpy', default=True, type=str2bool)
    parser.add_argument('--SSTB', default=True, type=str2bool)
    parser.add_argument('--beta', default=14, type=int)
    parser.add_argument('--gama', default=0.6, type=float)
    parser.add_argument('--gpu', default='1', type=str)
    return parser.parse_args()


class Evaluator:
    def __init__(
        self,
        dataset='PubBUSI',
        domain='_',
        SSTB=True,
        save_np_path=None,
        savepng_path=None,
        muti_scale=False,
        th_list=None,
        refine_list=None,
    ):
        self.C_model = None
        self.Q_model = None
        self.SSTB = SSTB
        self.scale_list = [0.5, 1.0, 1.5, 2.0, -0.5, -1, -1.5, -2.0] if muti_scale else [1]
        self.th_list = th_list if th_list is not None else list(np.arange(0.05, 0.4, 0.05))
        self.refine_list = refine_list if refine_list is not None else list(range(0, 50, 10))
        self.parms = [(refine_num, th) for refine_num in self.refine_list for th in self.th_list]
        self.meterlist = [Calculator_For_mIoU(3) for _ in self.parms]
        self.save_png_path = savepng_path
        self.save_np_path = save_np_path

        if self.save_png_path is not None and not os.path.exists(self.save_png_path):
            os.mkdir(self.save_png_path)

        test_transform = transforms.Compose([
            Normalize_For_Segmentation([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            Transpose_For_Segmentation(),
        ])

        if dataset in ['PubBUSI', 'PriMETA']:
            valid_dataset = Dataset_For_Evaluation_PUBBUSI(dataset_root.PubBUSI_ROOT, domain, test_transform, dataset)
        else:
            valid_dataset = Dataset_For_Evaluation_PUBBUSI(dataset_root.PriBUTS_ROOT, domain, test_transform, 'PubDB')

        self.valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=True)

    def get_cam(self, images, beta=10, it=0.6):
        with torch.no_grad():
            cam_list = []
            _, _, h, w = images.shape

            for scale in self.scale_list:
                target_size = (round(h * abs(scale)), round(w * abs(scale)))
                scaled_images = F.interpolate(images, target_size, mode='bilinear', align_corners=False)
                H_, W_ = int(np.ceil(target_size[0] / 16.0) * 16), int(np.ceil(target_size[1] / 16.0) * 16)
                scaled_images = F.interpolate(scaled_images, (H_, W_), mode='bilinear', align_corners=False)

                if scale < 0:
                    scaled_images = torch.flip(scaled_images, dims=[3])

                logits, _, _ = self.C_model(scaled_images, pcm=beta, th=it)
                cam_list.append(logits)

        return cam_list

    def get_Q(self, images):
        _, _, h, w = images.shape
        q_list = []
        affmat_list = []

        for scale in self.scale_list:
            target_size = (round(h * abs(scale)), round(w * abs(scale)))
            H_, W_ = int(np.ceil(target_size[0] / 16.0) * 16), int(np.ceil(target_size[1] / 16.0) * 16)
            scaled_images = F.interpolate(images, (H_, W_), mode='bilinear', align_corners=False)

            if scale < 0:
                scaled_images = torch.flip(scaled_images, dims=[3])

            pred = self.Q_model(scaled_images)
            q_list.append(pred)
            affmat_list.append(None)

        return q_list, affmat_list

    def get_multiscale_cam(self, cam_list, q_list, affmat_list, refine_time=0):
        _, _, h, w = cam_list[-1].shape
        h *= 16
        w *= 16
        refine_cam_list = []

        for cam, q, affmat, scale in zip(cam_list, q_list, affmat_list, self.scale_list):
            if self.SSTB:
                for _ in range(refine_time):
                    cam = refine_with_affmat(cam, affmat)
                cam = upfeat(cam, q, 16, 16)

            cam = F.interpolate(cam, (int(h), int(w)), mode='bilinear', align_corners=False)
            if scale < 0:
                cam = torch.flip(cam, dims=[3])
            refine_cam_list.append(cam)

        return torch.sum(torch.stack(refine_cam_list), dim=0)

    def get_best_miou(self, clear=True):
        iou_list = []
        for parm, meter in zip(self.parms, self.meterlist):
            cur_iou, _, _, _, _ = meter.get(clear=clear, detail=True)
            iou_list.append((cur_iou, parm))
        iou_list.sort(key=lambda x: x[0], reverse=True)
        return iou_list

    def evaluate(self, C_model, Q_model=None, beta=10, ite=0.6):
        self.C_model = C_model
        self.Q_model = Q_model
        self.C_model.eval()

        if self.SSTB:
            self.Q_model.eval()

        with torch.no_grad():
            length = len(self.valid_loader)

            for step, (images, image_ids, tags, gt_masks) in enumerate(self.valid_loader):
                images = images.cuda()
                gt_masks = gt_masks.cuda()
                _, _, h, w = images.shape

                if self.SSTB:
                    q_list, affmats = self.get_Q(images)
                else:
                    q_list = [images for _ in range(len(self.scale_list))]
                    affmats = [None for _ in range(len(self.scale_list))]

                cams_list = self.get_cam(images, beta, ite)
                mask = tags.unsqueeze(2).unsqueeze(3).cuda()

                for refine_num in self.refine_list:
                    refine_cams = self.get_multiscale_cam(cams_list, q_list, affmats, refine_num)
                    cams = make_cam(refine_cams) * mask
                    cams = F.interpolate(cams, (int(h), int(w)), mode='bilinear', align_corners=False)

                    if self.save_np_path is not None:
                        cams_to_save = F.interpolate(cams, (int(h), int(w)), mode='bilinear', align_corners=False)
                        np.save(os.path.join(self.save_np_path, image_ids[0] + '.npy'), cams_to_save.cpu().numpy())

                        img_8 = convert_to_tf(images[0])
                        foreground_max = cams[0, 1:].max(0, True)[0]
                        cams[0, 0] = foreground_max

                        saveimg = None
                        first_mask = True
                        for class_idx in range(1, 3):
                            if tags[0][class_idx] == 1:
                                colored = torch.zeros(cams.shape)[0][0]
                                colored[:, :] = class_idx
                                colored = get_colored_mask(colored.cpu().numpy())
                                colored = cv2.cvtColor(colored, cv2.COLOR_RGB2HSV)
                                colored[:, :, 2] = (230 * cams[0][class_idx].cpu().numpy()).astype(np.uint8)
                                colored = cv2.cvtColor(colored, cv2.COLOR_HSV2BGR)

                                if first_mask:
                                    saveimg = colored.astype(np.float32)
                                    first_mask = False
                                else:
                                    saveimg += colored.astype(np.float32)

                        if saveimg is not None:
                            saveimg[saveimg > 255] = 255
                            saveimg = saveimg.astype(np.uint8)
                            cv2.imwrite(os.path.join(self.save_np_path, image_ids[0] + '_2.png'), saveimg)

                        cam_vis = generate_vis(
                            cams[0].cpu().numpy(),
                            None,
                            img_8,
                            func_label2color=VOClabel2colormap,
                            threshold=None,
                            norm=False,
                        )

                        for class_idx in range(3):
                            if tags[0][class_idx] == 1:
                                save_img = cam_vis[class_idx].transpose(1, 2, 0) * 255
                                save_img = cv2.cvtColor(save_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                                cv2.imwrite(os.path.join(self.save_np_path, image_ids[0] + '_' + str(class_idx) + '.png'), save_img)

                    if step in {100, 200, 600, 1450}:
                        print(self.get_best_miou(clear=False))

                    for th in self.th_list:
                        cams[:, 0] = th
                        predictions = torch.argmax(cams, dim=1)

                        for batch_index in range(images.size(0)):
                            pred_mask = get_numpy_from_tensor(predictions[batch_index])
                            gt_mask = get_numpy_from_tensor(gt_masks[batch_index])
                            gt_mask = cv2.resize(gt_mask, (pred_mask.shape[1], pred_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                            self.meterlist[self.parms.index((refine_num, th))].add(pred_mask, gt_mask)

                            if self.save_png_path is not None:
                                cur_save_path = os.path.join(self.save_png_path, str(th), str(refine_num))
                                os.makedirs(cur_save_path, exist_ok=True)
                                img_path = os.path.join(cur_save_path, image_ids[batch_index] + '.png')
                                save_colored_mask(pred_mask, img_path)

                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()

        self.C_model.train()

        if self.save_png_path is not None:
            savetxt_path = os.path.join(self.save_png_path, 'result.txt')
            with open(savetxt_path, 'wb') as f:
                for parm, meter in zip(self.parms, self.meterlist):
                    cur_iou = meter.get(clear=False)[-2]
                    f.write('{:>10.2f} {:>10.2f} {:>10.2f}\n'.format(cur_iou, parm[0], parm[1]).encode())

        return self.get_best_miou()


if __name__ == '__main__':
    args = get_params()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tagA = args.tagA
    if args.beta != 0:
        tagA = tagA + '_beta=%s' % args.beta

    time_string = time.strftime('%Y_%m_%d_%H_%M_%S')
    log_dir = create_directory(f'./experiments/logs/{args.tag}/{tagA}/')
    log_path = os.path.join(log_dir, f'{time_string}.txt')

    prediction_path = None
    if args.savepng or args.savenpy:
        prediction_tag = create_directory(f'./experiments/predictions/{args.tag}/{tagA}/')
        prediction_path = create_directory(prediction_tag + f'{time_string}/')

    log_func = lambda string='': log_print(string, log_path)
    log_func('[i] {}'.format(args.tag))
    log_func(str(args))

    model = CLSNet('resnet50', num_classes=3).cuda()
    model.train()
    model.load_state_dict(torch.load(args.Cmodel_path))

    if args.SSTB:
        q_model = fcnmodel.SpixelNet1l_bn().cuda()
        q_model.load_state_dict(torch.load(args.Qmodel_path))
        q_model.eval()
    else:
        q_model = None

    savepng_path = create_directory(prediction_path + 'pseudo/') if args.savepng and prediction_path is not None else None
    savenpy_path = create_directory(prediction_path + 'camnpy/') if args.savenpy and prediction_path is not None else None

    log_func(str(args.beta))

    evaluator = Evaluator(
        dataset=args.dataset,
        domain=args.domain,
        muti_scale=True,
        SSTB=args.SSTB,
        save_np_path=savenpy_path,
        savepng_path=savepng_path,
        refine_list=[0],
        th_list=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    )
    ret = evaluator.evaluate(model, q_model, args.beta, args.gama)

    log_func(str(ret))
    log_func('IMG_train')
    log_func(str(args.beta))
    log_func(str(args.gama))
