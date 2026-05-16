

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch.nn.init import kaiming_normal_, constant_
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from collections import OrderedDict

# from tools.ai.demo_utils import crf_inference
# from tools.general.Q_util import *
# from tools.ai.torch_utils import make_cam
# from core.models.model_util import conv

import tools
from tools.ai.demo_utils import crf_inference
from tools.ai.torch_utils import resize_for_tensors
from tools.general.Q_util import *
from tools.ai.torch_utils import make_cam
from core.models.model_util import conv


from .deeplab_utils import ASPP, Decoder, ASPP_V2
from .arch_resnet import resnet
from .arch_resnest import resnest
from .abc_modules import ABC_Model



#######################################################################
# Normalization
#######################################################################
class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, training=False, eps=self.eps)

def group_norm(features):
    return nn.GroupNorm(4, features)
#######################################################################

def conv_bn(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

def conv_dilation(batchNorm, in_planes, out_planes, kernel_size=3, stride=1,dilation=16):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation, bias=False,dilation=dilation,padding_mode='circular'),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True,dilation=dilation,padding_mode='circular'),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

def get_noliner(features):
            b, c, h, w = features.shape
            if(c==9):
                feat_pd = F.pad(features, (1, 1, 1, 1), mode='constant', value=0)
            elif(c==25):
                feat_pd = F.pad(features, (2, 2, 2, 2), mode='constant', value=0)

            diff_map_list=[]
            nn=int(math.sqrt(c))
            for i in range(nn):
                for j in range(nn):
                        diff_map_list.append(feat_pd[:,i*nn+j,i:i+h,j:j+w])
            ret = torch.stack(diff_map_list,dim=1)
            return ret


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1)
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1)
    )


class Backbone(nn.Module, ABC_Model):
    def __init__(self, model_name, num_classes=20, mode='fix', segmentation=False):
        super().__init__()

        self.mode = mode

        if self.mode == 'fix':
            self.norm_fn = FixedBatchNorm
        else:
            self.norm_fn = nn.BatchNorm2d

        if 'resnet' in model_name:

            if('moco' in model_name ):
                state_dict = torch.load("/media/ders/XS/SPCAM/models_ckpt/moco_r50_v2-e3b0c442.pth")['state_dict']
                model_name = model_name[:-5]

            elif('detco' in model_name ):
                state_dict = torch.load("/media/ders/XS/SPCAM/models_ckpt/detco_200ep.pth")
                model_name = model_name[:-6]
            elif('dino' in model_name ):
                state_dict = torch.load("/media/ders/XS/SPCAM/models_ckpt/dino_resnet50_pretrain.pth")
                model_name = model_name[:-5]

            elif('resnet101' in model_name):
                print("#################################################已经#￥333333")
                state_dict = torch.load("/media/ders/sdb1/hjw/SPCAM_GCMS/resnet101-5d3b4d8f.pth")
            elif ('resnet50' in model_name):
                state_dict = torch.load("/media/ders/sdb1/hjw/SPCAM_GCMS/resnet50-19c8e357.pth")
            else:
                print('resnet101' in model_name)

                state_dict = model_zoo.load_url(resnet.urls_dic[model_name])
            state_dict.pop('fc.weight')
            state_dict.pop('fc.bias')
            self.model = resnet.ResNet(resnet.Bottleneck, resnet.layers_dic[model_name], strides=(2, 2, 2, 1), batch_norm_fn=self.norm_fn)

            # self.initialize(self.model.modules())


            # for k, v in state_dict.items():
            #     name = k[15:]   # remove `vgg.`，即只取vgg.0.weights的后面几位
            #     if(name[:2]=="fc") or (name[:2]=="r."):
            #         continue
            #     new_state_dict[name] = v
            # state_dict=  new_state_dict
            #state_dict = torch.load("models_ckpt/dino_resnet50_pretrain.pth")

            self.model.load_state_dict(state_dict)
        else:
            if segmentation:
                dilation, dilated = 4, True
            else:
                dilation, dilated = 2, False

            self.model = eval("resnest." + model_name)(pretrained=True, dilated=dilated, dilation=dilation, norm_layer=self.norm_fn)

            del self.model.avgpool
            del self.model.fc

        self.stage1 = nn.Sequential(self.model.conv1,
                                    self.model.bn1,
                                    self.model.relu,
                                    self.model.maxpool)
        self.stage2 = nn.Sequential(self.model.layer1)
        self.stage3 = nn.Sequential(self.model.layer2)
        self.stage4 = nn.Sequential(self.model.layer3)
        self.stage5 = nn.Sequential(self.model.layer4)


class AffinityNet(Backbone):
    def __init__(self, model_name, path_index=None):
        super().__init__(model_name, None, 'fix')

        if '50' in model_name:
            fc_edge1_features = 64
        else:
            fc_edge1_features = 128

        self.fc_edge1 = nn.Sequential(
            nn.Conv2d(fc_edge1_features, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge2 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
        )
        self.fc_edge3 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge4 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge5 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.ReLU(inplace=True),
        )
        self.fc_edge6 = nn.Conv2d(160, 1, 1, bias=True)

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4, self.stage5])
        self.edge_layers = nn.ModuleList([self.fc_edge1, self.fc_edge2, self.fc_edge3, self.fc_edge4, self.fc_edge5, self.fc_edge6])

        if path_index is not None:
            self.path_index = path_index
            self.n_path_lengths = len(self.path_index.path_indices)
            for i, pi in enumerate(self.path_index.path_indices):
                self.register_buffer("path_indices_" + str(i), torch.from_numpy(pi))
    
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()

    def forward(self, x, with_affinity=False):
        x1 = self.stage1(x).detach()
        x2 = self.stage2(x1).detach()
        x3 = self.stage3(x2).detach()
        x4 = self.stage4(x3).detach()
        x5 = self.stage5(x4).detach()
        
        edge1 = self.fc_edge1(x1)
        edge2 = self.fc_edge2(x2)
        edge3 = self.fc_edge3(x3)[..., :edge2.size(2), :edge2.size(3)]
        edge4 = self.fc_edge4(x4)[..., :edge2.size(2), :edge2.size(3)]
        edge5 = self.fc_edge5(x5)[..., :edge2.size(2), :edge2.size(3)]

        edge = self.fc_edge6(torch.cat([edge1, edge2, edge3, edge4, edge5], dim=1))

        if with_affinity:
            return edge, self.to_affinity(torch.sigmoid(edge))
        else:
            return edge

    def get_edge(self, x, image_size=512, stride=4):
        feat_size = (x.size(2)-1)//stride+1, (x.size(3)-1)//stride+1

        x = F.pad(x, [0, image_size-x.size(3), 0, image_size-x.size(2)])
        edge_out = self.forward(x)
        edge_out = edge_out[..., :feat_size[0], :feat_size[1]]
        edge_out = torch.sigmoid(edge_out[0]/2 + edge_out[1].flip(-1)/2)
        
        return edge_out
    
    """
    aff = self.to_affinity(torch.sigmoid(edge_out))
    pos_aff_loss = (-1) * torch.log(aff + 1e-5)
    neg_aff_loss = (-1) * torch.log(1. + 1e-5 - aff)
    """
    def to_affinity(self, edge):
        aff_list = []
        edge = edge.view(edge.size(0), -1)
        
        for i in range(self.n_path_lengths):
            ind = self._buffers["path_indices_" + str(i)]
            ind_flat = ind.view(-1)
            dist = torch.index_select(edge, dim=-1, index=ind_flat)
            dist = dist.view(dist.size(0), ind.size(0), ind.size(1), ind.size(2))
            aff = torch.squeeze(1 - F.max_pool2d(dist, (dist.size(2), 1)), dim=2)
            aff_list.append(aff)
        aff_cat = torch.cat(aff_list, dim=1)
        return aff_cat


class CAM_Model(Backbone):
    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)

        self.num_classes =num_classes
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)

        self.ala2 = nn.Conv2d(2048, 2048, 1, bias=False)
        self.ala1 = nn.Conv2d(1024, 1024, 1, bias=False)

    def forward(self, inputs, pcm=0):
        x = self.stage1(inputs)
        x = self.stage2(x)
        x = self.stage3(x).detach()
        x4 = self.stage4(x)
        # ala1=self.ala1(F.adaptive_avg_pool2d( x4.detach(), 1))
        # ala1=torch.sigmoid(ala2)
        # x4=x4*ala1

        x5 = self.stage5(x4)
        # ala2=self.ala2(F.adaptive_avg_pool2d( x5.detach(), 1))
        # ala2=torch.sigmoid(ala2)
        # x5=x5*ala2
        logits = self.classifier(x5)
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        logits_min =(F.adaptive_avg_pool2d(self.classifier(x5), 1))

        # GAM模块
        if (pcm > 0):
            x4 = torch.cat([x4], dim=1)                 # train: x4:(16,1024,30,30)        infer: x4:(1,1024,9,16)
            b, c, h, w = x4.shape
            x4 = x4.view(b, c, -1)
            x4 = F.normalize(x4, dim=1)                 # train: x4:(16,1024,900)          infer: x4:(1,1024,144)
            aff_b = torch.bmm(x4.transpose(1, 2), x4)   # train: aff_b:(16,900,900)        infer: aff_b: x4:(1,144,144)
            aff = torch.clamp(aff_b, 0.01, 0.999)       # train: aff:(16,900,900)          infer: aff: x4:(1,144,144)
            # th=0.5
            aff[aff < 0.6] = 0
            # aff=F.relu(aff-th)
            # aff=F.relu()
            # aff[aff>th]=1
            #aff[aff>0.8]=0.2
            aff = aff/aff.sum(1, True)
            logits_flat = logits.view(b, self.num_classes, -1)#aff.max()  # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
            for i in range(pcm):
                logits_flat = torch.bmm(logits_flat, aff)   # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
            logits = logits_flat.view(b, self.num_classes, h, w)          # train: logits:(16,21,30,30)      infer: logits:(1,1024,9,16)

        return logits, logits_min, x4


class SP_CAM_Model3(Backbone):

    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)
        ch_q = 32
        self.outc = 9
        self.num_classes = num_classes

        self.get_qfeats = nn.Sequential(
                        conv(True, 9, ch_q,  4, stride=4),
                        conv(True, ch_q, ch_q * 4,  4, stride=4),
                        conv(False, ch_q * 4, ch_q * 4, 3, stride=1),
                        )

        self.x4_feats = nn.Sequential(
                        conv(True, 1024, 128, 1, stride=1),
                        )
        self.x5_feats = nn.Sequential(
                        conv(True, 2048, 128, 1, stride=1),
                        )

        self.get_tran_conv5 = nn.Sequential(
                conv(False, 128, 256, 3),  # conv(False, 128, 256, 3),
                conv(False, 256,  self.outc, 1),
                nn.Softmax(1)
            )
        self.get_tran_conv4 = nn.Sequential(
                conv(False, 128, 256, 3),  # conv(False, 128, 256, 3)
                conv(False, 256, self.outc, 1),
                nn.Softmax(1)
            )

        self.ala2 = nn.Sequential(nn.Conv2d(2048, 128, 1, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 2048, 1, bias=False),
                                  nn.Sigmoid())
        self.ala1 = nn.Sequential(nn.Conv2d(1024, 64, 1, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 1024, 1, bias=False),
                                  nn.Sigmoid())

        self.classifier = nn.Sequential(nn.Conv2d(2048, num_classes, 1, bias=False))



    def forward(self, inputs, probs, labels=None, pcm=0, th=0.5):
        q_feat = self.get_qfeats(probs)

        x1 = self.stage1(inputs)    # train: x1:(16,64,120,120)        infer: x1:(1,64,36,64) downsample*4
        x2 = self.stage2(x1)        # train: x2:(16,256,120,120)       infer: x2:(1,256,36,64) 不变
        x3 = self.stage3(x2)        # train: x3:(16,512,60,60)         infer: x3:(1,512,18,32) downsample*2

        x4 = self.stage4(x3)                                                # train: x4:(16,1024,30,30)        infer: x4:(1,1024,9,16) downsample*2
        
        x4_dp = self.get_tran_conv4(torch.cat([self.x4_feats(x4)], dim=1))  # train: x4_dp:(16,9,30,30)        infer: x4_dp:(1,9,9,16) 不变
        # x4_dp = self.get_tran_conv4(torch.cat([q_feat], dim=1))  # train: x4_dp:(16,9,30,30)        infer: x4_dp:(1,9,9,16) 不变
        # x4_dp = self.get_tran_conv4(torch.cat([self.x4_feats(x4.detach()), q_feat], dim=1))
        
        ala1 = self.ala1(F.adaptive_avg_pool2d(x4, 1))                      # train: ala1:(16,1024,1,1)        infer: ala1:(1,1024,1,1) 池化
        x4 = x4 * ala1                                                      # train: x4:(16,1024,30,30)        infer: x4:(1,1024,9,16)
        x4 = upfeat(x4, x4_dp, 1, 1)                                        # train: x4:(16,1024,30,30)        infer: x4:(1,1024,9,16)

        x5 = self.stage5(x4)                                                # train: x5:(16,2048,30,30)        infer: x5:(1,2048,9,16)
        
        x5_dp = self.get_tran_conv5(torch.cat([self.x5_feats(x5)], dim=1))  # train: x5_dp:(16,9,30,30)        infer: x5_dp:(1,9,9,16)
        # x5_dp = self.get_tran_conv5(torch.cat([q_feat], dim=1))  # train: x5_dp:(16,9,30,30)        infer: x5_dp:(1,9,9,16)
        # x5_dp = self.get_tran_conv5(torch.cat([self.x5_feats(x5.detach()), q_feat], dim=1))  # train: x5_dp:(16,9,30,30)        infer: x5_dp:(1,9,9,16)
        
        ala2 = self.ala2(F.adaptive_avg_pool2d(x5.detach(), 1))             # train: ala2:(16,2048,1,1)        infer: ala2:(1,2048,1,1) 池化
        x5 = x5 * ala2                                                      # train: x5:(16,2048,30,30)        infer: x5:(1,2048,9,16)
        # x5 = upfeat(x5, x5_dp, 1, 1)

        # 用来计算分类损失
        # logits_min = self.classifier(F.adaptive_avg_pool2d(x5, 1))  # train: logits_min:(16,21,1,1)    infer: logits_min:(16,21,1,1)
        # logits_min =self.classifier(F.adaptive_avg_pool2d((x5), 1))
        logits_min = F.adaptive_avg_pool2d(self.classifier(x5), 1)

        # 用来计算ipc loss/gc loss
        logits = self.classifier(x5)                                        # train: logits:(16,21,30,30)      infer: logits: (1,21,9,16)
        logits = upfeat(logits, x5_dp, 1, 1)                                # train: logits:(16,21,30,30)      infer: logits: (1,21,9,16)

        # GAM模块
        if (pcm > 0):
            x4 = torch.cat([x4], dim=1)                 # train: x4:(16,1024,30,30)        infer: x4:(1,1024,9,16)
            b, c, h, w = x4.shape
            x4 = x4.view(b, c, -1)
            x4 = F.normalize(x4, dim=1)                 # train: x4:(16,1024,900)          infer: x4:(1,1024,144)
            aff_b = torch.bmm(x4.transpose(1, 2), x4)   # train: aff_b:(16,900,900)        infer: aff_b: x4:(1,144,144)
            aff = torch.clamp(aff_b, 0.01, 0.999)       # train: aff:(16,900,900)          infer: aff: x4:(1,144,144)
            # th=0.5
            aff[aff < th] = 0
            # aff=F.relu(aff-th)
            # aff=F.relu()
            # aff[aff>th]=1
            #aff[aff>0.8]=0.2
            aff = aff/aff.sum(1, True)
            logits_flat = logits.view(b, self.num_classes, -1)#aff.max()  # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
            for i in range(pcm):
                logits_flat = torch.bmm(logits_flat, aff)   # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
            logits = logits_flat.view(b, self.num_classes, h, w)          # train: logits:(16,21,30,30)      infer: logits:(1,1024,9,16)

        return logits, logits_min, x4   # logtis用来计算ipc loss/gc loss 或 sal loss；logits_min用来计算多标签分类损失，所以用F.adaptive_avg_pool2d做了GAP的操作；


class MCGN_FULL(Backbone):
    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)

        self.num_classes =num_classes
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)

    def forward(self, inputs, pcm=0, th=0.6):
        x = self.stage1(inputs)
        x = self.stage2(x)
        x = self.stage3(x).detach()
        x4 = self.stage4(x)
        # ala1=self.ala1(F.adaptive_avg_pool2d( x4.detach(), 1))
        # ala1=torch.sigmoid(ala2)
        # x4=x4*ala1

        x5 = self.stage5(x4)
        # ala2=self.ala2(F.adaptive_avg_pool2d( x5.detach(), 1))
        # ala2=torch.sigmoid(ala2)
        # x5=x5*ala2
        logits = self.classifier(x5)  #这里得到的是num_class*h*w
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        logits_min =(F.adaptive_avg_pool2d(self.classifier(x5), 1))#这里得到的是num_class*1*1

        # GAM模块
        if (pcm > 0):
            x4 = torch.cat([x4], dim=1)                 # train: x4:(16,1024,30,30)        infer: x4:(1,1024,9,16)
            b, c, h, w = x4.shape
            x4 = x4.view(b, c, -1)
            x4 = F.normalize(x4, dim=1)                 # train: x4:(16,1024,900)          infer: x4:(1,1024,144)
            aff_b = torch.bmm(x4.transpose(1, 2), x4)   # train: aff_b:(16,900,900)        infer: aff_b: x4:(1,144,144)
            aff = torch.clamp(aff_b, 0.01, 0.999)       # train: aff:(16,900,900)          infer: aff: x4:(1,144,144)
            # th=0.5
            aff[aff < th] = 0
            # aff=F.relu(aff-th)
            # aff=F.relu()
            # aff[aff>th]=1
            #aff[aff>0.8]=0.2
            aff = aff/aff.sum(1, True)
            logits_flat = logits.view(b, self.num_classes, -1)#aff.max()  # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
            for i in range(pcm):
                logits_flat = torch.bmm(logits_flat, aff)   # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
            logits = logits_flat.view(b, self.num_classes, h, w)          # train: logits:(16,21,30,30)      infer: logits:(1,1024,9,16)

        return logits, logits_min, x4



class MCGN_FULL1(Backbone):

    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)
        ch_q = 32
        self.outc = 9
        self.num_classes = num_classes
        self.assign_ch = 9

        # SSTaskBranch
        self.x4_feats = nn.Sequential(conv(True, 1024, 128, 1, stride=1))
        self.x5_feats = nn.Sequential(conv(True, 2048, 128, 1, stride=1))

        self.get_tran_conv5 = nn.Sequential(
                conv(False, 128, 256, 3),  # conv(False, 128, 256, 3),
                conv(False, 256,  self.outc, 1),
                nn.Softmax(1))
        self.get_tran_conv4 = nn.Sequential(
                conv(False, 128, 256, 3),  # conv(False, 128, 256, 3)
                conv(False, 256, self.outc, 1),
                nn.Softmax(1))
        self.ala1 = nn.Sequential(nn.Conv2d(1024, 64, 1, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(64, 1024, 1, bias=False),
                                  nn.Sigmoid())
        self.ala2 = nn.Sequential(nn.Conv2d(2048, 128, 1, bias=False),
                                  nn.ReLU(),
                                  nn.Conv2d(128, 2048, 1, bias=False),
                                  nn.Sigmoid())

        # SSTaskBranch_deconv
        self.x5dp_up16 = nn.Sequential(nn.ConvTranspose2d(self.assign_ch, self.assign_ch, kernel_size=4, stride=4, padding=0, bias=True),
                                nn.LeakyReLU(0.1),
                                nn.ConvTranspose2d(self.assign_ch, self.assign_ch, kernel_size=4, stride=4, padding=0, bias=True),
                                nn.LeakyReLU(0.1),
                                nn.Softmax(1))
        
        # SAPTaskBranch
        self.q = nn.Sequential(nn.Conv2d(1024, 1024, 1, bias=False),
                                nn.LeakyReLU(0.1))
        self.k = nn.Sequential(nn.Conv2d(1024, 1024, 1, bias=False),
                                nn.LeakyReLU(0.1))
        
        # ICTaskBranch
        self.classifier = nn.Sequential(nn.Conv2d(2048, num_classes, 1, bias=False))

    def get_sstb_refine_x4(self, x4):
        x4_dp = self.get_tran_conv4(torch.cat([self.x4_feats(x4)], dim=1))  # train: x5_dp:(16,9,30,30)        infer: x5_dp:(1,9,9,16)
        ala1 = self.ala1(F.adaptive_avg_pool2d(x4, 1))             # train: ala2:(16,2048,1,1)        infer: ala2:(1,2048,1,1) 池化
        x4 = x4 * ala1                                                      # train: x5:(16,2048,30,30)        infer: x5:(1,2048,9,16)
        x4 = upfeat(x4, x4_dp, 1, 1)
        x4_fp = self.x5dp_up16(x4_dp)

        return x4, x4_fp

    def get_sstb_refine_x5(self, x5):
        x5_dp = self.get_tran_conv5(torch.cat([self.x5_feats(x5)], dim=1))  # train: x5_dp:(16,9,30,30)        infer: x5_dp:(1,9,9,16)
        ala2 = self.ala2(F.adaptive_avg_pool2d(x5.detach(), 1))             # train: ala2:(16,2048,1,1)        infer: ala2:(1,2048,1,1) 池化
        x5 = x5 * ala2                                                      # train: x5:(16,2048,30,30)        infer: x5:(1,2048,9,16)
        # x5 = upfeat(x5, x5_dp, 1, 1)
        x5_fp = self.x5dp_up16(x5_dp)

        return x5, x5_fp, x5_dp
    
    def get_saptb_refine_logits(self, logits, x4, pcm=0, th=0.6):
        if (pcm > 0):
            x4 = torch.cat([x4], dim=1)                 # train: x4:(16,1024,30,30)        infer: x4:(1,1024,9,16)
            b, c, h, w = x4.shape
            x4 = x4.view(b, c, -1)
            x4 = F.normalize(x4, dim=1)                 # train: x4:(16,1024,900)          infer: x4:(1,1024,144)
            aff_b = torch.bmm(x4.transpose(1, 2), x4)   # train: aff_b:(16,900,900)        infer: aff_b: x4:(1,144,144)
            aff = torch.clamp(aff_b, 0.01, 0.999)       # train: aff:(16,900,900)          infer: aff: x4:(1,144,144)
            aff[aff < th] = 0
            aff = aff/aff.sum(1, True)
            logits_flat = logits.view(b, self.num_classes, -1)#aff.max()  # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
            for i in range(pcm):
                logits_flat = torch.bmm(logits_flat, aff)   # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
            logits = logits_flat.view(b, self.num_classes, h, w)          # train: logits:(16,21,30,30)      infer: logits:(1,1024,9,16)
        else:
            aff = None
            logits = logits

        return logits, aff
    
    
    def forward(self, inputs, pcm=0, th=0.6):

        x1 = self.stage1(inputs)    # train: x1:(16,64,120,120)        infer: x1:(1,64,36,64) downsample*4
        x2 = self.stage2(x1)        # train: x2:(16,256,120,120)       infer: x2:(1,256,36,64) 不变
        x3 = self.stage3(x2)        # train: x3:(16,512,60,60)         infer: x3:(1,512,18,32) downsample*2

        x4 = self.stage4(x3)                                                # train: x4:(16,1024,30,30)        infer: x4:(1,1024,9,16) downsample*2
        x4, _ = self.get_sstb_refine_x4(x4)    # use dcm module
        
        x5 = self.stage5(x4)                                                # train: x5:(16,2048,30,30)        infer: x5:(1,2048,9,16)
        x5, PRC, x5_dp = self.get_sstb_refine_x5(x5)    # use dcm module
        
        # 用来计算分类损失
        logits_min = F.adaptive_avg_pool2d(self.classifier(x5), 1)

        # 用来计算ipc loss/gc loss
        logits = self.classifier(x5)                                        # train: logits:(16,21,30,30)      infer: logits: (1,21,9,16)
        logits = upfeat(logits, x5_dp, 1, 1)                                # train: logits:(16,21,30,30)      infer: logits: (1,21,9,16)

        logits, PPC = self.get_saptb_refine_logits(logits, x4, pcm=pcm, th=th)
        
        return logits, logits_min, x4   # logtis用来计算ipc loss/gc loss 或 sal loss；logits_min用来计算多标签分类损失，所以用F.adaptive_avg_pool2d做了GAP的操作；


class MCGN_ABLA(Backbone):

    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)
        self.assign_ch = 9
        self.num_classes = num_classes

        # SSTaskBranch
        self.x5_feats = nn.Sequential(conv(True, 2048, 128, 1, stride=1))
        self.get_tran_conv5 = nn.Sequential(
                                conv(False, 128, 256, 3),  # conv(False, 128, 256, 3),
                                conv(False, 256,  self.assign_ch, 1),
                                nn.Softmax(1))
        self.ala2 = nn.Sequential(nn.Conv2d(2048, 128, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(128, 2048, 1, bias=False),
                                nn.Sigmoid())
        
        # SSTaskBranch_deconv
        self.x5dp_up16 = nn.Sequential(nn.ConvTranspose2d(self.assign_ch, self.assign_ch, kernel_size=4, stride=4, padding=0, bias=True),
                                nn.LeakyReLU(0.1),
                                nn.ConvTranspose2d(self.assign_ch, self.assign_ch, kernel_size=4, stride=4, padding=0, bias=True),
                                nn.LeakyReLU(0.1),
                                nn.Softmax(1))
        
        # SAPTaskBranch
        self.q = nn.Sequential(nn.Conv2d(1024, 1024, 1, bias=False),
                                nn.LeakyReLU(0.1))
        self.k = nn.Sequential(nn.Conv2d(1024, 1024, 1, bias=False),
                                nn.LeakyReLU(0.1))
        
        # ICTaskBranch
        self.classifier = nn.Sequential(nn.Conv2d(2048, num_classes, 1, bias=False))


    def get_sstb_refine_x5(self, x5):
        x5_dp = self.get_tran_conv5(torch.cat([self.x5_feats(x5)], dim=1))  # train: x5_dp:(16,9,30,30)        infer: x5_dp:(1,9,9,16)
        ala2 = self.ala2(F.adaptive_avg_pool2d(x5.detach(), 1))             # train: ala2:(16,2048,1,1)        infer: ala2:(1,2048,1,1) 池化
        x5 = x5 * ala2                                                      # train: x5:(16,2048,30,30)        infer: x5:(1,2048,9,16)
        x5 = upfeat(x5, x5_dp, 1, 1)
        x5_fp = self.x5dp_up16(x5_dp)

        return x5, x5_fp


    def get_saptb_refine_logits(self, logits, x4, pcm=0, th=0.6, get_aff_way='empform'):

        if get_aff_way == 'learn':
            if (pcm > 0):
                query = self.q(x4)
                key = self.k(x4)
                b, c, h, w = query.shape
                query = query.view(b, c, -1)
                query = F.normalize(query, dim=1) 

                key = key.view(b, c, -1)
                key = F.normalize(key, dim=1) 

                aff_b = torch.bmm(query.transpose(1, 2), key)   # train: aff_b:(16,900,900)        infer: aff_b: x4:(1,144,144)
                aff = torch.clamp(aff_b, 0.01, 0.999)       # train: aff:(16,900,900)          infer: aff: x4:(1,144,144)
                aff[aff < th] = 0
                aff = aff/aff.sum(1, True)
                logits_flat = logits.view(b, self.num_classes, -1)#aff.max()  # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
                for i in range(pcm):
                    logits_flat = torch.bmm(logits_flat, aff)   # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
                logits = logits_flat.view(b, self.num_classes, h, w)          # train: logits:(16,21,30,30)      infer: logits:(1,1024,9,16)
            else:
                aff = None
                logits = logits

        if get_aff_way == 'empform':
            if (pcm > 0):
                x4 = torch.cat([x4], dim=1)                 # train: x4:(16,1024,30,30)        infer: x4:(1,1024,9,16)
                b, c, h, w = x4.shape
                x4 = x4.view(b, c, -1)
                x4 = F.normalize(x4, dim=1)                 # train: x4:(16,1024,900)          infer: x4:(1,1024,144)
                aff_b = torch.bmm(x4.transpose(1, 2), x4)   # train: aff_b:(16,900,900)        infer: aff_b: x4:(1,144,144)
                aff = torch.clamp(aff_b, 0.01, 0.999)       # train: aff:(16,900,900)          infer: aff: x4:(1,144,144)
                aff[aff < th] = 0
                aff = aff/aff.sum(1, True)
                logits_flat = logits.view(b, self.num_classes, -1)#aff.max()  # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
                for i in range(pcm):
                    logits_flat = torch.bmm(logits_flat, aff)   # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
                logits = logits_flat.view(b, self.num_classes, h, w)          # train: logits:(16,21,30,30)      infer: logits:(1,1024,9,16)
            else:
                aff = None
                logits = logits

        return logits, aff

    def forward(self, inputs, mode='train', pcm=0, th=0.6):

        x1 = self.stage1(inputs)    # train: x1:(16,64,120,120)        infer: x1:(1,64,36,64) downsample*4
        x2 = self.stage2(x1)        # train: x2:(16,256,120,120)       infer: x2:(1,256,36,64) 不变
        x3 = self.stage3(x2)        # train: x3:(16,512,60,60)         infer: x3:(1,512,18,32) downsample*2    
        x4 = self.stage4(x3)                                                # train: x4:(16,1024,30,30)        infer: x4:(1,1024,9,16) downsample*2
    
        x5 = self.stage5(x4)                                                # train: x5:(16,2048,30,30)        infer: x5:(1,2048,9,16)
        # x5_, PRC = self.get_sstb_refine_x5(x5)     # no use dcm module(pure CAM)
        x5, PRC = self.get_sstb_refine_x5(x5)    # use dcm module

        # 用来计算分类损失
        # logits_min =self.classifier(F.adaptive_avg_pool2d((x5), 1))
        logits_min = F.adaptive_avg_pool2d(self.classifier(x5), 1)

        # 用来计算ipc loss/gc loss
        logits = self.classifier(x5)                                        # train: logits:(16,21,30,30)      infer: logits: (1,21,9,16)
       
        logits, PPC = self.get_saptb_refine_logits(logits, x4, pcm=pcm, th=th)
    
        return logits, logits_min, PRC, PPC   # logtis用来计算ipc loss/gc loss 或 sal loss；logits_min用来计算多标签分类损失，所以用F.adaptive_avg_pool2d做了GAP的操作；


class MCGN_ABLA_noSS(Backbone):

    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)
        ch_q = 32
        self.outc = 9
        self.num_classes = num_classes
        self.assign_ch = 9
        
        # SAPTaskBranch
        self.q = nn.Sequential(nn.Conv2d(1024, 1024, 1, bias=False),
                                nn.LeakyReLU(0.1))
        self.k = nn.Sequential(nn.Conv2d(1024, 1024, 1, bias=False),
                                nn.LeakyReLU(0.1))
        
        # ICTaskBranch
        self.classifier = nn.Sequential(nn.Conv2d(2048, num_classes, 1, bias=False))

    def get_saptb_refine_logits(self, logits, x4, pcm=0, th=0.6):
        if (pcm > 0):
            x4 = torch.cat([x4], dim=1)                 # train: x4:(16,1024,30,30)        infer: x4:(1,1024,9,16)
            b, c, h, w = x4.shape
            x4 = x4.view(b, c, -1)
            x4 = F.normalize(x4, dim=1)                 # train: x4:(16,1024,900)          infer: x4:(1,1024,144)
            aff_b = torch.bmm(x4.transpose(1, 2), x4)   # train: aff_b:(16,900,900)        infer: aff_b: x4:(1,144,144)
            aff = torch.clamp(aff_b, 0.01, 0.999)       # train: aff:(16,900,900)          infer: aff: x4:(1,144,144)
            aff[aff < th] = 0
            aff = aff/aff.sum(1, True)
            logits_flat = logits.view(b, self.num_classes, -1)#aff.max()  # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
            for i in range(pcm):
                logits_flat = torch.bmm(logits_flat, aff)   # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
            logits = logits_flat.view(b, self.num_classes, h, w)          # train: logits:(16,21,30,30)      infer: logits:(1,1024,9,16)
        else:
            aff = None
            logits = logits

        return logits, aff
    
    
    def forward(self, inputs, labels=None, pcm=0, th=0.6):

        x1 = self.stage1(inputs)    # train: x1:(16,64,120,120)        infer: x1:(1,64,36,64) downsample*4
        x2 = self.stage2(x1)        # train: x2:(16,256,120,120)       infer: x2:(1,256,36,64) 不变
        x3 = self.stage3(x2)        # train: x3:(16,512,60,60)         infer: x3:(1,512,18,32) downsample*2

        x4 = self.stage4(x3)                                                # train: x4:(16,1024,30,30)        infer: x4:(1,1024,9,16) downsample*2
        # x4, _ = self.get_sstb_refine_x4(x4)    # use dcm module
        
        x5 = self.stage5(x4)                                                # train: x5:(16,2048,30,30)        infer: x5:(1,2048,9,16)
        # x5, PRC, x5_dp = self.get_sstb_refine_x5(x5)    # use dcm module
        
        # 用来计算分类损失
        logits_min = F.adaptive_avg_pool2d(self.classifier(x5), 1)

        # 用来计算ipc loss/gc loss
        logits = self.classifier(x5)                                        # train: logits:(16,21,30,30)      infer: logits: (1,21,9,16)
        # logits = upfeat(logits, x5_dp, 1, 1)                                # train: logits:(16,21,30,30)      infer: logits: (1,21,9,16)

        logits, PPC = self.get_saptb_refine_logits(logits, x4, pcm=pcm, th=th)
        
        return logits, logits_min, x4   # logtis用来计算ipc loss/gc loss 或 sal loss；logits_min用来计算多标签分类损失，所以用F.adaptive_avg_pool2d做了GAP的操作；



class CLSNet(Backbone):
    def __init__(self, model_name, num_classes=21):
        super().__init__(model_name, num_classes, mode='fix', segmentation=False)

        self.num_classes =num_classes
        self.classifier = nn.Conv2d(2048, num_classes, 1, bias=False)

    def forward(self, inputs, pcm=0, th=0.6):
        x = self.stage1(inputs)
        x = self.stage2(x)
        x = self.stage3(x).detach()
        x4 = self.stage4(x)
        # ala1=self.ala1(F.adaptive_avg_pool2d( x4.detach(), 1))
        # ala1=torch.sigmoid(ala2)
        # x4=x4*ala1

        x5 = self.stage5(x4)
        # ala2=self.ala2(F.adaptive_avg_pool2d( x5.detach(), 1))
        # ala2=torch.sigmoid(ala2)
        # x5=x5*ala2
        logits = self.classifier(x5)  #这里得到的是num_class*h*w
        # logits = resize_for_tensors(logits, inputs.size()[2:], align_corners=False)
        logits_min =(F.adaptive_avg_pool2d(self.classifier(x5), 1))#这里得到的是num_class*1*1

        # GAM模块
        if (pcm > 0):
            x4 = torch.cat([x4], dim=1)                 # train: x4:(16,1024,30,30)        infer: x4:(1,1024,9,16)
            b, c, h, w = x4.shape
            x4 = x4.view(b, c, -1)
            x4 = F.normalize(x4, dim=1)                 # train: x4:(16,1024,900)          infer: x4:(1,1024,144)
            aff_b = torch.bmm(x4.transpose(1, 2), x4)   # train: aff_b:(16,900,900)        infer: aff_b: x4:(1,144,144)
            aff = torch.clamp(aff_b, 0.01, 0.999)       # train: aff:(16,900,900)          infer: aff: x4:(1,144,144)
            # th=0.5
            aff[aff < th] = 0
            # aff=F.relu(aff-th)
            # aff=F.relu()
            # aff[aff>th]=1
            #aff[aff>0.8]=0.2
            aff = aff/aff.sum(1, True)
            logits_flat = logits.view(b, self.num_classes, -1)#aff.max()  # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
            for i in range(pcm):
                logits_flat = torch.bmm(logits_flat, aff)   # train: logits_flat:(16,21,900)   infer: logits_flat:(1,21,144)
            logits = logits_flat.view(b, self.num_classes, h, w)          # train: logits:(16,21,30,30)      infer: logits:(1,1024,9,16)

        return logits, logits_min, x4






class DeepLabv3_Plus(Backbone):
    def __init__(self, model_name, num_classes=21, mode='fix', use_group_norm=False):
        super().__init__(model_name, num_classes, mode, segmentation=False)
        
        if use_group_norm:
            norm_fn_for_extra_modules = group_norm
        else:
            norm_fn_for_extra_modules = self.norm_fn
        
        self.aspp = ASPP(output_stride=16, norm_fn=norm_fn_for_extra_modules)
        self.decoder = Decoder(num_classes, 256, norm_fn_for_extra_modules)
        
    def forward(self, x, with_cam=False):
        inputs = x

        x = self.stage1(x)
        x = self.stage2(x)
        x_low_level = x
        
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        x = self.aspp(x)
        x = self.decoder(x, x_low_level)
        x = resize_for_tensors(x, inputs.size()[2:], align_corners=True)

        return x


class DeepLabv2(Backbone):
    def __init__(self, model_name, num_classes=21, mode='fix', use_group_norm=False):
        super().__init__(model_name, num_classes, mode, segmentation=False)
        
        if use_group_norm:
            norm_fn_for_extra_modules = group_norm
        else:
            norm_fn_for_extra_modules = self.norm_fn
        
        self.aspp = ASPP_V2(output_stride=16, norm_fn=norm_fn_for_extra_modules)
        self.classifier = nn.Conv2d(256, num_classes, 1, bias=False)
        
    def forward(self, x, with_cam=False):
        inputs = x

        x = self.stage1(x)
        x = self.stage2(x)        
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        
        x = self.aspp(x)   # 16,256,32,32
        # x = self.decoder(x, x_low_level)
        x = self.classifier(x)   # 16,3,32,32
        x = resize_for_tensors(x, inputs.size()[2:], align_corners=True)

        return x
    


       