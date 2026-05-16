


from weakref import ref
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.Q_util import *
from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from torch.nn.modules.loss import _Loss



class SP_CAM_Loss2(_Loss):
  def __init__(self, args, size_average=None, reduce=None, reduction='mean'):
    super(SP_CAM_Loss2, self).__init__(size_average, reduce, reduction)
    self.args = args
    self.fg_c_num = 20 if args.dataset == 'voc12' else 80
    self.class_loss_fn = nn.CrossEntropyLoss().cuda()

  def forward(self, fg_cam, sailencys):#sailencys.max()

        """

        fg_cam：超像素cam
        saliencys：超像素Q下采样的RGB图，监督信息

        """

        b, c, h, w = fg_cam.size()                  # fg_cam:(144,20,10,10)
        imgmin_mask = sailencys.sum(1, True) != 0   # sailencys:(144,3,10,10);imgmin_mask:(144,1,10,10);bool; sailencys.sum(1, True):1是sum第一个维度，True是sum后保持维度不变; != 是将sailencys.sum(1, True) 的结果变成布尔量
        sailencys = F.interpolate(sailencys.float(), size=(h, w))   # sailencys：（144,3,10,10)

        bg = 1-torch.max(fg_cam, dim=1, keepdim=True)[0] ** 1       # bg:(144,1,10,10)

        nnn = torch.max((1 - bg.detach() * imgmin_mask).view(b, 1, -1), dim=2)[0] > self.args.ig_th   # nnn:(144,1);bool; self.args.ig_th=0.1
        nnn2 = torch.max((bg.detach() * imgmin_mask).view(b, 1, -1), dim=2)[0] > self.args.ig_th      # nnn2:(144,1);bool
        nnn = nnn * nnn2        # nnn:(144,1);bool
        if (nnn.sum() == 0):
          nnn = torch.ones(nnn.shape).cuda()
        imgmin_mask = nnn.view(b, 1, 1, 1) * imgmin_mask    # imgmin_mask:(144,1,10,10);bool;非零，前景、背景激活值均大于self.args.ig_th=0.1

        probs = torch.cat([bg, fg_cam], dim=1)              # probs:(144,21,10,10)
        probs1 = probs * imgmin_mask                        # probs1:(144,21,10,10)

        origin_f = F.normalize(sailencys.detach(), dim=1)   # origin_f:(144,3,10,10)
        origin_f = origin_f * imgmin_mask                   # origin_f:(144,3,10,10)

        f_min = pool_feat_2(probs1, origin_f)               # f_min(144,3,21)
        up_f = up_feat_2(probs1, f_min)                     # up_f(144,3,10,10)

        sal_loss = F.mse_loss(up_f, origin_f, reduce=False)           # sal_loss:(144,3,10,10)
        sal_loss = (sal_loss * imgmin_mask).sum() / (torch.sum(imgmin_mask) + 1e-3)    # 标量

        return sal_loss

