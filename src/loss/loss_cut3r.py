import torch
import torch.nn as nn
from copy import copy, deepcopy

from .loss_dust3r import BaseCriterion, Criterion, MultiLoss, Sum, ConfLoss
from .loss_point import get_pred_pts3d
from .loss_dust3r import Regr3D as Regr3D_dust3r
from ..dust3r.utils.geometry import (geotrf, inv, normalize_pointcloud)
from ..dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


# ! from CUT3R
class RGBLoss(Criterion, MultiLoss):
    def __init__(self, criterion):
        super().__init__(criterion)
        self.ssim = SSIM()

    def img_loss(self, a, b):
        return self.criterion(a, b)

    def compute_loss(self, gts, preds, **kw):
        gt_rgbs = [gt["img"].permute(0, 2, 3, 1) for gt in gts]
        pred_rgbs = [pred["rgb"] for pred in preds]
        ls = [
            self.img_loss(pred_rgb, gt_rgb)
            for pred_rgb, gt_rgb in zip(pred_rgbs, gt_rgbs)
        ]
        details = {}
        self_name = type(self).__name__
        for i, l in enumerate(ls):
            details[self_name + f"_rgb/{i+1}"] = float(l)
            details[f"pred_rgb_{i+1}"] = pred_rgbs[i]
        rgb_loss = sum(ls) / len(ls)
        return rgb_loss, details