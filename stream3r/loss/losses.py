# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Implementation of DUSt3R training losses
# --------------------------------------------------------
from copy import copy, deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from stream3r.dust3r.utils.geometry import (
    geotrf,
    inv,
)
from stream3r.loss.utils import camera_loss, point_loss, depth_loss



class LLoss(nn.Module):
    """L-norm loss"""

    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, a, b):
        assert (
            a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3
        ), f"Bad shape = {a.shape}"
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss(LLoss):
    """Euclidean distance between 3d points"""

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class Criterion(nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, LLoss), (
            f"{criterion} is not a proper criterion!" + bb()
        )
        self.criterion = copy(criterion)

    def get_name(self):
        return f"{type(self).__name__}({self.criterion})"

    def with_reduction(self, mode):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = "none"  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss(nn.Module):
    """Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res

    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f"{self._alpha:g}*{name}"
        if self._loss2:
            name = f"{name} + {self._loss2}"
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class CausalLoss(MultiLoss):
    def __init__(self, gradient_loss="grad", is_metric=True):
        super().__init__()

        self.gradient_loss = gradient_loss
        self.is_metric = is_metric

    def get_name(self):
        return f"CausalLoss"

    def get_pts3d_from_views(self, gt_views, dist_clip=None, local=False, key_word="pts3d"):
        """Get point clouds and valid masks for multiple views."""
        gt_pts_list = []
        valid_mask_list = []

        if key_word == "pts3d":
            mask_key_word = "valid_mask"
        elif key_word == "track":
            mask_key_word = "track_valid_mask"
        else:
            raise ValueError(f"Invalid key_word: {key_word}")

        if not local:  # compute the inverse transformation for the anchor view (first view)
            inv_matrix_anchor = inv(gt_views[0]["camera_pose"].float())

        for gt_view in gt_views:
            if local:
                # Rotate GT points to align with the local camera origin for supervision
                inv_matrix_local = inv(gt_view["camera_pose"].float())
                gt_pts = geotrf(inv_matrix_local, gt_view[key_word])  # Transform GT points to local view's origin
            else:
                # Use the anchor view (first view) transformation for global loss
                gt_pts = geotrf(inv_matrix_anchor, gt_view[key_word])  # Transform GT points to anchor view

            valid_gt = gt_view[mask_key_word].clone()

            if dist_clip is not None:
                dis = gt_pts.norm(dim=-1)
                valid_gt &= dis <= dist_clip

            gt_pts_list.append(gt_pts)
            valid_mask_list.append(valid_gt)

        gt_pts = torch.stack(gt_pts_list, dim=1)
        valid_masks = torch.stack(valid_mask_list, dim=1)

        return gt_pts, valid_masks
    
    def get_depth_from_views(self, gt_views, dist_clip=None):
        gt_pts_list = []
        valid_mask_list = []

        mask_key_word = "valid_mask"

        for gt_view in gt_views:
            gt_pts =  gt_view["depthmap"]
            valid_gt = gt_view[mask_key_word].clone()

            if dist_clip is not None:
                dis = gt_pts.norm(dim=-1)
                valid_gt &= dis <= dist_clip

            gt_pts_list.append(gt_pts)
            valid_mask_list.append(valid_gt)

        gt_pts = torch.stack(gt_pts_list, dim=1)
        valid_masks = torch.stack(valid_mask_list, dim=1)

        return gt_pts, valid_masks

    def get_camera_from_views(self, gt_views):
        gt_extrinsic_list = []
        gt_intrinsic_list = []

        image_size_hw = gt_views[0]["img"].shape[-2:]
        for gt_view in gt_views:
            gt_extrinsic_list.append(gt_view["camera_pose"])
            gt_intrinsic_list.append(gt_view["camera_intrinsics"])

        gt_extrinsics = torch.stack(gt_extrinsic_list, dim=1)
        gt_intrinsics = torch.stack(gt_intrinsic_list, dim=1)

        return gt_extrinsics, gt_intrinsics, image_size_hw

    def compute_loss(self, gts, preds, **kw):
        details = {}
        self_name = type(self).__name__

        gt_pts3d_global, valid_mask_global = self.get_pts3d_from_views(gts, key_word="pts3d", **kw) # B, N, H, W, C
        gt_depth, valid_mask_depth = self.get_depth_from_views(gts, **kw) # B, N, H, W, C
        gt_extrinsics, gt_intrinsics, image_size_hw = self.get_camera_from_views(gts)

        pred_pts3d_global, pred_conf_global = preds["world_points"], preds["world_points_conf"]
        pred_depth, pred_depth_conf = preds["depth"], preds["depth_conf"]

        # loss for pts3d global
        loss_pts3d_global = point_loss(pred_pts3d_global, pred_conf_global, gt_pts3d_global, valid_mask_global, gradient_loss=self.gradient_loss, temporal_matching_loss=False, all_mean=True, valid_range=0.98, ormalize_pred=True, normalize_gt=True, normalize_using_first_view=False)

        # loss for depth
        loss_depth = depth_loss(pred_depth, pred_depth_conf, gt_depth, valid_mask_depth, gradient_loss=self.gradient_loss, temporal_matching_loss=False, all_mean=True, valid_range=0.98, normalize_pred=True, normalize_gt=True, normalize_using_first_view=False)
        gt_pts3d_scale = loss_depth[f"gt_pts3d_scale"]
        pred_pts3d_scale = loss_depth[f"pred_pts3d_scale"]

        # loss for camera
        pred_pose_enc_list = preds["pose_enc_list"]
        loss_camera = camera_loss(pred_pose_enc_list, gt_extrinsics, gt_intrinsics, image_size_hw, loss_type="l1", gt_pts3d_scale=gt_pts3d_scale, pred_pts3d_scale=pred_pts3d_scale, pose_encoding_type="relT_quaR_FoV")

        # total loss
        pts3d_loss = loss_pts3d_global["loss_conf"] + loss_pts3d_global["loss_grad"] + loss_depth["loss_conf"] + loss_depth["loss_grad"]
        total_loss = pts3d_loss + loss_camera["loss_camera"]

        # logs
        details[self_name + "_pts3d_loss" + "/00"] = float(pts3d_loss.detach())
        details[self_name + "_pts3d_loss_global" + "_conf" + "/00"] = float(loss_pts3d_global["loss_conf"].detach())
        details[self_name + "_pts3d_loss_global" + "_grad" + "/00"] = float(loss_pts3d_global["loss_grad"].detach())
        details[self_name + "_depth_loss" + "_conf" + "/00"] = float(loss_depth["loss_conf"].detach())
        details[self_name + "_depth_loss" + "_grad" + "/00"] = float(loss_depth["loss_grad"].detach())

        details[self_name + "_camera_loss" + "_loss_camera" + "/00"] = float(loss_camera["loss_camera"].detach())
        details[self_name + "_camera_loss" + "_loss_T" + "/00"] = float(loss_camera["loss_T"].detach())
        details[self_name + "_camera_loss" + "_loss_R" + "/00"] = float(loss_camera["loss_R"].detach())
        details[self_name + "_camera_loss" + "_loss_fl" + "/00"] = float(loss_camera["loss_fl"].detach())

        return total_loss, details