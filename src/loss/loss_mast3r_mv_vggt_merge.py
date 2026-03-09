# --------------------------------------------------------
# (Re)Implementation of Must3R training losses
# --------------------------------------------------------
import torch
import einops
import torch.nn as nn
import itertools
import numpy as np
from pdb import set_trace as st
from sklearn.metrics import average_precision_score

import trimesh
import os
# from dust3r.losses import BaseCriterion, Criterion, MultiLoss, Sum, ConfLoss
from .loss_dust3r import BaseCriterion, Criterion, MultiLoss, Sum, ConfLoss
from .loss_point import get_pred_pts3d
from .loss_dust3r import Regr3D as Regr3D_dust3r
from ..dust3r.utils.geometry import (geotrf, inv, normalize_pointcloud_group)
from ..dust3r.utils.geometry import get_joint_pointcloud_depth, get_joint_pointcloud_center_scale
from ..dust3r.utils.camera import (
    pose_encoding_to_camera,
    camera_to_pose_encoding,  # predicting [E]
    relative_pose_absT_quatR,
)

from functools import partial
from src.model.vggt.utils.pose_enc import extri_intri_to_pose_encoding  # predicting [E|K]

from .loss_mast3r import Regr3D

from torch.nn.functional import huber_loss
from .losses_vggt import camera_loss, point_loss, normalize_pointcloud


def save_pts3d(pts3d, output_path, output_suffix, rgb=None, v=False):
    assert isinstance(pts3d, torch.Tensor)
    assert pts3d.shape[0] == 1
    pts3d = pts3d.reshape(-1, 3).detach().cpu().numpy()

    pcd = trimesh.Trimesh(vertices=pts3d, vertex_colors=rgb)

    pcd_output_path = os.path.join(output_path, output_suffix)
    pcd.export(pcd_output_path, 'obj')
    if v:
        print('pcd saved to ', pcd_output_path)


class DepthScaleShiftInvLoss(BaseCriterion):
    """scale and shift invariant loss"""

    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, pred, gt, mask):
        assert pred.shape == gt.shape and pred.ndim == 3, f"Bad shape = {pred.shape}"
        dist = self.distance(pred, gt, mask)
        # assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def normalize(self, x, mask):
        x_valid = x[mask]
        splits = mask.sum(dim=(1, 2)).tolist()
        x_valid_list = torch.split(x_valid, splits)
        shift = [x.mean() for x in x_valid_list]
        x_valid_centered = [x - m for x, m in zip(x_valid_list, shift)]
        scale = [x.abs().mean() for x in x_valid_centered]
        scale = torch.stack(scale)
        shift = torch.stack(shift)
        x = (x - shift.view(-1, 1, 1)) / scale.view(-1, 1, 1).clamp(min=1e-6)
        return x

    def distance(self, pred, gt, mask):
        pred = self.normalize(pred, mask)
        gt = self.normalize(gt, mask)
        return torch.abs((pred - gt)[mask])


class ScaleInvLoss(BaseCriterion):
    """scale invariant loss"""

    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, pred, gt, mask):
        assert pred.shape == gt.shape and pred.ndim == 4, f"Bad shape = {pred.shape}"
        dist = self.distance(pred, gt, mask)
        # assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, pred, gt, mask):
        pred_norm_factor = (torch.norm(pred, dim=-1) * mask).sum(
            dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1e-6)
        gt_norm_factor = (torch.norm(gt, dim=-1) * mask).sum(
            dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1e-6)
        pred = pred / pred_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)
        gt = gt / gt_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)
        return torch.norm(pred - gt, dim=-1)[mask]


class ConfLoss_MV(MultiLoss):
    """Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10)

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction("none")

    def get_name(self):
        return f"ConfLoss({self.pixel_loss})"

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gts, preds, **kw):
        # ! is_self not required.
        # compute per-pixel loss
        losses_and_masks, details = self.pixel_loss(gts, preds, **kw)
        # st()
        if "is_self" in details and "img_ids" in details:
            is_self = details["is_self"]
            img_ids = details["img_ids"]
        else:
            is_self = [False] * len(losses_and_masks)
            img_ids = list(range(len(losses_and_masks)))

        # weight by confidence
        conf_losses = []

        for i in range(len(losses_and_masks)):
            pred = preds[img_ids[i]]
            conf_key = "conf_self" if is_self[i] else "conf"
            if not is_self[i]:
                camera_only = gts[0]["camera_only"]
                conf, log_conf = self.get_conf_log(
                    pred[conf_key][~camera_only][losses_and_masks[i][1]])
            else:
                conf, log_conf = self.get_conf_log(
                    pred[conf_key][losses_and_masks[i][1]])

            conf_loss = losses_and_masks[i][0] * conf - self.alpha * log_conf
            conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
            conf_losses.append(conf_loss)

            if is_self[i]:
                details[self.get_name() +
                        f"_conf_loss_self/{img_ids[i]+1}"] = float(conf_loss)
            else:
                details[self.get_name() +
                        f"_conf_loss/{img_ids[i]+1}"] = float(conf_loss)

        details.pop("is_self", None)
        details.pop("img_ids", None)

        final_loss = sum(conf_losses) / len(conf_losses) * 2.0
        if "pose_loss" in details:
            # final_loss = (final_loss + details["pose_loss"].clip(max=0.3) * 5.0
            #               )  # , details
            final_loss = (final_loss + details["pose_loss"]
                          )  # ! vggt does not weight this loss term
        if "scale_loss" in details:
            final_loss = final_loss + details["scale_loss"]
        return final_loss, details


def check_and_fix_inf_nan(loss_tensor, loss_name, hard_max=100):
    """
    Checks if 'loss_tensor' contains inf or nan. If it does, replace those 
    values with zero and print the name of the loss tensor.

    Args:
        loss_tensor (torch.Tensor): The loss tensor to check.
        loss_name (str): Name of the loss (for diagnostic prints).

    Returns:
        torch.Tensor: The checked and fixed loss tensor, with inf/nan replaced by 0.
    """

    if torch.isnan(loss_tensor).any() or torch.isinf(loss_tensor).any():
        for _ in range(10):
            print(f"{loss_name} has inf or nan. Setting those values to 0.")
        loss_tensor = torch.where(
            torch.isnan(loss_tensor) | torch.isinf(loss_tensor),
            torch.tensor(0.0, device=loss_tensor.device), loss_tensor)

    loss_tensor = torch.clamp(loss_tensor, min=-hard_max, max=hard_max)

    return loss_tensor


# https://github.com/facebookresearch/vggt/blob/4d9f4be1e85c44efda107828ec9a09535820bdfc/training/loss.py#L124
def camera_loss_single(cur_pred_pose_enc, gt_pose_encoding, loss_type="huber"):
    if loss_type == "l1":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).abs()
        loss_R = (cur_pred_pose_enc[..., 3:7] -
                  gt_pose_encoding[..., 3:7]).abs()
        loss_fl = (cur_pred_pose_enc[..., 7:] -
                   gt_pose_encoding[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (cur_pred_pose_enc[..., :3] - gt_pose_encoding[..., :3]).norm(
            dim=-1, keepdim=True)
        loss_R = (cur_pred_pose_enc[..., 3:7] -
                  gt_pose_encoding[..., 3:7]).norm(dim=-1)
        loss_fl = (cur_pred_pose_enc[..., 7:] -
                   gt_pose_encoding[..., 7:]).norm(dim=-1)
    elif loss_type == "huber":
        loss_T = huber_loss(cur_pred_pose_enc[..., :3],
                            gt_pose_encoding[..., :3])
        loss_R = huber_loss(cur_pred_pose_enc[..., 3:7], gt_pose_encoding[...,
                                                                          3:7])
        loss_fl = huber_loss(cur_pred_pose_enc[..., 7:], gt_pose_encoding[...,
                                                                          7:])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_fl = check_and_fix_inf_nan(loss_fl, "loss_fl")

    # loss_T = loss_T.clamp(max=100) # TODO: remove this
    loss_T = loss_T.mean()
    loss_R = loss_R.mean()
    loss_fl = loss_fl.mean()

    return loss_T, loss_R, loss_fl


class VGGTLoss(MultiLoss):

    def __init__(self, use_camera_head=True, norm_mode="?avg_dis", valid_range=-1):
        super().__init__()

        # the output track is relative
        self.use_camera_head = use_camera_head
        self.norm_mode = norm_mode
        self.valid_range = valid_range

        if self.norm_mode.startswith('?'):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = self.norm_mode[1:]

    def get_name(self):
        return f"VGGTLoss"

    def get_pts3d_from_views(self,
                             gt_views,
                             dist_clip=None,
                             local=False,
                             key_word="pts3d"):
        """Get point clouds and valid masks for multiple views."""
        gt_pts_list = []
        valid_mask_list = []

        if key_word == "pts3d":
            mask_key_word = "valid_mask"  # TODO, add sky_mask
        # elif key_word == "track":
        #     mask_key_word = "track_valid_mask"
        else:
            raise ValueError(f"Invalid key_word: {key_word}")

        if not local:  # compute the inverse transformation for the anchor view (first view)
            inv_matrix_anchor = inv(gt_views[0]["camera_pose"].float())

        for gt_view in gt_views:
            if local:
                # Rotate GT points to align with the local camera origin for supervision
                inv_matrix_local = inv(gt_view["camera_pose"].float())
                gt_pts = geotrf(inv_matrix_local, gt_view[key_word]
                                )  # Transform GT points to local view's origin
            else:
                # Use the anchor view (first view) transformation for global loss
                gt_pts = geotrf(
                    inv_matrix_anchor,
                    gt_view[key_word])  # Transform GT points to anchor view

            valid_gt = gt_view[mask_key_word].clone()

            if dist_clip is not None:
                dis = gt_pts.norm(dim=-1)  # type: ignore
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

    def get_norm_factor_poses(self, gt_trans):
        # gt_trans: B S 3

        if self.norm_mode:
            gt_trans = einops.rearrange(
                gt_trans, 'b s c -> b s 1 1 c')  # B S 3 --> B S 1 1 3
            valid_mask = torch.ones_like(gt_trans[..., 0],
                                         dtype=torch.bool)  # B S 1 1
            _, norm_factor_gt = normalize_pointcloud(gt_trans, valid_mask)

        else:
            norm_factor_gt = torch.ones(len(gt_trans),
                                        dtype=gt_trans[0].dtype,
                                        device=gt_trans[0].device)

        return norm_factor_gt

    def compute_loss(self, gts, preds, **kw):
        details = {}
        self_name = type(self).__name__

        # flags for different loss choice
        is_metric = gts[0]["is_metric"]
        camera_only = gts[0]["camera_only"]
        # print("debugging camera only and is_metric !!!")
        # camera_only[0] = True  # ! debug
        # is_metric[0] = True  # ! debug

        # depth_only = gts[0]["depth_only"] # not applicable for all the points.
        single_view = gts[0]["single_view"]

        gt_pts3d_global, valid_mask_global = self.get_pts3d_from_views(
            gts, key_word="pts3d", **kw)  # B, N, H, W, C
        gt_pts3d_local, valid_mask_local = self.get_pts3d_from_views(
            gts, key_word="pts3d", local=True, **kw)  # B, N, H, W, C
        gt_extrinsics, gt_intrinsics, image_size_hw = self.get_camera_from_views(
            gts)

        # stack all views predic
        num_pose_iteration = len(preds[0]['camera_pose'])
        num_views = len(preds)
        pred_pts3d_global = torch.stack(
            [preds[i]["pts3d_in_other_view"] for i in range(num_views)], dim=1)
        pred_conf_global = torch.stack(
            [preds[i]["conf"] for i in range(num_views)], dim=1)
        pred_pts3d_local = torch.stack(
            [preds[i]["pts3d_in_self_view"] for i in range(num_views)], dim=1)
        pred_conf_local = torch.stack(
            [preds[i]["conf_self"] for i in range(num_views)], dim=1)
        # st()
        pred_pose_enc_list = [
            torch.stack([preds[j]["camera_pose"][i] for j in range(num_views)],
                        dim=1) for i in range(num_pose_iteration)
        ]

        # ? skys
        # valids = [gt["valid_mask"].clone() for gt in gts]
        # skys = [gt["sky_mask"] & ~valid for gt, valid in zip(gts, valids)]

        # ! process camera_only for local pts3d
        if any(camera_only):
            # this is equal to a loss for camera intrinsics
            gt_pts3d_local = torch.where(
                camera_only[:, None, None, None, None],
                (gt_pts3d_local / gt_pts3d_local[..., -1:].clip(1e-6)).clip(
                    -2, 2),
                gt_pts3d_local,
            )
            pred_pts3d_local = torch.where(
                camera_only[:, None, None, None, None],
                (pred_pts3d_local /
                 pred_pts3d_local[..., -1:].clip(1e-6)).clip(-2, 2),
                pred_pts3d_local,
            )

        # st()
        pred_pts3d_global = pred_pts3d_global[
            ~camera_only]  # remove camera only views
        pred_conf_global = pred_conf_global[~camera_only]
        gt_pts3d_global = gt_pts3d_global[~camera_only]
        valid_mask_global = valid_mask_global[~camera_only]
        is_metric_global = is_metric[~camera_only]

        # loss for pts3d global
        loss_pts3d_global = point_loss(
            pred_pts3d_global,
            pred_conf_global,
            gt_pts3d_global,
            valid_mask_global,
            gradient_loss="grad",
            normalize_pred=False,
            normalize_gt=True,
            gt_pts3d_scale=None,  # TODO, set to metric scale
            metric_flag=is_metric_global,  # for global loss
            temporal_matching_loss=False, 
            valid_range=self.valid_range)

        gt_pts3d_scale = loss_pts3d_global[
            "gt_pts3d_scale"]  # a global scale calculated from all the views
        # not_metric_mask = ~is_metric
        # pose_norm_factor_gt = gt_pts3d_scale.clone()

        if any(camera_only
               ):  # ! use gt pose translation to normalize camera pose
            # ! need to recreate a Tensor here for pose_norm_factor_gt
            pose_norm_factor_gt = torch.ones(camera_only.shape[0],
                                                  device=gt_pts3d_scale.device)

            # gt_trans = [gt[..., :3] for gt in gt_extrinsics] # pr_trans shall also align with gt_pts3d_scale
            # gt_trans = [x[:, None, None, :].clone() for x in gt_trans]
            # valids = [
            #     torch.ones_like(x[..., 0], dtype=torch.bool) for x in gt_trans
            # ]

            # gt_trans = [gt[..., :3, 3] for gt in gt_extrinsics]
            gt_trans = gt_extrinsics[..., :3, 3]  # B S 3
            pose_only_norm_factor_gt = (self.get_norm_factor_poses(gt_trans))
        
            # st()

            pose_norm_factor_gt[camera_only] = pose_only_norm_factor_gt[
                camera_only]
            pose_norm_factor_gt[~camera_only] = gt_pts3d_scale.clone(
            )  # B == len(~camera_only)

        else:
            pose_norm_factor_gt = gt_pts3d_scale.clone()
        
        # for camera_only pts3d, set the scale to 1 for camera_only samples
        local_gt_pts3d_scale = torch.ones(camera_only.shape[0],
                                          device=gt_pts3d_scale.device)
        local_gt_pts3d_scale[~camera_only] = gt_pts3d_scale.clone()

        # loss for pts3d local
        loss_pts3d_local = point_loss(
            pred_pts3d_local,
            pred_conf_local,
            gt_pts3d_local,
            valid_mask_local,
            gradient_loss="grad",
            normalize_pred=False,
            normalize_gt=True,
            gt_pts3d_scale=local_gt_pts3d_scale,  # 
            temporal_matching_loss=False, 
            valid_range=self.valid_range)

        # loss for camera
        # ! TODO, normalize camera first
        loss_camera = camera_loss(
            pred_pose_enc_list,
            gt_extrinsics,
            gt_intrinsics,
            image_size_hw,
            loss_type="huber",
            gt_pts3d_scale=pose_norm_factor_gt,
            #   gt_pts3d_scale=None, # will normalize first
            pose_encoding_type="relT_quaR_FoV")

        # total loss
        pts3d_loss = loss_pts3d_global["loss_conf"] + loss_pts3d_global[
            "loss_grad"]
        pts3d_local_loss = loss_pts3d_local["loss_conf"] + loss_pts3d_local[
            "loss_grad"]
        total_loss = pts3d_loss + pts3d_local_loss + loss_camera[
            "loss_camera"]  # same loss weight as in vggt paper.
        
        details['norm_factor_cross_gt'] = gt_pts3d_scale.clone()
        details['norm_factor_cross_pr'] = torch.ones_like(gt_pts3d_scale) # force pred scale to normalized gt scale

        # logs
        details[self_name + "_pts3d_loss" + "/00"] = float(pts3d_loss)
        details[self_name + "_pts3d_loss" + "_conf" + "/00"] = float(
            loss_pts3d_global["loss_conf"])
        details[self_name + "_pts3d_loss" + "_grad" + "/00"] = float(
            loss_pts3d_global["loss_grad"])

        details[self_name + "_pts3d_local_loss" +
                "/00"] = float(pts3d_local_loss)
        details[self_name + "_pts3d_local_loss" + "_conf" + "/00"] = float(
            loss_pts3d_local["loss_conf"])
        details[self_name + "_pts3d_local_loss" + "_grad" + "/00"] = float(
            loss_pts3d_local["loss_grad"])

        details[self_name + "_camera_loss" + "_loss_camera" + "/00"] = float(
            loss_camera["loss_camera"])
        details[self_name + "_camera_loss" + "_loss_T" + "/00"] = float(
            loss_camera["loss_T"])
        details[self_name + "_camera_loss" + "_loss_R" + "/00"] = float(
            loss_camera["loss_R"])
        details[self_name + "_camera_loss" + "_loss_fl" + "/00"] = float(
            loss_camera["loss_fl"])

        # st()

        return total_loss, details
