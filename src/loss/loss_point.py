import torch
import torch.nn as nn
from copy import copy, deepcopy

from ..geometry.ptc_geometry import geotrf, inv, normalize_pointcloud, depthmap_to_pts3d
# from torchmetrics.functional.regression import pearson_corrcoef
# from pytorch3d.loss import chamfer_distance

def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction


def get_pred_pts3d(gt, pred, use_pose=False):
    if 'depth' in pred and 'pseudo_focal' in pred:
        try:
            pp = gt['camera_intrinsics'][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif 'pts3d' in pred:
        # pts3d from my camera
        pts3d = pred['pts3d']

    elif 'pts3d_in_other_view' in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred['pts3d_in_other_view']  # return!

    if use_pose:
        camera_pose = pred.get('camera_pose')
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


class LLoss (nn.Module):
    """ L-norm loss
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim-1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
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
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
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


class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, LLoss), f'{criterion} is not a proper criterion!'+bb()
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = 'none'  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class ConfLoss (MultiLoss):
    """ Weighted regression by learned confidence.
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
        self.pixel_loss = pixel_loss.with_reduction('none')

    def get_name(self):
        return f'ConfLoss({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gt1, gt2, pred1, pred2, **kw):
        # compute per-pixel loss
        ((loss1, msk1), (loss2, msk2)), details = self.pixel_loss(gt1, gt2, pred1, pred2, **kw)
        if loss1.numel() == 0:
            print('NO VALID POINTS in img1', force=True)
        if loss2.numel() == 0:
            print('NO VALID POINTS in img2', force=True)

        # weight by confidence
        conf1, log_conf1 = self.get_conf_log(pred1['conf'][msk1])
        conf2, log_conf2 = self.get_conf_log(pred2['conf'][msk2])
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2

        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0

        return conf_loss1 + conf_loss2, dict(conf_loss_1=float(conf_loss1), conf_loss2=float(conf_loss2), **details)


class Regr3D(nn.Module):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(self, norm_mode='avg_dis', alpha=0.2, gt_scale=False):
        super().__init__()
        self.norm_mode = norm_mode
        self.alpha = alpha
        self.gt_scale = gt_scale

    def get_conf_log(self, x):
        return x, torch.log(x)

    def forward(self, gt_pts1, gt_pts2, pr_pts1, pr_pts2, conf1=None, conf2=None, dist_clip=None, disable_view1=False):
        valid1 = valid2 = torch.ones_like(conf1, dtype=torch.bool)
        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            valid1 = (dis1 <= dist_clip)
            valid2 = (dis2 <= dist_clip)
        else:
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)

            # only keep the points norm whithin the range of 1% to 99% of each batch
            # Flatten along the H and W dimensions
            dis1_flat = dis1.view(dis1.shape[0], -1)
            dis2_flat = dis2.view(dis2.shape[0], -1)

            # Compute the 0.1% and 99.9% quantiles for each batch
            # quantiles_1 = torch.quantile(dis1_flat, torch.tensor([0.01, 0.99]).to(dis1_flat.device), dim=1)
            # quantiles_2 = torch.quantile(dis2_flat, torch.tensor([0.01, 0.99]).to(dis2_flat.device), dim=1)
            quantiles_1 = torch.quantile(dis1_flat, torch.tensor([0.002, 0.998]).to(dis1_flat.device), dim=1)
            quantiles_2 = torch.quantile(dis2_flat, torch.tensor([0.002, 0.998]).to(dis2_flat.device), dim=1)

            # Create masks based on the quantiles
            valid1 = (dis1 >= quantiles_1[0].view(-1, 1, 1)) & (dis1 <= quantiles_1[1].view(-1, 1, 1))
            valid2 = (dis2 >= quantiles_2[0].view(-1, 1, 1)) & (dis2 <= quantiles_2[1].view(-1, 1, 1))

            # set min confidence to 3
            valid1 = valid1 & (conf1 >= 3)
            valid2 = valid2 & (conf2 >= 3)

        # normalize 3d points
        if self.norm_mode:
            pr_pts1, pr_pts2 = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2)
        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2)

        loss1 = torch.norm(pr_pts1 - gt_pts1, dim=-1)
        loss2 = torch.norm(pr_pts2 - gt_pts2, dim=-1)
        # loss1 = (pr_pts1[..., -1] - gt_pts1[..., -1]).abs()
        # loss2 = (pr_pts2[..., -1] - gt_pts2[..., -1]).abs()

        loss1, loss2 = loss1[valid1], loss2[valid2]

        if disable_view1:
            return loss2.mean()
        return loss1.mean() + loss2.mean()

        # conf1, conf2 = conf1[valid1], conf2[valid2]
        # conf1, conf2 = conf1.softmax(dim=-1), conf2.softmax(dim=-1)
        # loss1 = (loss1 * conf1).sum()
        # loss2 = (loss2 * conf2).sum()
        # return loss1 + loss2
        #
        # # weight by confidence
        # conf1, log_conf1 = self.get_conf_log(conf1[valid1])
        # conf2, log_conf2 = self.get_conf_log(conf2[valid2])
        # conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        # conf_loss2 = loss2 * conf2 - self.alpha * log_conf2
        #
        # # average + nan protection (in case of no valid pixels at all)
        # conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        # conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0
        #
        # return conf_loss1 + conf_loss2

    # def forward(self, gt_pts1, gt_pts2, pr_pts1, pr_pts2, conf1=None, conf2=None, dist_clip=None, disable_view1=False):
    #     # valid1 = valid2 = torch.ones_like(conf1, dtype=torch.bool)
    #     if dist_clip is not None:
    #         # points that are too far-away == invalid
    #         dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
    #         dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
    #         valid1 = (dis1 <= dist_clip)
    #         valid2 = (dis2 <= dist_clip)
    #     else:
    #         dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
    #         dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
    #
    #         # only keep the points norm whithin the range of 1% to 99% of each batch
    #         # Flatten along the H and W dimensions
    #         dis1_flat = dis1.view(dis1.shape[0], -1)
    #         dis2_flat = dis2.view(dis2.shape[0], -1)
    #
    #         # Compute the 0.1% and 99.9% quantiles for each batch
    #         quantiles_1 = torch.quantile(dis1_flat, torch.tensor([0.1, 0.9]).to(dis1_flat.device), dim=1)
    #         quantiles_2 = torch.quantile(dis2_flat, torch.tensor([0.1, 0.9]).to(dis2_flat.device), dim=1)
    #         # quantiles_1 = torch.quantile(dis1_flat, torch.tensor([0.002, 0.998]).to(dis1_flat.device), dim=1)
    #         # quantiles_2 = torch.quantile(dis2_flat, torch.tensor([0.002, 0.998]).to(dis2_flat.device), dim=1)
    #
    #         # Create masks based on the quantiles
    #         valid1 = (dis1 >= quantiles_1[0].view(-1, 1, 1)) & (dis1 <= quantiles_1[1].view(-1, 1, 1))
    #         valid2 = (dis2 >= quantiles_2[0].view(-1, 1, 1)) & (dis2 <= quantiles_2[1].view(-1, 1, 1))
    #
    #         # set min opacity to 3
    #         valid1 = valid1 & (conf1 >= 0.2)
    #         valid2 = valid2 & (conf2 >= 0.2)
    #
    #     # # normalize 3d points
    #     # if self.norm_mode:
    #     #     pr_pts1, pr_pts2 = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2)
    #     # if self.norm_mode and not self.gt_scale:
    #     #     gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2)
    #
    #     # L1 loss
    #     # loss1 = (pr_pts1[..., -1] - gt_pts1[..., -1]).abs()
    #     # loss2 = (pr_pts2[..., -1] - gt_pts2[..., -1]).abs()
    #
    #     # L2 loss
    #     loss1 = torch.norm(pr_pts1 - gt_pts1, dim=-1)
    #     loss2 = torch.norm(pr_pts2 - gt_pts2, dim=-1)
    #     loss1, loss2 = loss1[valid1], loss2[valid2]
    #
    #     # Pearson correlation coefficient loss
    #     # pr_pts1, pr_pts2 = pr_pts1[valid1], pr_pts2[valid2]
    #     # gt_pts1, gt_pts2 = gt_pts1[valid1], gt_pts2[valid2]
    #     # loss1 = 1 - pearson_corrcoef(pr_pts1.view(-1, 3), gt_pts1.view(-1, 3))
    #     # loss2 = 1 - pearson_corrcoef(pr_pts2.view(-1, 3), gt_pts2.view(-1, 3))
    #
    #     # # Chamfer distance loss
    #     # pr_pts = torch.cat([pr_pts1.flatten(1, 2), pr_pts2.flatten(1, 2)], dim=1)
    #     # gt_pts = torch.cat([gt_pts1.flatten(1, 2), gt_pts2.flatten(1, 2)], dim=1)
    #     # valid_mask = torch.cat([valid1.flatten(1, 2), valid2.flatten(1, 2)], dim=1)
    #     # nan_pts_pr, nnz = invalid_to_zeros(pr_pts, valid_mask, ndim=3)
    #     # nan_pts_gt, nnz = invalid_to_zeros(gt_pts, valid_mask, ndim=3)
    #     #
    #     # loss, _ = chamfer_distance(nan_pts_pr, nan_pts_gt, batch_reduction=None, point_reduction=None)
    #     # loss1, loss2 = loss[0], loss[1]
    #     # return loss1.sum() / valid_mask.sum()
    #
    #     if disable_view1:
    #         return loss2.mean()
    #     return loss1.mean() + loss2.mean()
    #
    #     # conf1, conf2 = conf1[valid1], conf2[valid2]
    #     # conf1, conf2 = conf1.softmax(dim=-1), conf2.softmax(dim=-1)
    #     # loss1 = (loss1 * conf1).sum()
    #     # loss2 = (loss2 * conf2).sum()
    #     # return loss1 + loss2
    #     #
    #     # # weight by confidence
    #     # conf1, log_conf1 = self.get_conf_log(conf1[valid1])
    #     # conf2, log_conf2 = self.get_conf_log(conf2[valid2])
    #     # conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
    #     # conf_loss2 = loss2 * conf2 - self.alpha * log_conf2
    #     #
    #     # # average + nan protection (in case of no valid pixels at all)
    #     # conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
    #     # conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0
    #     #
    #     # return conf_loss1 + conf_loss2

