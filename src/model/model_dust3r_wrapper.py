from dataclasses import dataclass
import torchvision
import numpy as np
import os
import trimesh
import einops
import imageio
import json
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable, Any

from pdb import set_trace as st

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn, optim

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..loss.loss_dust3r import ConfLoss, L21, MSE
# from ..loss.loss_point import Regr3D as Regr3D_dust3r
from ..loss.loss_mast3r import Regr3D as Regr3D_mast3r
from ..loss.loss_mast3r_mv import ConfLoss_MV, Regr3D_MV
# from ..loss.loss_mast3r_mv_vggt_merge import ConfLoss_MV, Regr3D_MV
from ..loss.loss_mast3r_mv_vggt_merge import VGGTLoss
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, get_pnp_pose
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.nn_module_tools import convert_to_buffer
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .ply_export import export_ply
from .model_wrapper import ModelWrapper, OptimizerCfg, TestCfg, TrainCfg
from src.misc.cam_utils import camera_normalization
from ..visualization.color_map import apply_color_map_to_image, colorize_depth
from src.dust3r.utils.geometry import geotrf, inv


class ModelWrapperMust3R(ModelWrapper):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        distiller: Optional[nn.Module] = None,
        freeze_gs_head=True,
    ) -> None:
        super().__init__(optimizer_cfg,
                         test_cfg,
                         train_cfg,
                         encoder,
                         encoder_visualizer,
                         decoder,
                         losses,
                         step_tracker,
                         distiller,
                         freeze_gs_head=freeze_gs_head)

        # ! criterion from mast3r for pre-training
        # self.dust3r_criterion = eval("ConfLoss(Regr3D_mast3r(L21, norm_mode='?avg_dis'), alpha=0.2) + RGBLoss(MSE)")
        # self.dust3r_criterion = eval(
        #     "ConfLoss(Regr3D_mast3r(L21, norm_mode='?avg_dis', loss_in_log='after'), alpha=0.2)"
        # )

        # self.must3r_criterion = eval(
        #     "ConfLoss(Regr3D_mast3r(L21, norm_mode='?avg_dis', loss_in_log='after'), alpha=0.2)"
        # )

        self.must3r_mv_criterion = eval(
            # "ConfLoss_MV(Regr3D_MV(L21, norm_mode='?avg_dis', loss_in_log='after'), alpha=0.2)"
            # "VGGTLoss(use_camera_head=True, norm_mode='?avg_dis', valid_range=-1)"
            "VGGTLoss(use_camera_head=True, norm_mode='?avg_dis', valid_range=0.95)"
        )

        self.target_keys = [
            'depthmap', 'pts3d', 'valid_mask', 'sky_mask', 'ray_map',
            'camera_pose', 'pts3d', 'is_metric_scale', 'camera_only',
            'is_metric', 'camera_intrinsics', 'img', 'depth_only',
            'single_view', 'img_mask'
        ]

        self.ignore_gs = True  # no need to predict gs parameters

    def _prep_pred_and_gt_paired(self, views, visualization_dump):

        # V = len(views)
        V = visualization_dump['means'].shape[1]
        assert V == 2
        pred1 = {
            'pts3d': visualization_dump['means'][:, 0].squeeze(-2),
            'conf': visualization_dump['conf'][:, 0],
        }
        gt1 = {k: v for k, v in views[0].items() if k in self.target_keys}

        # if V==2: # paired_pred
        pred2 = {
            'pts3d_in_other_view': visualization_dump['means'][:,
                                                               1].squeeze(-2),
            'conf': visualization_dump['conf'][:, 1],
        }

        gt2 = {k: v for k, v in views[1].items() if k in self.target_keys}
        return pred1, pred2, gt1, gt2

    def _prep_pred_and_gt_mv(self, views, visualization_dump):

        # V = len(views)

        # assert V > 2
        V = visualization_dump['means'].shape[1]
        # st()

        all_preds = [
            {
                'pts3d_in_other_view':
                visualization_dump['means'][:, v].squeeze(-2),
                'conf':
                visualization_dump['conf'][:, v],
                'camera_pose':
                # visualization_dump['camera_pose'][:, v, :] # ! if returning a single list
                [
                    camera_pose[:, v, :]
                    for camera_pose in visualization_dump['camera_pose']
                ]  # ! if iterative prediction
                if visualization_dump.get('camera_pose') is not None else
                None,  # ! align with the current camera gt first
                **{
                    k:
                    visualization_dump[k][:, v] if k in visualization_dump else None
                    for k in ['pts3d_in_self_view', 'conf_self']
                }
                # 'pts3d_in_self_view': visualization_dump['pts3d_in_self_view'][:, v] if 'pts3d_in_self_view' in visualization_dump else None,
                # 'conf_self': visualization_dump['conf_self'][:, v] if 'conf_self' in visualization_dump else None,
            } for v in range(0, V)
        ]

        all_gts = [{
            k: v
            for k, v in views[v].items() if k in self.target_keys
        } for v in range(0, V)]

        return all_preds, all_gts

    def save_pts3d(self, pts3d, output_path, output_suffix, rgb=None, v=False):
        assert isinstance(pts3d, torch.Tensor)
        assert pts3d.shape[0] == 1
        pts3d = pts3d.reshape(-1, 3).cpu().numpy()

        pcd = trimesh.Trimesh(vertices=pts3d, vertex_colors=rgb)

        pcd_output_path = output_path / f'step_{self.global_step}-batch_{batch_idx}-{output_suffix}.obj'
        pcd.export(pcd_output_path, 'obj')
        if v:
            print('pcd saved to ', pcd_output_path)

    @rank_zero_only
    @torch.inference_mode()
    def log_step(self,
                 batch,
                 views,
                 depth_preds,
                 wandb_log=False,
                 save_suffix=''):

        # Construct comparison image.
        # st()
        context_img = inverse_normalize(
            batch["context"]["image"][0])  # [-1,1] -> [0,1]

        # rgb_gt = inverse_normalize(rgb_gt[0])  # normalize to [0,1]
        # rgb_gt = batch['target']['image'][0]  # normalize to [0,1]
        # st()
        depth_gt = [
            colorize_depth(view['depthmap'][0].cpu().numpy(),
                           cmap='Spectral_r') / 255.0 for view in views
        ]  # retrieve the first view
        depth_gt = torch.from_numpy(
            rearrange(np.stack(depth_gt), 'v h w c -> v c h w'))

        # gs predictions, make then all in [0,1]
        # rgb_pred = inverse_normalize(gs_output.color[0])
        # rgb_pred = gs_output.color[0]
        # st()
        depth_pred = [
            colorize_depth(depth, cmap='Spectral_r') / 255.0
            for depth in depth_preds[0].detach().cpu().numpy()
        ]
        depth_pred = torch.from_numpy(
            rearrange(np.stack(depth_pred), 'v h w 1 1 c -> v c h w'))

        comparison = hcat(
            add_label(
                vcat(*context_img),
                f'{batch["scene"]["dataset"][0][0]}-{batch["scene"]["label"][0][0]}'
                # f'RGB (Context)-{batch["scene"]["dataset"][0][0]}-{batch["scene"]["label"][0][0]}'
            ),
            # add_label(vcat(*rgb_gt), "RGB (Ground Truth)"),
            add_label(vcat(*depth_gt), "Depth (Ground Truth)"),
            # add_label(vcat(*rgb_pred), "RGB (Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
            # add_label(vcat(*frustum_mask), "FrustumMask (Ground Truth)")
        )

        # st()  # check the caption, dataset-label
        if wandb_log:  # must provide a list
            self.logger.log_image(
                f"comparison-Training_{self.training}",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=[
                    f'{batch["scene"]["dataset"][0][0]}-{batch["scene"]["label"][0][0]}'
                ],
            )
        # else:

        # st()
        torchvision.utils.save_image(
            comparison,
            os.path.join(self.logger.save_dir,
                         f'{self.global_step}{save_suffix}.jpg'),
            normalize=True,
            value_range=(0, 1))

        return comparison

    @rank_zero_only
    def validation_step(self, views, batch_idx, dataloader_idx=0):

        return # ignore for now

        if batch_idx > 10:
            return

        # V = len(views)
        batch, V = self._preprocess_views(views, return_ctx_indices=False)  # type: ignore
        b, _, _, h, w = batch["context"]["image"].shape  # type: ignore

        output_path = Path(self.logger.save_dir)  # type: ignore

        assert b == 1

        # st()

        # visualization_dump = {}
        visualization_dump = {
            'ignore_gs': self.ignore_gs
        }  # ! since imposing depth / pts3d loss here
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )

        # st()
        scale_factors = {}

        if V == 2:  # for compat issue, use mast3r default loss class
            pred1, pred2, gt1, gt2 = self._prep_pred_and_gt_paired(
                views, visualization_dump)

            # dust3r_loss, dust3r_loss_details = self.dust3r_criterion(
            #     gt1, gt2, pred1, pred2)

        else:  # cut3r/must3r-like loss class

            all_preds, all_gts = self._prep_pred_and_gt_mv(
                views, visualization_dump)

            # st()
            dust3r_loss, dust3r_loss_details = self.must3r_mv_criterion(
                all_gts, all_preds)

            self.log("val/loss/Reg3d/loss_sum", dust3r_loss)
            for k, v in dust3r_loss_details.items():  # log details
                self.log(f"val/loss/Reg3d/{k}", v)

            scale_factors = {
                k: dust3r_loss_details.pop(k)
                for k in ['norm_factor_cross_gt', 'norm_factor_cross_pr']
            }

        if self.global_rank == 0:
            print(f"validation step {self.global_step}; "
                  f"scene = {batch['scene']['label'][0]}; "
                  f"context = {batch['context']['index']['instance']}")

            _ = self.log_step(
                # all_preds, all_gts
                batch,
                views,
                depth_preds=visualization_dump['depth'],
                wandb_log=True,
                save_suffix=f'-val_batch-{batch_idx}')

        if self.dust3r_criterion is not None:
            # with torch.no_grad():
            #     pseudo_gt1, pseudo_gt2 = self.distiller(
            #         batch["context"], False)
            # depth1, depth2 = pred1['pts3d'][
            #     ..., -1], pred2['pts3d_in_other_view'][..., -1]
            # conf1, conf2 = pred1['conf'], pred2['conf']
            # depth_dust = torch.cat([depth1, depth2], dim=0)
            # depth_dust = vis_depth_map(depth_dust)
            # conf_dust = torch.cat([conf1, conf2], dim=0)
            # conf_dust = confidence_map(conf_dust)
            # dust_vis = torch.cat([depth_dust, conf_dust], dim=0)
            # dust_vis = depth_dust
            # comparison = hcat(add_label(vcat(*dust_vis), "Context"),
            #                   comparison)
            # comparison = add_label(vcat(*dust_vis), "Pred-depth_conf")

            def save_pts3d(pts3d, output_suffix, rgb=None, v=False):
                assert isinstance(pts3d, torch.Tensor)
                assert pts3d.shape[0] == 1
                pts3d = pts3d.reshape(-1, 3).cpu().numpy()

                pcd = trimesh.Trimesh(vertices=pts3d, vertex_colors=rgb)

                pcd_output_path = output_path / f'step_{self.global_step}-batch_{batch_idx}-{output_suffix}.obj'
                pcd.export(pcd_output_path, 'obj')
                if v:
                    print('pcd saved to ', pcd_output_path)

            if V == 2:

                rgb_0 = (batch["context"]["image"][0][0].reshape(
                    3, -1).permute(1, 0) * 127.5 + 127.5).cpu().numpy().astype(
                        np.uint8)
                rgb_1 = (batch["context"]["image"][0][1].reshape(
                    3, -1).permute(1, 0) * 127.5 + 127.5).cpu().numpy().astype(
                        np.uint8)

                # ! save pcd also for visualization
                save_pts3d(pred1['pts3d'], 'v1-pred', v=True, rgb=rgb_0)
                save_pts3d(pred2['pts3d_in_other_view'],
                           'v2-pred',
                           v=True,
                           rgb=rgb_1)

                # if self.global_step == 0:

                save_pts3d(gt1['pts3d'], 'v1-gt', v=True, rgb=rgb_0)
                save_pts3d(gt2['pts3d'], 'v2-gt', v=True, rgb=rgb_1)

            else:  # for V>2 views, save all stuffs together
                all_rgb = [
                    (batch["context"]["image"][0][v].reshape(3, -1).permute(
                        1, 0) * 127.5 + 127.5).cpu().numpy().astype(np.uint8)
                    for v in range(V)
                ]
                all_pred_pts = [
                    all_preds[v]['pts3d_in_other_view'] for v in range(0, V)
                ]

                in_camera1 = inv(views[0]["camera_pose"][0])
                all_gt_pts = [
                    geotrf(in_camera1, views[v]['pts3d'][0])
                    for v in range(0, V)
                ]

                all_rgb = np.concatenate(all_rgb, 0)
                # st()

                all_pred_pts = rearrange(torch.cat(all_pred_pts, 0),
                                         'b h w c -> 1 (b h w) c')
                # st()
                all_gt_pts = rearrange(
                    torch.stack(all_gt_pts, 0),
                    'v h w c -> 1 (v h w) c')  # for compat issue
                # transform gt_pts to canonical frame

                save_pts3d(all_pred_pts /
                           scale_factors['norm_factor_cross_pr'][0].squeeze(),
                           'pred-all',
                           v=True,
                           rgb=all_rgb)
                save_pts3d(all_gt_pts /
                           scale_factors['norm_factor_cross_gt'][0].squeeze(),
                           'gt-all',
                           v=True,
                           rgb=all_rgb)

        # st()

        # self.logger.log_image(
        #     "comparison",
        #     [prep_image(add_border(comparison))],
        #     step=self.global_step,
        #     caption=batch["scene"]['label'],
        # )

        # Render Gaussians.
        '''
        output = self.decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            "depth",
        )
        rgb_pred = output.color[0]
        depth_pred = vis_depth_map(output.depth[0])

        # direct depth from gaussian means (used for visualization only)
        gaussian_means = visualization_dump["depth"][0].squeeze()
        if gaussian_means.shape[-1] == 3:
            gaussian_means = gaussian_means.mean(dim=-1)

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim)

        # Construct comparison image.
        context_img = inverse_normalize(batch["context"]["image"][0])
        context_img_depth = vis_depth_map(gaussian_means)
        context = []
        for i in range(context_img.shape[0]):
            context.append(context_img[i])
            context.append(context_img_depth[i])
        comparison = hcat(
            add_label(vcat(*context), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
        )
        '''

        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.

        if not self.ignore_gs:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                projections = hcat(*render_projections(
                    gaussians,
                    256,
                    extra_label="",
                )[0])

            self.logger.log_image(
                "projection",
                [prep_image(add_border(projections))],
                step=self.global_step,
            )

        # Draw cameras.
        # cameras = hcat(*render_cameras(batch, 256))
        # self.logger.log_image("cameras", [prep_image(add_border(cameras))],
        #                       step=self.global_step)

        # if self.encoder_visualizer is not None:
        #     for k, image in self.encoder_visualizer.visualize(
        #             batch["context"], self.global_step).items():
        #         self.logger.log_image(k, [prep_image(image)],
        #                               step=self.global_step)

        # Run video validation step.
        # with torch.amp.autocast(device_type="cuda", enabled=False):
        #     # self.render_video_wobble(batch)
        #     self.render_video_interpolation(batch)
        #     if self.train_cfg.extended_visualization:
        #         self.render_video_interpolation_exaggerated(batch)

        # st()
        # pass

    def _merge_list_of_views(self, views):

        assert isinstance(views, list)

        batch_combined = None
        batch_combined = {}  # merge list of views into a single batch
        V = len(views)

        # ! merge
        for k, v in views[0].items():

            if isinstance(v, list):
                batch_combined[k] = [views[i][k]
                                     for i in range(V)]  # list of list
            if isinstance(v, torch.Tensor):
                batch_combined[k] = torch.stack(
                    [views[i][k] for i in range(V)], dim=1)  # B V ...

        return batch_combined

    @torch.autocast(device_type="cuda", enabled=False)
    def _preprocess_views(self,
                          views,
                          img_mask=None,
                          ray_mask=None,
                          add_target=False,
                          for_nvs=False, 
                          return_ctx_indices=False):
        # to merge a list of inputs to batch
        # list of views [B C H W] -> B V C H W
        # batch = {}

        batch_combined = self._merge_list_of_views(views)
        #       dict_keys(['img', 'depthmap', 'camera_pose', 'camera_intrinsics', 'dataset', 'label', 'instance', 'is_metric', 'is_video', 'quantile', 'img_mask', 'ray_mask', 'camera_only'
        # , 'depth_only', 'single_view', 'reset', 'idx', 'true_shape', 'sky_mask', 'ray_map', 'pts3d', 'valid_mask', 'rng'])

        # ! update the key
        # intrinsic = [input_camera_intrinsics[0,0] / w, input_camera_intrinsics[1,1] / h, input_camera_intrinsics[0,2] / w, input_camera_intrinsics[1,2] / h, 0.0, 0.0]
        # st()
        b, v, _, w, h = batch_combined['img'].shape[:]

        # ! for 3dgs rendering, use normalized K
        if 'camera_intrinsics' in batch_combined:
            intrinsics = batch_combined['camera_intrinsics'].clone()
            intrinsics[..., 0, 0] /= w
            intrinsics[..., 0, 2] /= w
            intrinsics[..., 1, 1] /= h
            intrinsics[..., 1, 2] /= h

        else:
            intrinsics = None

        if 'camera_pose' in batch_combined:  # ! during demo.py, no [E|K] available
            # extrinsics = batch_combined['camera_pose'].inverse()  # B V 4, 4
            extrinsics = batch_combined['camera_pose']  # B V 4, 4
        else:
            extrinsics = None
            assert not add_target

        if for_nvs:  # split context, target following nvs schedule
            context_indices = torch.arange(
                start=0,
                end=v,
                # end=3,
                # end=2,
                # end=v // 2,
                step=1,
                dtype=torch.long).to(intrinsics.device)
            '''
            target_indices = torch.arange(
                # start=0, end=v, step=1, dtype=torch.long
                start=v // 2 - 2,
                # start=0,
                # start=2,
                # start=, # mimic noposplat setting
                end=v,
                step=1,
                dtype=torch.long
            ).to().to(
                # start=max(0, v // 2 - 2), end=v, step=1, dtype=torch.long).to(
                # start=v // 2 - 1, end=v, step=1, dtype=torch.long).to(
                intrinsics.device)  # ! supervise input views also
            '''
            # st()
            indices = torch.randperm(v,
                                     dtype=torch.long,
                                     device=intrinsics.device)
            assert v >= 4
            # context_indices_for_render = indices[:v //
            #                                      2]  # randomly pick two gaussian maps for rendering supervision
            # context_indices_for_render = context_indices # avoid holes since fg mask is not stable
            target_indices = indices[v // 2 - 2:]
        else:
            context_indices = torch.arange(0, b, dtype=torch.long)
            target_indices = context_indices.clone()

        batch_combined_compat = {
            'context': {
                'overlap': -1,  # ! placeholder
                'image': batch_combined['img'],
                'intrinsics': intrinsics,
                'extrinsics': extrinsics,
                'index': {
                    k: batch_combined.get(k, None)
                    for k in [
                        'idx',
                        'instance',
                    ]
                },
                **{
                    k: batch_combined.get(k, None)
                    for k in [
                        'ray_mask', 'img_mask', 'true_shape', 'camera_pose', 'valid_mask', 'pts3d', 'depthmap'
                    ]
                },
                # 'near': -1, # TODO
                # 'far': -1,
            },
            'scene': {
                k: batch_combined.get(k, None)
                for k in ['label', 'dataset']
            },
            # 'metadata': {
            #     # k: batch_combined[k]
            #     k: batch_combined.get(k, None)
            #     for k in [
            #         'single_view',
            #         'reset',
            #         'depth_only',
            #         'camera_only',
            #         'quantile',
            #         'is_video',
            #         'is_metric',
            #         'rng',
            #         'valid_mask'
            #     ]
            # }
        }

        batch_context = batch_combined_compat['context']

        # ! normalize extrinsics first
        extrinsics = batch_context['extrinsics'].clone()
        normalized_extrinsics = torch.stack([
            camera_normalization(extrinsics[idx, 0:1], extrinsics[idx, :])
            for idx in range(b)
        ],
                                            dim=0)

        # st()
        batch_context['extrinsics'] = normalized_extrinsics

        # ! update 'target' before in-place update 'context'
        if add_target:  # if adding rendering loss for the dust3r dataset
            batch_combined_compat.update({
                # 'target': {
                # k: batch_combined[k]
                # for k in ['depthmap', 'pts3d', 'valid_mask', 'sky_mask', 'ray_map']
                # },
                # ! self reconstruction first
                'target': {
                    **{
                        k: batch_combined_compat['context'][k][:, target_indices]
                        for k in [
                            'image', 'intrinsics', 'valid_mask', 'extrinsics', 'camera_pose', 'pts3d', 'depthmap'
                        ]  # camera_pose is un-canonicalized
                    },
                    # ! overfit, use co3d for now
                    # 'near': torch.ones(b, v, device=self.device) * 0.5,
                    # 'far': torch.ones(b, v, device=self.device) * 40,
                    'near':
                    torch.ones(b, len(target_indices), device=self.device) * 1,
                    'far':
                    torch.ones(b, len(target_indices), device=self.device) *
                    100,
                    "scene_scale":
                    torch.ones(b, len(target_indices), device=self.device) *
                    1,  # ! for co3d also
                    'normalize_pts':
                    False,
                }
            })

        if for_nvs:
            for parent_key in batch_context:  # ! retrieve context_indices

                if isinstance(batch_context[parent_key], torch.Tensor):
                    batch_context[parent_key] = batch_context[
                        parent_key][:, context_indices]

                elif isinstance(batch_context[parent_key], dict):
                    for sub_key in batch_context[parent_key]:
                        if isinstance(batch_context[parent_key][sub_key],
                                      torch.Tensor):
                            batch_context[parent_key][sub_key] = batch_context[
                                parent_key][sub_key][:, context_indices]

            # copy context extrinsics directly for now
            # context_extrinsics = batch_combined_compat['context']['extrinsics']
            # target_extrinsics = context_extrinsics.clone()
            # batch_combined_compat['target']['extrinsics'] = extrinsics

            # make relative pose
            # batch_combined_compat['target']['extrinsics'] = target_extrinsics
            # st()

            # batch_combined_compat['target'][
            #     'extrinsics'] = batch_combined_compat['target'][
            #         'extrinsics'][:, target_indices]

        # return batch_combined_compat, v
        if return_ctx_indices:
            return batch_combined_compat, len(
                context_indices) if for_nvs else v, context_indices
        else:
            return batch_combined_compat, len(
                context_indices) if for_nvs else v

    # ! for demo.py inference purpose
    def run_step(self, views, enable_loss=False):

        batch, V = self._preprocess_views(
            views, return_ctx_indices=False)  # for views target stuffs, no need to merge

        visualization_dump = {}  # ! since imposing depth / pts3d loss here
        gaussians = self.encoder(
            batch["context"],  # type: ignore
            self.global_step,
            visualization_dump=visualization_dump)
        # total_loss = 0

        res = {
            'gaussians': gaussians,
            'visualization_dump': visualization_dump,
            'batch': batch
        }

        assert V > 2
        # ! dust3r supervision to regularize the pts3d

        all_preds, all_gts = self._prep_pred_and_gt_mv(views,
                                                       visualization_dump)

        res.update({
            'pred': all_preds,
            'views': all_gts,
        })

        if enable_loss:
            dust3r_loss, dust3r_loss_details = self.must3r_mv_criterion(
                all_gts, all_preds)

            res.update({
                'dust3r_loss': dust3r_loss,
                'dust3r_loss_details': dust3r_loss_details,
            })

        return res

    def training_step(self, views, batch_idx):  # views is batch here
        # st()

        # ! debugging
        # print('debugging validation, ignore training step')
        # return torch.tensor(0.0, device=self.device, requires_grad=True)

        batch, V = self._preprocess_views(
            views, return_ctx_indices=False)  # for views target stuffs, no need to merge

        # batch = self.data_shim(batch)  # ?

        _, _, _, h, w = batch["context"]["image"].shape

        # scale_factors = {
        #     k: dust3r_loss_details.pop(k)
        #     for k in ['norm_factor_cross_gt', 'norm_factor_cross_pr']
        # }
        # scale_factors['norm_factor_cross_pr'] = scale_factors['norm_factor_cross_gt']

        # ! catch some failed train samples
        # try:
        # if True:

        # Run the model.
        # visualization_dump = None
        # if self.distiller is not None:
        # visualization_dump = {}  # ! since imposing depth / pts3d loss here
        visualization_dump = {
            'ignore_gs': self.ignore_gs
        }  # ! since imposing depth / pts3d loss here
        # visualization_dump = {'ignore_gs': False}  # ! since imposing depth / pts3d loss here
        gaussians = self.encoder(
            batch["context"],  # type: ignore
            self.global_step,
            visualization_dump=visualization_dump)
        total_loss = 0

        # ! dust3r supervision to regularize the pts3d
        # if self.global_step <= self.train_cfg.distill_max_steps:
        if True:

            # process preds and gts
            # now just dict, later extend to a list of prediction views.

            # pred1 = {
            #     'pts3d': visualization_dump['means'][:, 0].squeeze(-2),
            #     'conf': visualization_dump['conf'][:, 0],
            # }
            # pred2 = {
            #     'pts3d_in_other_view':
            #     visualization_dump['means'][:, 1].squeeze(-2),
            #     'conf':
            #     visualization_dump['conf'][:, 1],
            # }

            # gt1 = {
            #     k: v
            #     for k, v in views[0].items() if k in self.target_keys
            # }
            # gt2 = {
            #     k: v
            #     for k, v in views[1].items() if k in self.target_keys
            # }

            # st()

            # if V == 2:  # for compat issue, use mast3r default loss class
            #     pred1, pred2, gt1, gt2 = self._prep_pred_and_gt_paired(
            #         views, visualization_dump)

            #     dust3r_loss, dust3r_loss_details = self.dust3r_criterion(
            #         gt1, gt2, pred1, pred2)

            # else:  # cut3r/must3r-like loss class

            all_preds, all_gts = self._prep_pred_and_gt_mv(
                views, visualization_dump)

            dust3r_loss, dust3r_loss_details = self.must3r_mv_criterion(
                all_gts, all_preds)

        for k in ['norm_factor_cross_gt', 'norm_factor_cross_pr']:
            dust3r_loss_details.pop(k)

        ''' ignore rendering loss for now
        output = self.decoder.forward(
            gaussians,  # means: 8 * 196608 (65536 * 3) * 3
            batch["target"]
            ["extrinsics"],  # ! camera_pose (requires inverse)
            batch["target"]["intrinsics"],  # camera_intrinsics
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=self.train_cfg.depth_mode,  # unused.
        )
        target_gt = batch["target"]["image"]  # 8, 4, 3, 256, 256

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

        # Compute and log loss.
        for loss_fn in self.losses:
            loss = loss_fn.forward(output, batch, gaussians,
                                    self.global_step)
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss
        '''

        self.log("loss/Reg3d/loss_sum", dust3r_loss)
        for k, v in dust3r_loss_details.items():  # log details
            self.log(f"loss/Reg3d/{k}", v)

        total_loss = total_loss + dust3r_loss
        self.log("loss/total", total_loss)

        if (self.global_rank == 0
                and self.global_step % self.train_cfg.print_log_every_n_steps
                == 0):
            log_to_print = f"train step {self.global_step}; loss = {total_loss:.6f}"
            if "pose_loss" in dust3r_loss_details:
                log_to_print += f"; pose_loss= {dust3r_loss_details['pose_loss']:.6f}"
            print(log_to_print)
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # if self.global_step % 2500 == 0:
        if self.global_step % 500 == 0:
            # if self.global_step % 1 == 0:
            # st()
            if self.global_rank == 0:
                _ = self.log_step(
                    batch,
                    views,
                    depth_preds=visualization_dump['depth'],
                    wandb_log=True,
                    save_suffix=f'-train'
                    # f'-train-{batch["scene"]["dataset"][0][0]}-{batch["scene"]["label"][0][0]}'
                )

            # torch.cuda.empty_cache()  # log img and avoid oom

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        return total_loss

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         # if param.grad is None:
    #         if param.grad is None and param.requires_grad:
    #             print(name)
    #     st()
    #     pass
