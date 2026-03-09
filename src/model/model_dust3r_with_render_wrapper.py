from dataclasses import dataclass
import matplotlib
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
from ..visualization.color_map import apply_color_map_to_image, colorize_depth
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .ply_export import export_ply
from .model_wrapper import ModelWrapper, OptimizerCfg, TestCfg, TrainCfg

from .model_dust3r_wrapper import ModelWrapperMust3R
from src.model.types import Gaussians
from src.dust3r.utils.geometry import geotrf, inv, get_frustum_mask

from src.loss.loss_zoedepth import GradL1Loss, ScaleAndShiftInvariantLoss


class ModelWrapperMust3RWithRender(ModelWrapperMust3R):
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
        freeze_gs_head=False,
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
        # self.cmap_depth = matplotlib.colormaps.get_cmap('Spectral_r')
        self.enable_depth_loss = False
        # self.enable_depth_loss = True
        if self.enable_depth_loss:
            self.depth_losses = nn.ModuleList([
                GradL1Loss(weight=0.1),
                ScaleAndShiftInvariantLoss(weight=0.25),
            ])

        self.analysis_flag = False
        # self.analysis_flag = True  # save some intermediate results

        # self.sparcity_loss = False
        self.sparcity_loss = True  # encourage opacity=0, encourage scale<scene_scale*0.05

    @rank_zero_only
    @torch.inference_mode()
    def log_step(
            self,
            batch,
            #  rgb_gt,
            gs_output,
            in_frustum_mask_dict,
            wandb_log=False):

        # torchvision.utils.save_image(
        #     rearrange(
        #         torch.cat([
        #             in_frustum_mask_dict['in_frustum_mask'],
        #             batch['target']['valid_mask'],
        #             in_frustum_mask_dict['in_frustum_mask_final']
        #         ], -2).float()[0:1], 'b v h w -> b 1 h (v w)'),
        #     os.path.join(self.logger.save_dir,
        #                     f'{self.global_step}-frustrum_mask.jpg'),

        # frustum_mask = rearrange(
        #     torch.cat([
        #         in_frustum_mask_dict['in_frustum_mask'],
        #         batch['target']['valid_mask'],
        #         in_frustum_mask_dict['in_frustum_mask_final'].squeeze(2)
        #     ], -2).float()[0:1],
        #     'b v h w -> b 1 h (v w)').repeat_interleave(3, dim=1)
        # st()
        frustum_mask = in_frustum_mask_dict['in_frustum_mask_final'][
            0].squeeze(2).repeat_interleave(3, dim=1)

        # Construct comparison image.
        context_img = inverse_normalize(
            batch["context"]["image"][0])  # [-1,1] -> [0,1]

        # rgb_gt = inverse_normalize(rgb_gt[0])  # normalize to [0,1]
        rgb_gt = batch['target']['image'][0]  # normalize to [0,1]
        depth_gt = [
            colorize_depth(depth, cmap='Spectral_r') / 255.0
            for depth in batch['target']['depthmap'][0].cpu().numpy()
        ]
        depth_gt = torch.from_numpy(
            rearrange(np.stack(depth_gt), 'v h w c -> v c h w'))

        # gs predictions, make then all in [0,1]
        # rgb_pred = inverse_normalize(gs_output.color[0])
        rgb_pred = gs_output.color[0]
        # st()
        depth_pred = [
            colorize_depth(depth, cmap='Spectral_r') / 255.0
            for depth in gs_output.depth[0].detach().cpu().numpy()
        ]
        depth_pred = torch.from_numpy(
            rearrange(np.stack(depth_pred), 'v h w c -> v c h w'))

        comparison = hcat(
            add_label(vcat(*context_img), "RGB (Context)"),
            add_label(vcat(*rgb_gt), "RGB (Ground Truth)"),
            add_label(vcat(*depth_gt), "Depth (Ground Truth)"),
            add_label(vcat(*rgb_pred), "RGB (Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
            add_label(vcat(*frustum_mask), "FrustumMask (Ground Truth)"))

        # st()  # check the caption, dataset-label
        if wandb_log:
            self.logger.log_image(
                "comparison",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"]['dataset'][0],
            )
        else:
            torchvision.utils.save_image(comparison,
                                         os.path.join(
                                             self.logger.save_dir,
                                             f'{self.global_step}.jpg'),
                                         normalize=True,
                                         value_range=(0, 1))

        return comparison
        '''
        # Render projections and construct projection image.
        # These are disabled for now, since RE10k scenes are effectively unbounded.

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
        '''

    @rank_zero_only
    @torch.no_grad()
    def save_pts3d(self,
                   pts3d,
                   output_path,
                   batch_idx,
                   output_suffix,
                   rgb=None,
                   verbose=False):
        assert isinstance(pts3d, torch.Tensor)
        assert pts3d.shape[0] == 1
        pts3d = pts3d.reshape(-1, 3).cpu().numpy()

        pcd = trimesh.Trimesh(vertices=pts3d, vertex_colors=rgb)

        pcd_output_path = output_path / f'step_{self.global_step}-batch_{batch_idx}-{output_suffix}.obj'
        pcd.export(pcd_output_path, 'obj')
        if verbose:
            print('pcd saved to ', pcd_output_path)

    def _to_pixelaligned_gs(self, gaussians, V):
        srf, spp = 1, 1
        pixel_aligned_gaussians = Gaussians(
            rearrange(gaussians.means,
                      "b (v r srf spp) xyz -> b v (r srf spp) xyz",
                      v=V,
                      srf=srf,
                      spp=spp),
            rearrange(
                gaussians.covariances,
                "b (v r srf spp) i j -> b v (r srf spp) i j",
                v=V,
                srf=srf,
                spp=spp,
            ),
            rearrange(gaussians.harmonics,
                      "b (v r srf spp) c d_sh -> b v (r srf spp) c d_sh",
                      v=V,
                      srf=srf,
                      spp=spp),
            rearrange(gaussians.opacities,
                      "b (v r srf spp) -> b v (r srf spp)",
                      v=V,
                      srf=srf,
                      spp=spp),
        )
        return pixel_aligned_gaussians

    def _merge_pixelaligned_gs(self, gaussians, V_indices):
        # merge the first V pixelaligned gs to a shared scene representation
        srf, spp, V = 1, 1, len(V_indices)
        # st()
        gaussians = Gaussians(
            rearrange(gaussians.means[:, V_indices],
                      "b v (r srf spp) xyz -> b (v r srf spp) xyz ",
                      v=V,
                      srf=srf,
                      spp=spp),
            rearrange(
                gaussians.covariances[:, V_indices],
                "b v (r srf spp) i j -> b (v r srf spp) i j",
                v=V,
                srf=srf,
                spp=spp,
            ),
            rearrange(gaussians.harmonics[:, V_indices],
                      "b v (r srf spp) c d_sh -> b (v r srf spp) c d_sh",
                      v=V,
                      srf=srf,
                      spp=spp),
            rearrange(gaussians.opacities[:, V_indices],
                      "b v (r srf spp) -> b (v r srf spp)",
                      v=V,
                      srf=srf,
                      spp=spp),
        )
        return gaussians

    @rank_zero_only
    def validation_step(self, views, batch_idx, dataloader_idx=0):
        # st()

        return

        # V = len(views)
        batch, V = self._preprocess_views(views)  # type: ignore
        b, _, _, h, w = batch["context"]["image"].shape  # type: ignore

        output_path = Path(self.logger.save_dir)  # type: ignore

        assert b == 1

        visualization_dump = {}
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )

        # st()

        if V == 2:  # for compat issue, use mast3r default loss class
            pred1, pred2, gt1, gt2 = self._prep_pred_and_gt_paired(
                views, visualization_dump)

            # dust3r_loss, dust3r_loss_details = self.dust3r_criterion(
            #     gt1, gt2, pred1, pred2)

        else:  # cut3r/must3r-like loss class

            all_preds, all_gts = self._prep_pred_and_gt_mv(
                views, visualization_dump)

            # dust3r_loss, dust3r_loss_details = self.must3r_mv_criterion(
            #     all_preds, all_gts)

        if self.global_rank == 0:
            print(f"validation step {self.global_step}; "
                  f"scene = {batch['scene']['label'][0]}; "
                  f"context = {batch['context']['index']['instance']}")

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
                all_gt_pts = [all_gts[v]['pts3d'] for v in range(0, V)]

                all_rgb = np.concatenate(all_rgb, 0)

                # st()

                all_pred_pts = rearrange(torch.cat(all_pred_pts, 0),
                                         'b h w c -> 1 (b h w) c')
                all_gt_pts = rearrange(
                    torch.cat(all_gt_pts,
                              0), 'b h w c -> 1 (b h w) c')  # for compat issue

                save_pts3d(all_pred_pts, 'pred-all', v=True, rgb=all_rgb)
                save_pts3d(all_gt_pts, 'gt-all', v=True, rgb=all_rgb)

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

    def training_step(self, views, batch_idx):  # views is batch here

        # ! debugging
        # print('debugging validation, ignore training step')
        # return torch.tensor(0.0, device=self.device, requires_grad=True)

        if batch_idx % 1000 == 0:
            torch.cuda.empty_cache()  # avoid oom

        batch, V, context_indices_for_render = self._preprocess_views(
            views, add_target=True, for_nvs=True, return_ctx_indices=True
        )  # for views target stuffs, no need to merge

        # ! hard coded, normalize img form [-1,1] to [0,1]
        # to align with pre-trained ckpt and 3dgs rendering range

        # ! disable if not 3dgs rendering
        # batch['context']['image'] = batch['context']['image'] * 0.5 + 0.5
        # batch['target']['image'] = batch['target']['image'] * 0.5 + 0.5

        # V = len(views)

        # batch = self.data_shim(batch)  # ?

        B, _, _, h, w = batch["context"]["image"].shape

        # ! catch some failed train samples
        # try:
        if True:
            # Run the model.
            # visualization_dump = None
            # if self.distiller is not None:
            visualization_dump = {}  # ! since imposing depth / pts3d loss here

            gaussians = self.encoder(
                batch["context"],  # type: ignore
                self.global_step,
                visualization_dump=visualization_dump)
            total_loss = 0

            # st()
            # ! debug gs renderer, by assigning gt xyz as gs means. done.
            # gaussians.means[0] = rearrange(
            #     torch.stack(all_pts3d_to_save),
            #     'v h w c -> 1 (v h w) c')  # only replace batch 1
            # st()

            # ! dust3r supervision to regularize the pts3d
            if self.global_step <= self.train_cfg.distill_max_steps:

                # if V == 2:  # for compat issue, use mast3r default loss class
                #     pred1, pred2, gt1, gt2 = self._prep_pred_and_gt_paired(
                #         views, visualization_dump)

                #     dust3r_loss, dust3r_loss_details = self.dust3r_criterion(
                #         gt1, gt2, pred1, pred2)  # ! no pose loss here

                # else:  # cut3r/must3r-like loss class

                all_preds, all_gts = self._prep_pred_and_gt_mv(
                    views, visualization_dump)

                dust3r_loss, dust3r_loss_details = self.must3r_mv_criterion(
                    all_gts, all_preds)

            else:
                dust3r_loss_details = {}

            # st()
            scale_factors = {
                k: dust3r_loss_details.pop(k)
                for k in ['norm_factor_cross_gt', 'norm_factor_cross_pr']
            }

            with torch.amp.autocast(device_type="cuda", enabled=False):

                # ! supervise novel-views. what about self?
                # half_gs_number = V // 2 * h * w
                # gaussians_as_ctx = Gaussians(
                #     gaussians.means[:, :half_gs_number],
                #     gaussians.covariances[:, :half_gs_number],
                #     gaussians.harmonics[:, :half_gs_number],
                #     gaussians.opacities[:, :half_gs_number],
                # )
                # ! get the gaussians for certain views
                gaussians_pixelaligned = self._to_pixelaligned_gs(gaussians,
                                                                  V=V)
                gaussians_as_ctx = self._merge_pixelaligned_gs(
                    gaussians_pixelaligned,
                    V_indices=context_indices_for_render)

                # V_indices=list(range(V)))  # rendering all and compare
                # gaussians_pixelaligned, V_indices=list(range(V//2)))

                # ! only use the first v/2 gaussians for supervision.

                # st()
                if self.analysis_flag:
                    with torch.no_grad():
                        for v_idx in range(V):  # visualize batch@1
                            self.save_pts3d(
                                gaussians_pixelaligned.means[0:1, v_idx],
                                output_path=Path(
                                    self.logger.save_dir),  # type: ignore
                                batch_idx=batch_idx,
                                output_suffix=f'{v_idx}',
                                rgb=(batch["context"]["image"][
                                    0, v_idx].detach().cpu().reshape(
                                        3, -1).permute(1, 0).numpy() * 127.5 +
                                     127.5).astype(np.uint8),
                                verbose=True)

                    export_ply(
                        gaussians.means[0], visualization_dump["scales"][0],
                        visualization_dump["rotations"][0],
                        gaussians.harmonics[0], gaussians.opacities[0],
                        Path(self.logger.save_dir) /
                        f'step_{self.global_step}-batch_{batch_idx}-{v_idx}.ply'
                    )

                # st()
                output = self.decoder.forward(
                    # gaussians,  # means: 8 * 196608 (65536 * 3) * 3
                    gaussians_as_ctx,  # means: 8 * 196608 (65536 * 3) * 3
                    batch["target"]
                    ["extrinsics"],  # ! camera_pose (requires inverse)
                    batch["target"]["intrinsics"],  # camera_intrinsics
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=self.train_cfg.depth_mode,  # unused.
                    scale_factors=scale_factors,
                )

            pass

            # use the first half as the ctx
            in_frustum_mask_dict = get_frustum_mask(
                points_3d=batch['target']['pts3d'],
                c2w_ctx=batch['context']['camera_pose'][:, :V // 2],
                intrinsics_ctx=batch['context']['intrinsics'][:, :V // 2],
                tgt_valid_mask=batch['target']['valid_mask'],
            )

            # st()  # check in_frustum_mask_final
            valid_mask = in_frustum_mask_dict['in_frustum_mask_final']
            # target_gt = batch["target"]["image"] * in_frustum_mask_dict[
            #     'in_frustum_mask_final']  # B, V, 3, 256, 256

            # move gt and target to [-1,1] for lpips loss
            # target_gt = batch["target"]["image"]
            # output.color = output.color * 2 - 1  # for lpips supervision

            # [-1,1] range for lpips supervision
            batch["target"]["image"] = batch["target"]["image"] * 0.5 + 0.5
            # 0.5) * valid_mask
            # target_gt = batch["target"]["image"]
            # output.color = (output.color * 2 - 1) * valid_mask
            # output.color = output.color * valid_mask

            direct_gt_mask = False
            # direct_gt_mask = True
            if direct_gt_mask:
                batch['target'][
                    'image'] = batch['target']['image'] * valid_mask
                output.color = output.color * valid_mask
                output.depth = output.depth * valid_mask.squeeze(2)

            # local log code, leave here for compat issue.
            if self.global_step % 250 == 0 or self.analysis_flag:
                with torch.no_grad():
                    # st()
                    # visualize the mask also

                    _ = self.log_step(
                        batch,
                        # target_gt,
                        output,
                        in_frustum_mask_dict,
                        wandb_log=False)

            # Compute psnr only on the foreground.
            psnr_probabilistic = compute_psnr(
                rearrange(batch['target']['image'] * valid_mask,
                          "b v c h w -> (b v) c h w"),
                rearrange(output.color * valid_mask,
                          "b v c h w -> (b v) c h w"),
            ).mean()
            self.log("train/psnr_probabilistic", psnr_probabilistic)

            # Compute and log loss.
            # ! ignore the rendering loss
            render_loss_log = {}
            for loss_fn in self.losses:
                render_loss = loss_fn.forward(
                    output,
                    batch,
                    # gaussians,
                    gaussians_as_ctx,
                    self.global_step,
                    #   mask=None)  # !
                    # mask=None if direct_gt_mask else valid_mask)  # !
                    mask=valid_mask)
                self.log(f"loss/{loss_fn.name}", render_loss)
                render_loss_log.update({f"{loss_fn.name}": render_loss})
                total_loss = total_loss + render_loss  # ! not imposing the loss for now

            if self.sparcity_loss:
                # st()
                density_loss = gaussians.opacities.mean(
                )  # encourage to turn it off
                scale_loss = torch.clamp(
                    visualization_dump['scales'] -
                    scale_factors['norm_factor_cross_pr'].squeeze(-1) * 0.005,
                    min=0).mean()
            total_loss = total_loss + density_loss * 0.1 + scale_loss * 2
            self.log(f"loss/sparse/density", density_loss)
            self.log(f"loss/sparse/scale", scale_loss)

            # compute depth loss if needed
            if self.enable_depth_loss:
                for depth_loss_fn in self.depth_losses:
                    depth_loss = depth_loss_fn.forward(
                        rearrange(output.depth, 'b v h w -> (b v) h w'),
                        rearrange(batch['target']['depthmap'],
                                  'b v h w -> (b v) h w'),
                        mask=rearrange(valid_mask,
                                       'b v 1 h w -> (b v) h w'))  # !
                    #   mask=valid_mask)
                    self.log(f"loss/{depth_loss_fn.name}", depth_loss)
                    render_loss_log.update(
                        {f"{depth_loss_fn.name}": depth_loss})
                    total_loss = total_loss + depth_loss  # ! not imposing the loss for now

            # else:
            # render_loss = None
            # '''

            self.log("loss/Reg3d/loss_sum", dust3r_loss)
            for k, v in dust3r_loss_details.items():  # log details
                self.log(f"loss/Reg3d/{k}", v)

            # total_loss = total_loss + dust3r_loss
            # total_loss = total_loss * 0.25 + dust3r_loss
            total_loss = total_loss * 4 + dust3r_loss  # ! tune the weight
            self.log("loss/total", total_loss)

            if (self.global_rank == 0 and self.global_step %
                    self.train_cfg.print_log_every_n_steps == 0):
                log_to_print = f"train step {self.global_step}; total_loss = {total_loss:.6f}; psnr = {psnr_probabilistic:.6f}; dust3r_loss = {dust3r_loss:.6f}; lpips = {render_loss_log['lpips'].item():.6f}; mse = {render_loss_log['mse'].item():.6f}"
                if "pose_loss" in dust3r_loss_details:
                    log_to_print += f"; pose_loss= {dust3r_loss_details['pose_loss']:.6f}"
                if self.enable_depth_loss:
                    log_to_print += f"; GradL1 = {render_loss_log['GradL1'].item():.6f}"
                    log_to_print += f"; SSILoss = {render_loss_log['SSILoss'].item():.6f}"
                if self.sparcity_loss:
                    log_to_print += f"; density_loss = {density_loss.item():.6f}"
                    log_to_print += f"; scale_loss = {scale_loss.item():.6f}"
                print(log_to_print)

            self.log("info/global_step",
                     self.global_step)  # hack for ckpt monitor

            # Tell the data loader processes about the current step.
            if self.step_tracker is not None:
                self.step_tracker.set_step(self.global_step)

            return total_loss
