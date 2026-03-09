from dataclasses import dataclass
import random
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

# from .model_dust3r_wrapper import ModelWrapperMust3R
from .model_dust3r_with_render_wrapper import ModelWrapperMust3RWithRender
from src.dust3r.utils.geometry import geotrf, inv, get_frustum_mask

from src.loss.loss_zoedepth import GradL1Loss, ScaleAndShiftInvariantLoss


class ModelWrapperMust3RWithRenderHybrid(ModelWrapperMust3RWithRender):
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
                         freeze_gs_head=False)
                        #  freeze_gs_head=True)

        # freeze self branch and pose prediction branch
        # for name, param in self.named_parameters():
        #     if any(head_key in name for head_key in ['intrinsic_encoder']):
        #         param.requires_grad_(False)
        #         print('freeze: ', name)

        # print('!!!!!!! freezing gs head for debugging !!!!!!!!!!!!!!!!!!')
        # self.enable_dust3r_rendering_loss = False
        self.enable_dust3r_rendering_loss = True
        assert self.train_cfg.noposplat_training or self.train_cfg.dust3r_splat_joint_training

        # ! reproduced
        # if self.train_cfg.noposplat_training:
        #     # freeze self branch and pose prediction branch
        #     for name, param in self.named_parameters():
        #         if any(head_key in name for head_key in [
        #                 'encoder.downstream_head_self', 'encoder.camera_head',
        #                 'global_3d_feedback_Inj'
        #         ]):
        #             param.requires_grad_(False)
        #             print('freeze: ', name)

    def pixelsplat_training_step(self, batch, batch_idx):
        # ! debugging
        # print('debugging validation, ignore training step')
        # return torch.tensor(0.0, device=self.device, requires_grad=True)

        # combine batch from different dataloaders
        # st()
        if isinstance(batch, list):
            if len(batch) == 1:
                batch = batch[0]
            else:
                batch_combined = None
                for batch_per_dl in batch:
                    if batch_combined is None:
                        batch_combined = batch_per_dl
                    else:
                        for k in batch_combined.keys():
                            if isinstance(batch_combined[k], list):
                                batch_combined[k] += batch_per_dl[k]
                            elif isinstance(batch_combined[k], dict):
                                for kk in batch_combined[k].keys():
                                    batch_combined[k][kk] = torch.cat([
                                        batch_combined[k][kk],
                                        batch_per_dl[k][kk]
                                    ],
                                                                      dim=0)
                            else:
                                # st()
                                raise NotImplementedError
                batch = batch_combined

        # st()

        # ! normalize context images into [-1,1] !
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # ! randomly drop views here.
        loaded_num_context_views = batch['context']['image'].shape[1]

        num_context_views = self.schedule(
            2,
            loaded_num_context_views,
        )

        # ! set v=2 to encourage alignment
        # if num_context_views > 2 and self.global_step % 5 == 0:
        #     # ! failed below. only v=2 works
        #     # num_context_views = 3 # ! fails

        #     num_context_views = 2  # ! force good registration here

        # else:
        #     if num_context_views > 2 and self.global_step % 3 == 0:
        #         if loaded_num_context_views > 4:
        #             num_context_views = random.choice(
        #                 list(range(3, loaded_num_context_views - 1)))

        # ! drop middle views, try view=2
        '''turn off this schedule for now in joint training pipeline
        if num_context_views != loaded_num_context_views:
            # st()
            if num_context_views > 2:
                middle_views = np.random.choice(np.arange(
                    1, loaded_num_context_views - 1),
                                                size=num_context_views - 2,
                                                replace=False)
                middle_views = np.sort(middle_views)
                view_indices = [0] + middle_views.tolist() + [
                    loaded_num_context_views - 1
                ]
            else:
                view_indices = [0, loaded_num_context_views - 1]

            # st()
            masked_context = {
                k:
                v[:, view_indices] if v.shape[1] == loaded_num_context_views
                else v  # 'overlap' has shape B, 1
                for k, v in batch['context'].items()  # type: ignore
            }
            batch['context'] = masked_context  # type: ignore
        '''

        # ! catch some failed train samples
        # try:
        if True:
            # Run the model.
            visualization_dump = None
            if self.distiller is not None:
                visualization_dump = {}
            gaussians = self.encoder(batch["context"],
                                     self.global_step,
                                     visualization_dump=visualization_dump)
            # if self.global_step % 2 == 0:
            if True:
                output = self.decoder.forward(
                    gaussians,  # means: 8 * 196608 (65536 * 3) * 3
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=self.train_cfg.depth_mode,
                    # normalize_pts=batch['normalize_pts'],
                    # scene_scale=batch['scene_scale'],
                )
                target_gt = batch["target"]["image"]  # 8, 4, 3, 256, 256
            else:
                output = self.decoder.forward(
                    gaussians,
                    batch["context"]["extrinsics"],
                    batch["context"]["intrinsics"],
                    batch["context"]["near"],
                    batch["context"]["far"],
                    (h, w),
                    depth_mode=self.train_cfg.depth_mode,
                )
                target_gt = batch["context"]["image"]  # 8, 4, 3, 256, 256
                batch['target']['image'] = target_gt

            # ! training set log
            if self.global_rank and self.global_step % 2500 == 0:
                vis_pred_img = rearrange(output.color[0:1],
                                         "b v c h w -> b c h (v w)")
                vis_gt_img = rearrange(target_gt[0:1],
                                       "b v c h w -> b c h (v w)")
                vis_pred = torch.cat([vis_gt_img, vis_pred_img], dim=-2)
                # torchvision.utils.save_image(vis_pred, os.path.join(self.logger.save_dir, f'{self.global_step}.jpg'), normalize=True, value_range=(-1, 1))
                torchvision.utils.save_image(vis_pred,
                                             os.path.join(
                                                 self.logger.save_dir,
                                                 f'{self.global_step}.jpg'),
                                             normalize=True,
                                             value_range=(0, 1))
                print(
                    f"image of step {self.global_step} saved to {self.logger.save_dir}"
                )

            # Compute metrics.
            psnr_probabilistic = compute_psnr(
                rearrange(target_gt, "b v c h w -> (b v) c h w"),
                rearrange(output.color, "b v c h w -> (b v) c h w"),
            )
            self.log("train/psnr_probabilistic", psnr_probabilistic.mean())

            # Compute and log loss.
            total_loss = 0
            for loss_fn in self.losses:
                loss = loss_fn.forward(output, batch, gaussians,
                                       self.global_step)
                self.log(f"loss/{loss_fn.name}", loss)
                total_loss = total_loss + loss

            # distillation
            # if self.distiller is not None and self.global_step <= self.train_cfg.distill_max_steps:
            #     with torch.no_grad():
            #         pseudo_gt1, pseudo_gt2 = self.distiller(
            #             batch["context"], False)
            #     distillation_loss = self.distiller_loss(
            #         pseudo_gt1['pts3d'],
            #         pseudo_gt2['pts3d'],
            #         visualization_dump['means'][:, 0].squeeze(-2),
            #         visualization_dump['means'][:, 1].squeeze(-2),
            #         pseudo_gt1['conf'],
            #         pseudo_gt2['conf'],
            #         disable_view1=False) * 0.1
            #     self.log("loss/distillation_loss", distillation_loss)
            #     total_loss = total_loss + distillation_loss

            # self.log("loss/total", total_loss)

            if (self.global_rank == 0 and self.global_step %
                    self.train_cfg.print_log_every_n_steps == 0):
                print(f"train step {self.global_step}; "
                      f"scene = {[x[:20] for x in batch['scene']]}; "
                      f"context = {batch['context']['index'].tolist()}; "
                      f"loss = {total_loss:.6f}")

            # self.log("info/global_step",
            #          self.global_step)  # hack for ckpt monitor

            # Tell the data loader processes about the current step.

            return total_loss

    def dust3r_training_step(self, views, batch_idx):
        batch, V = self._preprocess_views(
            views, add_target=True,
            for_nvs=True)  # for views target stuffs, no need to merge
        B, _, _, h, w = batch["context"]["image"].shape

        # !  =================== dust3r loss =====================
        visualization_dump = {}
        gaussians = self.encoder(
            batch["context"],  # type: ignore
            self.global_step,
            visualization_dump=visualization_dump)
        total_loss = 0

        # ! dust3r supervision to regularize the pts3d
        # if self.global_step <= self.train_cfg.distill_max_steps:

        all_preds, all_gts = self._prep_pred_and_gt_mv(
            views, visualization_dump)

        dust3r_loss, dust3r_loss_details = self.must3r_mv_criterion(
            all_gts, all_preds)

        # else:
        #     dust3r_loss_details = {}

        # ! rendering loss from pixelsplat video datasets

        # ! do not add gs render for dust3r dataset for now

        scale_factors = {
            k: dust3r_loss_details.pop(k)
            for k in ['norm_factor_cross_gt', 'norm_factor_cross_pr']
        }

        total_render_loss = 0
        if self.enable_dust3r_rendering_loss:

            with torch.autocast(device_type="cuda", enabled=False):

                output = self.decoder.forward(
                    gaussians,  # means: 8 * 196608 (65536 * 3) * 3
                    batch["target"]
                    ["extrinsics"],  # ! camera_pose (requires inverse)
                    batch["target"]["intrinsics"],  # camera_intrinsics
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=self.train_cfg.depth_mode,  # unused.
                    scale_factors=scale_factors,
                )

            in_frustum_mask_dict = get_frustum_mask(
                points_3d=batch['target']['pts3d'],
                c2w_ctx=batch['context']['camera_pose'],
                intrinsics_ctx=batch['context']['intrinsics'],
                tgt_valid_mask=batch['target']['valid_mask'],
            )

            # st()  # check in_frustum_mask_final
            valid_mask = in_frustum_mask_dict['in_frustum_mask_final']

            # [0,1] range for lpips supervision (normalize=True)
            batch["target"]["image"] = batch["target"]["image"] * 0.5 + 0.5

            direct_gt_mask = False # ! amortized training
            # direct_gt_mask = True
            if direct_gt_mask:
                batch['target'][
                    'image'] = batch['target']['image'] * valid_mask
                output.color = output.color * valid_mask
                output.depth = output.depth * valid_mask.squeeze(2)

            # local log code, leave here for compat issue.
            if self.global_step % 500 == 0:
                with torch.no_grad():

                    _ = self.log_step(
                        batch,
                        # target_gt,
                        output,
                        in_frustum_mask_dict,
                        wandb_log=False)

            # st()

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
                render_loss = loss_fn.forward(output,
                                              batch,
                                              gaussians,
                                              self.global_step,
                                            #   mask=None)  # !
                mask=None if direct_gt_mask else valid_mask)  # !
                #   mask=valid_mask)
                self.log(f"loss/{loss_fn.name}", render_loss)
                render_loss_log.update({f"{loss_fn.name}": render_loss})
                total_render_loss = total_render_loss + render_loss  # ! not imposing the loss for now

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
                    total_render_loss = total_render_loss + depth_loss  # ! not imposing the loss for now

            # else:
            # render_loss = None
            # '''
        else:
            del scale_factors

        # log dust3r_loss
        self.log("loss/Reg3d/loss_sum", dust3r_loss)
        for k, v in dust3r_loss_details.items():  # log details
            self.log(f"loss/Reg3d/{k}", v)

        # total_loss = total_loss + dust3r_loss
        # total_loss = total_render_loss * 0.25 + dust3r_loss
        total_loss = total_render_loss * 0.25 + dust3r_loss
        # self.log("loss/total", total_loss)

        if (self.global_rank == 0
                and self.global_step % self.train_cfg.print_log_every_n_steps
                == 0):
            log_to_print = f"train step {self.global_step}; dust3r_loss = {dust3r_loss:.6f}; dust3r_step_total_loss = {total_loss:.6f}; "
            if "pose_loss" in dust3r_loss_details:
                log_to_print += f"; pose_loss= {dust3r_loss_details['pose_loss']:.6f}"
            if self.enable_dust3r_rendering_loss:
                log_to_print += f";psnr = {psnr_probabilistic:.6f}; lpips = {render_loss_log['lpips'].item():.6f}; mse = {render_loss_log['mse'].item():.6f}"
                if self.enable_depth_loss:
                    log_to_print += f"; GradL1 = {render_loss_log['GradL1'].item():.6f}"
                    log_to_print += f"; SSILoss = {render_loss_log['SSILoss'].item():.6f}"
            print(log_to_print)

        # Tell the data loader processes about the current step.
        # if self.step_tracker is not None:
        # self.step_tracker.set_step(self.global_step)

        return total_loss

    def training_step(self, batch, batch_idx):  # views is batch here
        # assert isinstance(batch, list)
        # ! move to combined_loader

        if self.train_cfg.noposplat_training:
            pixelsplat_batch = batch
            views = None
        else:
            pixelsplat_batch, views = batch[:-1], batch[
                -1]  # the last one is the dust3r dataloader

        pixelsplat_loss = torch.tensor(0., device=self.device)

        # if self.train_cfg.dust3r_splat_joint_training or self.train_cfg.noposplat_training:
        if self.train_cfg.noposplat_training:
            pixelsplat_loss = self.pixelsplat_training_step(
                pixelsplat_batch, batch_idx)

        if self.train_cfg.dust3r_splat_joint_training:
            dust3r_loss = self.dust3r_training_step(views, batch_idx)
        # else:
        #     dust3r_loss = torch.tensor(0., device=self.device)

        total_loss = dust3r_loss + pixelsplat_loss
        # total_loss = dust3r_loss

        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        self.log("loss/total", total_loss)
        self.log("loss/dust3r", dust3r_loss)
        self.log("loss/pixelsplat", pixelsplat_loss)

        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        del pixelsplat_batch, views, batch

        if batch_idx % 2500 == 0:
            torch.cuda.empty_cache()  # memory keep growing, why?

        return total_loss

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         # if param.grad is None:
    #         if param.grad is None and param.requires_grad:
    #             print(name)
    #     st()
    #     pass
