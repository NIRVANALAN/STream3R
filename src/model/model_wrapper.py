from dataclasses import dataclass
import os
import torchvision
import random
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
from ..loss.loss_point import Regr3D
from ..loss.loss_ssim import ssim
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose, get_pnp_pose
import numpy as np
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

from pdb import set_trace as st


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_video: bool
    save_compare: bool
    save_gaussian: bool


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    distiller: str
    distill_max_steps: int
    dust3r_splat_joint_training: bool
    noposplat_training: bool


@runtime_checkable
class TrajectoryFn(Protocol):

    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
            Float[Tensor, "batch view 4 4"],  # extrinsics
            Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
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
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        self.distiller = distiller
        self.distiller_loss = None
        if self.distiller is not None:
            convert_to_buffer(self.distiller, persistent=False)
            self.distiller_loss = Regr3D()

        # This is used for testing.
        self.benchmarker = Benchmarker()

        # freeze unused params to enable ddp
        self.encoder.backbone.mask_token.requires_grad_(False)

        for name, param in self.named_parameters():
            if 'refinenet4.resConfUnit1' in name:
                param.requires_grad_(False)
            if freeze_gs_head:
                if 'gaussian_param_head' in name:
                    param.requires_grad_(False)
                # print('freeze', name)

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)
    #     st()

    # encoder.backbone.mask_token
    # encoder.downstream_head1.dpt.scratch.refinenet4.resConfUnit1.conv1.weight
    # encoder.downstream_head1.dpt.scratch.refinenet4.resConfUnit1.conv1.bias
    # encoder.downstream_head1.dpt.scratch.refinenet4.resConfUnit1.conv2.weight
    # encoder.downstream_head1.dpt.scratch.refinenet4.resConfUnit1.conv2.bias
    # encoder.downstream_head2.dpt.scratch.refinenet4.resConfUnit1.conv1.weight
    # encoder.downstream_head2.dpt.scratch.refinenet4.resConfUnit1.conv1.bias
    # encoder.downstream_head2.dpt.scratch.refinenet4.resConfUnit1.conv2.weight
    # encoder.downstream_head2.dpt.scratch.refinenet4.resConfUnit1.conv2.bias
    # encoder.gaussian_param_head.dpt.scratch.refinenet4.resConfUnit1.conv1.weight
    # encoder.gaussian_param_head.dpt.scratch.refinenet4.resConfUnit1.conv1.bias
    # encoder.gaussian_param_head.dpt.scratch.refinenet4.resConfUnit1.conv2.weight
    # encoder.gaussian_param_head.dpt.scratch.refinenet4.resConfUnit1.conv2.bias
    # encoder.gaussian_param_head2.dpt.scratch.refinenet4.resConfUnit1.conv1.weight
    # encoder.gaussian_param_head2.dpt.scratch.refinenet4.resConfUnit1.conv1.bias
    # encoder.gaussian_param_head2.dpt.scratch.refinenet4.resConfUnit1.conv2.weight
    # encoder.gaussian_param_head2.dpt.scratch.refinenet4.resConfUnit1.conv2.bias
    # pass

    def schedule(self, initial: int, final: int) -> int:
        warmup_step = 15000
        if self.global_step < warmup_step:
            return initial 
        fraction = (self.global_step-warmup_step) / (warmup_step) # ! hard coded, warm up steps
        return min(initial + int((final - initial) * fraction), final)

    def training_step(self, batch, batch_idx):
        # ! debugging
        # print('debugging validation, ignore training step')
        # return torch.tensor(0.0, device=self.device, requires_grad=True)

        if batch_idx % 500 == 0:
            torch.cuda.empty_cache()  # avoid oom

        # combine batch from different dataloaders
        if isinstance(batch, list):
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
                                    batch_combined[k][kk], batch_per_dl[k][kk]
                                ],
                                                                  dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined

        # st()

        batch: BatchedExample = self.data_shim(batch)  # ?
        _, _, _, h, w = batch["target"]["image"].shape
        # (Pdb) p batch['context'].keys()
        # dict_keys(['extrinsics', 'intrinsics', 'image', 'near', 'far', 'index', 'overlap'])
        # (Pdb) p batch['context']['image'].shape
        # torch.Size([8, 3, 3, 256, 256])

        # ! randomly drop views here.
        loaded_num_context_views = batch['context']['image'].shape[1]

        num_context_views = self.schedule(
            2, loaded_num_context_views,
        )
        # num_context_views = 4  # for debug

        # if num_context_views > 2 and self.global_step % 5 == 0:
        #     num_context_views = 3 # ! force good registration here

        # st()
        if num_context_views > 2 and self.global_step % 5 == 0:
            # ! failed below. only v=2 works
            # num_context_views = 3 # ! fails

            num_context_views = 2 # ! force good registration here

        else:
            if num_context_views > 2 and self.global_step % 3 == 0:
                num_context_views = random.choice(list(range(3, loaded_num_context_views-1)))

        # ! drop middle views, try view=2
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
                k: v[:, view_indices] if v.shape[1] == loaded_num_context_views else v # 'overlap' has shape B, 1
                for k, v in batch['context'].items()  # type: ignore
            }
            batch['context'] = masked_context  # type: ignore

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
                    normalize_pts=batch['normalize_pts'],
                    scene_scale=batch['scene_scale'],
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

            if self.global_rank and self.global_step % 250 == 0:
                vis_pred_img = rearrange(output.color[0:1], "b v c h w -> b c h (v w)") * 2 - 1
                vis_gt_img = rearrange(target_gt[0:1], "b v c h w -> b c h (v w)")
                vis_pred = torch.cat([vis_gt_img, vis_pred_img], dim=-2)
                # torchvision.utils.save_image(vis_pred, os.path.join(self.logger.save_dir, f'{self.global_step}.jpg'), normalize=True, value_range=(-1, 1))
                torchvision.utils.save_image(vis_pred, os.path.join(self.logger.save_dir, f'{self.global_step}.jpg'), normalize=True, value_range=(0, 1))
                print(f"image of step {self.global_step} saved to {self.logger.save_dir}")

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
            if self.distiller is not None and self.global_step <= self.train_cfg.distill_max_steps:
                with torch.no_grad():
                    pseudo_gt1, pseudo_gt2 = self.distiller(
                        batch["context"], False)
                distillation_loss = self.distiller_loss(
                    pseudo_gt1['pts3d'],
                    pseudo_gt2['pts3d'],
                    visualization_dump['means'][:, 0].squeeze(-2),
                    visualization_dump['means'][:, 1].squeeze(-2),
                    pseudo_gt1['conf'],
                    pseudo_gt2['conf'],
                    disable_view1=False) * 0.1
                self.log("loss/distillation_loss", distillation_loss)
                total_loss = total_loss + distillation_loss

            self.log("loss/total", total_loss)

            if (self.global_rank == 0 and self.global_step %
                    self.train_cfg.print_log_every_n_steps == 0):
                print(f"train step {self.global_step}; "
                      f"scene = {[x[:20] for x in batch['scene']]}; "
                      f"context = {batch['context']['index'].tolist()}; "
                      f"loss = {total_loss:.6f}")
            self.log("info/global_step",
                     self.global_step)  # hack for ckpt monitor

            # Tell the data loader processes about the current step.
            if self.step_tracker is not None:
                self.step_tracker.set_step(self.global_step)

            return total_loss

        # except Exception as e:
        #     print(f"Skipping batch {batch['scene']} due to error: {e}")

        #     with open(f'datasets/train_steps_skipped.txt', 'a+') as f:
        #         # bad_examples=[item.strip() for item in f.readlines() ] # to read
        #         f.writelines([item+ ' ' + str(e) + '\n' for item in batch['scene']])

        #     # Return a zero loss (ensure it has `requires_grad=True` to allow backprop)
        #     return torch.tensor(0.0, device=self.device, requires_grad=True)
        #     # return None # this is not allowed in ddp

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # Render Gaussians.
        visualization_dump = {}
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                visualization_dump=visualization_dump,
            )

        # st()

        # align the target pose
        if self.test_cfg.align_pose:
            output = self.test_step_align(batch, gaussians)
        else:
            with self.benchmarker.time("decoder", num_calls=v):
                output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    normalize_pts=batch['normalize_pts'],
                    scene_scale=batch['scene_scale'],
                )

        # Save images.
        (scene, ) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        path.mkdir(exist_ok=True, parents=True)
        metrics_path = path / 'test_metrics.json'

        # compute scores
        if self.test_cfg.compute_scores:
            overlap = batch["context"]["overlap"][0]
            overlap_tag = get_overlap_tag(overlap)

            rgb_pred = output.color[0]
            rgb_gt = batch["target"]["image"][0]
            all_metrics = {
                f"lpips_ours": compute_lpips(rgb_gt, rgb_pred).mean(),
                f"ssim_ours": compute_ssim(rgb_gt, rgb_pred).mean(),
                f"psnr_ours": compute_psnr(rgb_gt, rgb_pred).mean(),
            }
            methods = ['ours']

            self.log_dict(all_metrics)
            self.print_preview_metrics(all_metrics,
                                       methods,
                                       overlap_tag=overlap_tag)

            # port metrics to local json
            with open(metrics_path, 'a+') as f:
                json.dump(
                    {scene: {
                        k: v.item()
                        for k, v in all_metrics.items()
                    }}, f)

        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0],
                                    output.color[0]):
                save_image(color, path / scene / f"color/{index:0>6}.jpg")

        if self.test_cfg.save_video:
            frame_str = "_".join(
                [str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in output.color[0]],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        if self.test_cfg.save_compare:
            # Construct comparison image.
            context_img = inverse_normalize(batch["context"]["image"][0])
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
            )
            save_image(comparison, path / f"{scene}.jpg")

        # export gaussians
        if self.test_cfg.save_gaussian:
            export_ply(gaussians.means[0], visualization_dump["scales"][0],
                       visualization_dump["rotations"][0],
                       gaussians.harmonics[0], gaussians.opacities[0],
                       path / f"{scene}.ply")

        print("saving output to: ", path)

    def test_step_align(self, batch, gaussians):
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(
                torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(
                torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append({
                "params": [cam_rot_delta],
                "lr": self.test_cfg.rot_opt_lr,
            })
            opt_params.append({
                "params": [cam_trans_delta],
                "lr": self.test_cfg.trans_opt_lr,
            })
            pose_optimizer = torch.optim.Adam(opt_params)

            extrinsics = batch["target"]["extrinsics"].clone()
            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()

                    output = self.decoder.forward(
                        gaussians,
                        extrinsics,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                        normalize_pts=batch['normalize_pts'],
                        scene_scale=batch['scene_scale'],
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians,
                                               self.global_step)
                        total_loss = total_loss + loss

                    total_loss.backward()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(
                            cam_rot_delta=rearrange(cam_rot_delta,
                                                    "b v i -> (b v) i"),
                            cam_trans_delta=rearrange(cam_trans_delta,
                                                      "b v i -> (b v) i"),
                            extrinsics=rearrange(extrinsics,
                                                 "b v i j -> (b v) i j"))
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic,
                                               "(b v) i j -> b v i j",
                                               b=b,
                                               v=v)

        # Render Gaussians.
        output = self.decoder.forward(
            gaussians,
            extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            normalize_pts=batch['normalize_pts'],
            scene_scale=batch['scene_scale'],
        )

        return output

    def on_test_end(self) -> None:
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name /
                              "benchmark.json")
        self.benchmarker.dump_memory(self.test_cfg.output_path / name /
                                     "peak_memory.json")
        self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return
        # st()

        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(f"validation step {self.global_step}; "
                  f"scene = {batch['scene']}; "
                  f"context = {batch['context']['index'].tolist()}")

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        visualization_dump = {}
        gaussians = self.encoder(
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump,
        )
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

        if self.distiller is not None:
            with torch.no_grad():
                pseudo_gt1, pseudo_gt2 = self.distiller(
                    batch["context"], False)
            depth1, depth2 = pseudo_gt1['pts3d'][...,
                                                 -1], pseudo_gt2['pts3d'][...,
                                                                          -1]
            conf1, conf2 = pseudo_gt1['conf'], pseudo_gt2['conf']
            depth_dust = torch.cat([depth1, depth2], dim=0)
            depth_dust = vis_depth_map(depth_dust)
            conf_dust = torch.cat([conf1, conf2], dim=0)
            conf_dust = confidence_map(conf_dust)
            dust_vis = torch.cat([depth_dust, conf_dust], dim=0)
            comparison = hcat(add_label(vcat(*dust_vis), "Context"),
                              comparison)

        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=self.global_step,
            caption=batch["scene"],
        )

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
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image("cameras", [prep_image(add_border(cameras))],
                              step=self.global_step)

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                    batch["context"], self.global_step).items():
                self.logger.log_image(k, [prep_image(image)],
                                      step=self.global_step)

        # Run video validation step.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            # self.render_video_wobble(batch)
            self.render_video_interpolation(batch)
            if self.train_cfg.extended_visualization:
                self.render_video_interpolation_exaggerated(batch)

        # st()
        # pass

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch,
                                         trajectory_fn,
                                         "wobble",
                                         num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2 else batch["target"]["extrinsics"][0, 0]
                    if v == 3 else batch["target"]["extrinsics"]
                    [0, -1]  # v=8, the first view is the middle view, instead.
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (batch["context"]["intrinsics"][0, 1]
                 if v == 2 else batch["target"]["intrinsics"][0, 0]),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self,
                                               batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (batch["context"]["extrinsics"][0, 1]
                 if v == 2 else batch["target"]["extrinsics"][0, 0]),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (batch["context"]["intrinsics"][0, 1]
                 if v == 2 else batch["target"]["intrinsics"][0, 0]),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians = self.encoder(batch["context"], self.global_step)

        t = torch.linspace(0,
                           1,
                           num_frames,
                           dtype=torch.float32,
                           device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.decoder.forward(gaussians, extrinsics, intrinsics, near,
                                      far, (h, w), "depth")
        images = [
            vcat(rgb, depth) for rgb, depth in zip(
                output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(
            torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.

        try:
            wandb.log(visualizations)  # requires to enable wandb
        except Exception:
            assert isinstance(self.logger, LocalLogger)

            for key, value in visualizations.items():

                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                video_path = str(dir / f"{name}_{self.global_step:0>6}.mp4")
                video = rearrange(
                    value.data[0],
                    'F rgb h w -> F h w rgb')  # h w 3 order to dump to video

                rgb_video_out = imageio.get_writer(video_path,
                                                   mode='I',
                                                   fps=24,
                                                   codec='libx264')

                for frame_idx in range(video.shape[0]):
                    rgb_video_out.append_data(video[frame_idx])

                rgb_video_out.close()
                print('video exported to {}'.format(video_path))

                # tensor = value._prepare_video(value.data)
                # clip = mpy.ImageSequenceClip(list(tensor), fps=30)
                # clip.write_videofile(
                #     str(dir / f"{self.global_step:0>6}.mp4"),
                #     # logger=None
                # )

    def print_preview_metrics(self,
                              metrics: dict[str, float | Tensor],
                              methods: list[str] | None = None,
                              overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {
                    k: ((s * v) + metrics[k]) / (s + 1)
                    for k, v in self.running_metrics_sub[overlap_tag].items()
                }
                self.running_metric_steps_sub[overlap_tag] += 1

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "gaussian_param_head" in name or "intrinsic_encoder" in name:
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
            },
            {
                "params":
                pretrained_params,
                "lr":
                self.optimizer_cfg.lr *
                self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts,
                                      lr=self.optimizer_cfg.lr,
                                      weight_decay=0.05,
                                      betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=get_cfg()["trainer"]["max_steps"],
            eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warm_up, lr_scheduler],
            milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
