import json
import os
import sys
from typing import Any

import math
from pytorch_lightning.utilities.types import STEP_OUTPUT

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..misc.cam_utils import camera_normalization, pose_auc, update_pose, get_pnp_pose

import csv
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningModule
from tabulate import tabulate

from pdb import set_trace as st

from ..loss.loss_ssim import ssim
from ..misc.image_io import load_image, save_image
from ..misc.utils import inverse_normalize, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from .evaluation_cfg import EvaluationCfg
from .metrics import compute_lpips, compute_psnr, compute_ssim, compute_pose_error
from ..model.types import Gaussians

from .pose_evaluator import PoseEvaluator

class PoseEvaluatorWindow(PoseEvaluator):

    def __init__(self, cfg: EvaluationCfg, encoder, decoder, losses) -> None:
        super().__init__(cfg, encoder, decoder, losses)


    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch) #  batch['context']['image'].shape torch.Size([1, 9, 3, 256, 256])

        # st()

        # set to eval
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["context"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # get overlap.
        overlap = batch["context"]["overlap"][0, 0]
        overlap_tag = get_overlap_tag(overlap)
        if overlap_tag == "ignore":
            return

        # runing encoder to obtain the 3DGS
        # input_images_view2 = batch["context"]["image"][:, 1:].clone()
        input_images_view2=rearrange(batch["context"]["image"][:, 1:].clone(),
            "b v ... -> (b v) 1 ...",
        )

        input_images_view2 = input_images_view2 * 0.5 + 0.5 # as the gt
        visualization_dump = {}
        gaussians = self.encoder( # means: 1, HxWxV, 3
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump, # returns in: 1, 9, 256, 256, 1, 3
        )
        V = batch["context"]["image"].shape[1]

        # reshape gaussians, v to b
        srf, spp = 1, 1
        pixel_aligned_gaussians = Gaussians(
            rearrange(
                gaussians.means.float(),
                "b (v r srf spp) xyz -> (b v) (r srf spp) xyz", v=V, srf=srf, spp=spp
            )[1:],
            rearrange(
                gaussians.covariances.float(),
                "b (v r srf spp) i j -> (b v) (r srf spp) i j", v=V, srf=srf, spp=spp, 
            )[1:],
            rearrange(
                gaussians.harmonics.float(),
                "b (v r srf spp) c d_sh -> (b v) (r srf spp) c d_sh", v=V, srf=srf, spp=spp
            )[1:],
            rearrange(
                gaussians.opacities.float(),
                "b (v r srf spp) -> (b v) (r srf spp)", v=V, srf=srf, spp=spp
            )[1:],
        )


        # retrieve the init pose via PnPRansac
        pnp_pose_list = []

        for v in range(1, V):
            pose_opt = get_pnp_pose(visualization_dump['means'][0, v].squeeze(),
                                    visualization_dump['opacities'][0, v].squeeze(),
                                    batch["context"]["intrinsics"][0, v], h, w)
            pnp_pose_list.append(pose_opt)

        stacked_pnp_pose = torch.stack(pnp_pose_list).to(self.device) # V-1, 4, 4

        # pose_opt = pose_opt.to(self.device) # shape: 4x4
        # pose_opt = batch["context"]["extrinsics"][0, 0].clone()  # initial pose as the first view: I

        with torch.set_grad_enabled(True):
            # cam_rot_delta = nn.Parameter(torch.zeros([b, 1, 3], requires_grad=True, device=self.device))
            # cam_trans_delta = nn.Parameter(torch.zeros([b, 1, 3], requires_grad=True, device=self.device))
            cam_rot_delta = nn.Parameter(torch.zeros([V-1, 1, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([V-1, 1, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": 0.005,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": 0.005,
                }
            )

            pose_optimizer = torch.optim.Adam(opt_params)

            number_steps = 200
            # extrinsics = pose_opt.unsqueeze(0).unsqueeze(0)  # initial pose use pose_opt
            extrinsics = stacked_pnp_pose.unsqueeze(1) # V-1, 1, 4, 4
            for i in range(number_steps):
                pose_optimizer.zero_grad()

                # ! only render the novel views (pts3d_in_other_view) and supervise the L_rec
                output = self.decoder.forward(
                    pixel_aligned_gaussians,
                    extrinsics,
                    batch["context"]["intrinsics"][:, 1:],
                    batch["context"]["near"][:, 1:],
                    batch["context"]["far"][:, 1:],
                    (h, w),
                    cam_rot_delta=cam_rot_delta,
                    cam_trans_delta=cam_trans_delta,
                )

                # Compute and log loss.
                batch["target"]["image"] = input_images_view2
                total_loss = 0
                for loss_fn in self.losses:
                    loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                    total_loss = total_loss + loss

                # add ssim structure loss
                ssim_, _, _, structure = ssim(rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
                                      rearrange(output.color, "b v c h w -> (b v) c h w"),
                                      size_average=True, data_range=1.0, retrun_seprate=True, win_size=11)
                ssim_loss = (1 - structure) * 1.0
                total_loss = total_loss + ssim_loss

                # backpropagate
                # print(f"Step {i} - Loss: {total_loss.item()}")
                total_loss.backward()
                with torch.no_grad():
                    pose_optimizer.step()
                    new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                )
                    cam_rot_delta.data.fill_(0)
                    cam_trans_delta.data.fill_(0)

                    # extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=1)
                    extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=V-1, v=1)
                    # st()
                    pass

            # eval pose
            # st()
            # gt_pose = batch["context"]["extrinsics"][0, 1]
            gt_pose = batch["context"]["extrinsics"][0, 1:] # V-1, 4, 4
            eval_pose = extrinsics[:, 0] # V-1, 4, 4
            for v in range(V-1):
                error_t, error_t_scale, error_R = compute_pose_error(gt_pose[v], eval_pose[v])
                error_pose = torch.max(error_t, error_R)  # find the max error

                all_metrics = {
                    "e_t_ours": error_t,
                    "e_R_ours": error_R,
                    "e_pose_ours": error_pose,
                }

                self.print_preview_metrics(all_metrics, overlap_tag) # auto running metrics avg

            return 0