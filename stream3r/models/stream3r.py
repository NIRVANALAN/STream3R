# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, List
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from stream3r.dust3r.utils.misc import freeze_all_params
from stream3r.models.components.aggregator.streamaggregator import STreamAggregator
from stream3r.models.components.heads.camera_head import CameraHead
from stream3r.models.components.heads.dpt_head import DPTHead


class STream3R(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024, freeze="none"):
        super().__init__()

        self.aggregator = STreamAggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.camera_head = CameraHead(dim_in=2 * embed_dim)
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1")
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1")

        self.set_freeze(freeze)
    
    def set_freeze(self, freeze):
        self.freeze = freeze

        to_be_frozen = {
            "none": [],
            "encoder": [self.aggregator.patch_embed],
        }
        freeze_all_params(to_be_frozen[freeze])

    def forward(
        self,
        images: torch.Tensor,
        mode: str = "causal",
        aggregator_kv_cache_list: List[List[torch.Tensor]] = None,
        camera_head_kv_cache_list: List[List[List[torch.Tensor]]] = None,
    ):
        """
        Forward pass of the STream3R model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            mode (str): Global attention mode, could be either "causal", "window", "full"
            aggregator_kv_cache_list (List[List[torch.Tensor]]): List of cached key-value pairs for
                each global attention layer of the aggregator
            camera_head_kv_cache_list (List[List[List[torch.Tensor]]]): List of cached key-value pairs for 
                each iterations and each attention layer of the camera head

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization
        """
        if self.training:
            images = torch.stack([view["img"] for view in images], dim=1)
            images = (images + 1.) / 2.

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if aggregator_kv_cache_list is not None:
            aggregated_tokens_list, patch_start_idx, aggregator_kv_cache_list = self.aggregator(images, mode=mode, kv_cache_list=aggregator_kv_cache_list)
        else:
            aggregated_tokens_list, patch_start_idx = self.aggregator(images, mode=mode)

        predictions = {}

        with torch.autocast(device_type=next(self.parameters()).device.type, dtype=torch.float32):
            if self.camera_head is not None:
                if camera_head_kv_cache_list is not None:
                    pose_enc_list, camera_head_kv_cache_list = self.camera_head(aggregated_tokens_list, mode=mode, kv_cache_list=camera_head_kv_cache_list)
                else:
                    pose_enc_list = self.camera_head(aggregated_tokens_list, mode=mode)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                if self.training:
                    predictions["pose_enc_list"] = pose_enc_list

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

        if aggregator_kv_cache_list is not None:
            predictions["aggregator_kv_cache_list"] = aggregator_kv_cache_list
        
        if camera_head_kv_cache_list is not None:
            predictions["camera_head_kv_cache_list"] = camera_head_kv_cache_list

        if not self.training:
            predictions["images"] = images

        return predictions