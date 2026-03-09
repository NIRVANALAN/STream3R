import json
import os
import os.path as osp

from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
import numpy as np
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
from torch.distributed import get_rank, get_world_size

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetScannetPoseCfg(DatasetCfgCommon):
    name: str
    roots: list[Path]
    baseline_min: float
    baseline_max: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    relative_pose: bool
    skip_bad_shape: bool


@dataclass
class DatasetScannetPoseCfgWrapper:
    scannet_pose: DatasetScannetPoseCfg


class DatasetScannetPose(IterableDataset):
    cfg: DatasetScannetPoseCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetScannetPoseCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # Collect data.
        self.data_root = cfg.roots[0]
        pair_file = os.path.join(cfg.roots[0], "test.npz")
        data_pairs = np.load(pair_file)

        pairs, rel_pose = data_pairs["name"], data_pairs["rel_pose"]
        self.pairs = pairs  # scene name, image_file1, image_file2
        self.rel_pose = rel_pose

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # When testing, the data loaders alternate data.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.pairs = [
                pair
                for pair_index, pair in enumerate(self.pairs)
                if pair_index % worker_info.num_workers == worker_info.id
            ]
            self.rel_pose = [
                pose
                for pose_index, pose in enumerate(self.rel_pose)
                if pose_index % worker_info.num_workers == worker_info.id
            ]

        for scene, rel_pose in zip(self.pairs, self.rel_pose):

            scene_name = f"scene0{scene[0]}_00"
            im_A_path = os.path.join(
                self.data_root,
                "scans_test",
                scene_name,
                "color",
                f"{scene[2]}.jpg",
            )
            im_B_path = os.path.join(
                self.data_root,
                "scans_test",
                scene_name,
                "color",
                f"{scene[3]}.jpg",
            )
            context_images = [im_A_path, im_B_path]
            context_images = self.convert_images(context_images)

            h, w = context_images.shape[-2:]

            K = np.stack(
                [
                    np.array([float(i) for i in r.split()])
                    for r in open(
                    osp.join(
                        self.data_root,
                        "scans_test",
                        scene_name,
                        "intrinsic",
                        "intrinsic_color.txt",
                    ),
                    "r",
                )
                .read()
                .split("\n")
                    if r
                ]
            )

            # crop the image ro make the principal point in the center of the image
            def center_principal_point(image, cx, cy, h, w):
                cx = round(cx)
                cy = round(cy)

                # Calculate the desired center
                center_x, center_y = w // 2, h // 2

                # Calculate the shift needed
                shift_x = center_x - cx
                shift_y = center_y - cy

                # Calculate new image dimensions
                new_w = max(w, w - 2 * shift_x)
                new_h = max(h, h - 2 * shift_y)

                # convert to int
                new_w = round(new_w)
                new_h = round(new_h)

                # Create a new blank image
                new_image = torch.zeros((2, 3, new_h, new_w), dtype=torch.float32)

                # Calculate padding
                pad_left = max(0, -shift_x)
                pad_top = max(0, -shift_y)

                # Calculate the region of the original image to copy
                src_left = max(0, shift_x)
                src_top = max(0, shift_y)
                src_right = min(w, w + shift_x)
                src_bottom = min(h, h + shift_y)

                # Copy the shifted image to the new image
                new_image[:, :, pad_top:pad_top + src_bottom - src_top,
                pad_left:pad_left + src_right - src_left] = image[:, :, src_top:src_bottom, src_left:src_right]

                # Calculate new intrinsic parameters
                new_cx = new_w // 2
                new_cy = new_h // 2

                return new_image, new_cx, new_cy

            # tgt_cx, tgt_cy = w // 2, h // 2
            context_images, tgt_cx, tgt_cy = center_principal_point(
                context_images, K[0, 2], K[1, 2], h, w
            )
            K[0, 2] = tgt_cx
            K[1, 2] = tgt_cy

            h, w = context_images.shape[-2:]
            target_images = context_images.clone()

            pose1 = torch.eye(4)
            pose2 = torch.eye(4)
            pose2[:3, :4] = torch.tensor(rel_pose.reshape(3, 4)).to(torch.float32)
            pose2 = torch.inverse(pose2)
            extrinsics = torch.stack((pose1, pose2), dim=0)

            # normolize K
            K = K[:3, :3]
            K[0, :3] /= w
            K[1, :3] /= h

            intrinsics = torch.tensor(K, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)

            overlap = torch.tensor([0.5], dtype=torch.float32)
            scale = torch.tensor([1.0], dtype=torch.float32)
            context_indices = torch.tensor([0, 1], dtype=torch.int64)

            example = {
                "context": {
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "image": context_images,
                    "near": self.get_bound("near", 2),
                    "far": self.get_bound("far", 2),
                    "index": context_indices,
                    "overlap": overlap,
                    "scale": scale,
                },
                "target": {
                    "extrinsics": extrinsics,
                    "intrinsics": intrinsics,
                    "image": target_images,
                    "near": self.get_bound("near", 2),
                    "far": self.get_bound("far", 2),
                    "index": context_indices,
                },
                "scene": scene_name,
            }
            yield apply_crop_shim(example, tuple(self.cfg.input_image_shape))

    def convert_images(
        self,
        images,
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(image)
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage
