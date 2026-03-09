# https://github.com/Chrixtar/latentsplat/blob/main/src/dataset/dataset_co3d.py
# https://github.com/dcharatan/pixelsplat/issues/25
# https://github.com/dcharatan/pixelsplat/issues/75
# https://github.com/facebookresearch/co3d/issues/18

import json
import tqdm
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler, ViewSamplerEvaluation
from ..misc.cam_utils import camera_normalization

from .dataset_re10k import DatasetRE10kCfg, DatasetRE10k

@dataclass
class DatasetCO3DCfgWrapper:
    co3d: DatasetRE10kCfg


class DatasetCo3D(DatasetRE10k):

    planes: tuple = (0.5, 40.0) # ! https://github.com/Chrixtar/latentsplat/blob/67ae0f664a1fb443934b06ccc60d1b6b84ebbc93/config/dataset/co3d_teddybear.yaml#L9C9-L9C20

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__(cfg, stage, view_sampler)

    def _process_near_far(self, extrinsics):
        if self.planes is None:
            # https://github.com/facebookresearch/co3d/issues/18
            cam_loc = extrinsics[:, :3, 3]
            near = (cam_loc.norm(dim=-1) - 8).clamp_(0.5)  # to avoid -ve near
            far = cam_loc.norm(dim=-1) + 8
        else:
            near, far = self.planes
            num_views = extrinsics.shape[0]
            near = repeat(torch.tensor(near, dtype=torch.float32), "-> v", v=num_views)
            far = repeat(torch.tensor(far, dtype=torch.float32), "-> v", v=num_views)
        return near, far

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        # if self.stage in ("train", "val"):
        if self.stage in ["train"]:
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        # st()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        # for chunk_path in self.chunks[:2]: # ! for debug
        # print('debugging self.chunks[:2]')

        for chunk_idx, chunk_path in enumerate(self.chunks[:]):
        # for chunk_path in tqdm.tqdm(self.chunks):

            # ! save skipped samples offline for later use
            # if chunk_idx%5==0 and self.stage=='train' and self.dataset_name == 're10k':

            # if chunk_idx%5==0 and self.stage=='train' and ('co3d' in self.dataset_name):
            #     with open(f'datasets/{self.dataset_name}_skipped.txt', 'w') as f:
            #         f.writelines([item+'\n' for item in self.skipped_instances])

            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in ("train", "val"):
                chunk = self.shuffle(chunk) # bs=16 here.

            # for example in chunk: # load a scene
            for idx, example in enumerate(chunk): # load a scene
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]

                if scene in self.skipped_instances: # avoid duplicated data loading
                    continue
             
                try:
                    context_indices, target_indices, overlap = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                    # st()

                except ValueError as e:
                    # Skip because the example doesn't have enough frames.
                    print('ValueError: ', e)
                    self.skipped_instances.add(scene)
                    continue

                # st() # randomly add K views to the taget?
                # target_indices_list += context_indices[:2]

                # available_indices = torch.arange(len(context_indices), dtype=torch.long)
                # shuffled_indices = torch.randperm(len(available_indices))
                # self_view_for_rec = context_indices[shuffled_indices[:len(target_indices)]]
                # target_indices += self_view_for_rec 

            
                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    self.skipped_instances.add(scene)
                    continue

                # Load the images.
                try:
                    target_images = [ example["images"][index.item()] for index in target_indices ]
                    target_images = self.convert_images(target_images)
                    context_images = [ example["images"][index.item()] for index in context_indices ]
                    context_images = self.convert_images(context_images)
                except IndexError:
                    self.skipped_instances.add(scene)
                    continue
                except RuntimeError as e:
                    # print(e)
                    self.skipped_instances.add(scene)
                    continue
                except OSError:
                    print(f"Skipped bad example {example['key']}.")  # DL3DV-Full have some bad images
                    # st()
                    self.skipped_instances.add(scene)
                    continue

                # Skip the example if the images don't have the right shape.
                # st()
                # if self.cfg.skip_bad_shape:
                #     if 'co3d' in self.dataset_name:
                #         invalid_sample = min(context_images.shape[2:]) < 256 # skip those images, or src/dataset/shims/crop_shim.py", line 61, in rescale_and_crop

                    # # if True:
                    # else: # ! skip samples based on context_indices
                    #     context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
                    #     target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
                    #     # indices_valid = bool(len(context_indices) != 9 or len(target_indices) != 4 )
                    #     indices_valid = bool(len(context_indices) != 8 or len(target_indices) != 4 )
                    #     # indices_valid = bool(len(context_indices) != 2 or len(target_indices) != 4 )
                    #     # indices_valid = bool(len(context_indices) != 4 or len(target_indices) != 4 )
                    #     # indices_valid = bool(len(context_indices) != 4 or len(target_indices) != 4 )
                    #     invalid_sample = (context_image_invalid or target_image_invalid or indices_valid )
                    #     # invalid_sample = (context_image_invalid or target_image_invalid)

                        # if 'co3d' in self.datset_name:
                        # invalid_sample = min(context_images.shape[2:]) < 256 # skip those images, or src/dataset/shims/crop_shim.py", line 61, in rescale_and_crop

                # if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    # if (invalid_sample):
                    #     print(
                    #         f"Skipped bad example {example['key']}. Context shape was "
                    #         f"{context_images.shape} and target shape was "
                    #         f"{target_images.shape}."
                    #     )
                    #     # st()
                    #     self.skipped_instances.add(scene)
                    #     continue

                # Resize the world to make the baseline 1.
                context_extrinsics = extrinsics[context_indices]
                # ! co3d already has fixed scale in [-8, +8] sphere
                # if self.cfg.make_baseline_1: # True
                #     a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
                #     scale = (a - b).norm()
                #     if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                #         print(
                #             f"Skipped {scene} because of baseline out of range: "
                #             f"{scale:.6f}"
                #         )
                #         self.skipped_instances.add(scene)
                #         continue
                #     extrinsics[:, :3, 3] /= scale
                # else:
                #     scale = 1

                context_extrinsics = extrinsics[context_indices]
                target_extrinsics = extrinsics[target_indices]

                # Load Near and Far
                context_near, context_far = self._process_near_far(context_extrinsics)
                target_near, target_far = self._process_near_far(target_extrinsics)

                if self.cfg.relative_pose: # True
                    extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)
                
                # st()

                example = {
                    "context": {
                        "extrinsics": context_extrinsics,
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        # "near": self.get_bound("near", len(context_indices)) / scale,
                        # "far": self.get_bound("far", len(context_indices)) / scale,
                        "near": context_near,
                        "far": context_far,
                        "index": context_indices,
                        "overlap": overlap,
                    },
                    "target": {
                        "extrinsics": target_extrinsics,
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        # "near": self.get_bound("near", len(target_indices)) / scale,
                        # "far": self.get_bound("far", len(target_indices)) / scale,
                        "near": target_near,
                        "far": target_far,
                        "index": target_indices,
                    },
                    "scene": scene,
                    "normalize_pts": True,
                    "scene_scale": 8, 
                }

                # if extrinsics[target_indices].shape[0]==1:
                #     st()

                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)

                crop_shim_example = apply_crop_shim(example, tuple(self.cfg.input_image_shape))
                yield crop_shim_example



class DatasetCo3DWindow(DatasetCo3D):
    '''return a video in the slicing window for evaluation / inference.
    '''

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSamplerEvaluation,
    ) -> None:
        super().__init__(cfg, stage, view_sampler)
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        # Collect chunks.
        self.chunks = []
        for root in cfg.roots:
            root = root / self.data_stage
            root_chunks = sorted(
                [path for path in root.iterdir() if path.suffix == ".torch"]
            )
            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        
    # def _get_extrinsics_scale(self, context_extrinsics):

    #     if self.cfg.make_baseline_1: # True
    #         a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
    #         scale = (a - b).norm()
    #         if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
    #             print(
    #                 f"Skipped {scene} because of baseline out of range: "
    #                 f"{scale:.6f}"
    #             )
    #             continue
    #         extrinsics[:, :3, 3] /= scale
    #     else:
    #         scale = 1


    #     pass

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train", "val"):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        # st()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:

            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in ("train", "val"):
                chunk = self.shuffle(chunk) # bs=16 here.

            # for example in chunk: # load a scene
            for idx, vid_example in enumerate(chunk): # load a scene
            # for idx, vid_example in enumerate(chunk[:1]): # 
                # print('load a single scene here ')
                extrinsics, intrinsics = self.convert_poses(vid_example["cameras"])
                scene = vid_example["key"]
                # (Pdb) p len(chunk[0]['images']) # the frames of each example. Pretty diverse actually.
                # 140

                # context_indices, target_indices, overlap = self.view_sampler.sample(
                #     scene,
                #     extrinsics,
                #     intrinsics,
                # )

                # ! return all pairs of a video
                try:
                    context_indices_list, target_indices_list, overlap_list = self.view_sampler.sample_sliding_window(
                        scene,
                        extrinsics,
                        intrinsics,
                        # return_whole_traj=True
                    )
                except ValueError as e:
                    print('skip scene: ', scene, e)
                    continue 


                vid_window_example_list = []

                for idx, (context_indices, target_indices, overlap) in enumerate(zip(context_indices_list, target_indices_list, overlap_list)):

                    extrinsics_of_the_window = extrinsics.clone()

                    # Skip the example if the field of view is too wide.
                    if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                        continue

                    # Load the images.
                    try:
                        context_images = [
                            vid_example["images"][index.item()] for index in context_indices
                        ]
                        context_images = self.convert_images(context_images)
                        target_images = [
                            vid_example["images"][index.item()] for index in target_indices
                        ]
                        target_images = self.convert_images(target_images)
                    except IndexError:
                        continue
                    except OSError:
                        print(f"Skipped bad example {example['key']}.")  # DL3DV-Full have some bad images
                        # st()
                        continue

                    # Skip the example if the images don't have the right shape.
                    context_image_invalid = context_images.shape[1:] != (3, *self.cfg.original_image_shape)
                    target_image_invalid = target_images.shape[1:] != (3, *self.cfg.original_image_shape)
                    # indices_valid = bool(len(context_indices) != 9 or len(target_indices) != 4 )
                    # indices_valid = bool(len(context_indices) != 9)
                    # st()
                    if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    # if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                        print(
                            f"Skipped bad example {example['key']}. Context shape was "
                            f"{context_images.shape} and target shape was "
                            f"{target_images.shape}."
                        )
                        continue

                    # Resize the world to make the baseline 1.
                    context_extrinsics = extrinsics_of_the_window[context_indices]
                    target_extrinsics = extrinsics[target_indices]

                    # Load Near and Far
                    context_near, context_far = self._process_near_far(context_extrinsics)
                    target_near, target_far = self._process_near_far(target_extrinsics)

                    # st()
                    # ! NO NORMALIZATION
                    # if self.cfg.make_baseline_1: # True
                    #     a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
                    #     scale = (a - b).norm()
                    #     if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                    #         print(
                    #             f"Skipped {scene} because of baseline out of range: "
                    #             f"{scale:.6f}"
                    #         )
                    #         continue
                    #     extrinsics_of_the_window[:, :3, 3] /= scale
                    # else:
                    #     scale = 1

                    # global_extrinsics = extrinsics.clone()
                    if self.cfg.relative_pose: # True
                        extrinsics_of_the_window = camera_normalization(extrinsics_of_the_window[context_indices][0:1], extrinsics_of_the_window)
                    
                    # st()

                    example = {
                        "context": {
                            "extrinsics": extrinsics_of_the_window[context_indices],
                            "intrinsics": intrinsics[context_indices],
                            "image": context_images,
                            # "near": self.get_bound("near", len(context_indices)) / scale,
                            # "far": self.get_bound("far", len(context_indices)) / scale,
                            "near": context_near,
                            "far": context_far,
                            "index": context_indices,
                            "overlap": overlap,
                        },
                        "target": {
                            "extrinsics": extrinsics_of_the_window[target_indices],
                            # "global_extrinsics": global_extrinsics[target_indices],
                            "intrinsics": intrinsics[target_indices],
                            "image": target_images,
                            "near": target_near, 
                            "far": target_far,
                            "index": target_indices,
                        },
                        "scene": scene,
                        "window_idx": idx,
                        # "scale": scale,
                        "normalize_pts": True,
                        "scene_scale": 8, 
                    }

                    if idx == 0: # record the global E
                        example['global_extrinsics'] = extrinsics_of_the_window

                    if self.stage == "train" and self.cfg.augment:
                        example = apply_augmentation_shim(example) # reflect views
                    cropped_example = apply_crop_shim(example, tuple(self.cfg.input_image_shape))
                    vid_window_example_list.append(cropped_example)

                    print('playing with 2 windows for debug')
                    # if len(vid_window_example_list) > 1:
                    #     break

                # return_global_traj = True
                return_global_traj = False

                if return_global_traj:
                    
                    # scale = vid_window_example_list[0]['scale'] # use the first window input as the global scale.

                    traj_extrinsics = vid_window_example_list[0]['global_extrinsics'].clone() # ctx[0] is the canonical frame, not extrinsics[0]
                    intrinsics = vid_window_example_list[0]['target']['intrinsics'].clone()[0:1].repeat_interleave(len(traj_extrinsics), 0) # already scaled

                    traj_near, traj_far = self._process_near_far(traj_extrinsics)

                    # st()

                    vid_window_example_list.append(
                        {
                            "extrinsics": traj_extrinsics,
                            "intrinsics": intrinsics,
                            "near": traj_near,
                            "far": traj_far,
                            "scene": scene,
                            "window_idx": -1, # sentinal for global trajectory
                            # 'scale': scale,
                            'image': torch.cat([vid_window_example_list[i]['target']['image'] for i in range(len(vid_window_example_list))]), # concat the target images
                            "normalize_pts": True,
                            "scene_scale": 8, 
                        }
                    )

                # st()
                yield vid_window_example_list

# '''