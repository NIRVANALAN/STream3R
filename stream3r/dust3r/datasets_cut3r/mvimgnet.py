import os.path as osp
import pickle
import json
from pathlib import Path
import cv2
from pdb import set_trace as st
import numpy as np
import itertools
import os
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from stream3r.dust3r.datasets_cut3r.base.base_multiview_dataset import BaseMultiViewDataset
from stream3r.dust3r.utils.image import imread_cv2

from safetensors.numpy import save_file, load_file


class MVImgNet_Multi(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = True
        self.is_metric = False
        self.max_interval = 32
        super().__init__(*args, **kwargs)

        self.loaded_data = self._load_data()

    def _load_data(self):
        # self.scenes = os.listdir(self.ROOT)
        # st()

        if os.path.exists(osp.join(self.ROOT, f'pre-calculated-loaddata-{self.num_views}.pkl')):
        # if False:
            with open(osp.join(self.ROOT, f'pre-calculated-loaddata-{self.num_views}.pkl'), 'rb') as f:
                pre_calculated_data = pickle.load(f)

                self.scenes = pre_calculated_data['scenes']
                self.sceneids = pre_calculated_data['sceneids']
                self.images = pre_calculated_data['images']
                self.start_img_ids = pre_calculated_data['start_img_ids']
                self.scene_img_list = pre_calculated_data['scene_img_list']
                 
                self.invalid_scenes = {scene: False for scene in self.scenes}
        else:

            ls_file = Path(self.ROOT) / 'mvimgnet_ls.txt'
            if ls_file.exists():
                with open(ls_file) as f:
                    self.scenes = [scene.strip() for scene in f.readlines()]
            else:
                self.scenes = os.listdir(self.ROOT)

            offset = 0
            scenes = []
            sceneids = []
            scene_img_list = []
            images = []
            start_img_ids = []

            j = 0
            for scene in tqdm(self.scenes):
                scene_dir = osp.join(self.ROOT, scene)
                if not os.path.isdir(scene_dir):
                    continue
                rgb_dir = osp.join(scene_dir, "rgb")
                basenames = sorted(
                    [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
                )

                num_imgs = len(basenames)
                cut_off = (
                    self.num_views if not self.allow_repeat else max(self.num_views // 3, 3)
                )

                if num_imgs < cut_off:
                    print(f"Skipping {scene}")
                    continue

                img_ids = list(np.arange(num_imgs) + offset)
                start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

                start_img_ids.extend([(scene, id) for id in start_img_ids_])
                sceneids.extend([j] * num_imgs)
                images.extend(basenames)
                scenes.append(scene)
                scene_img_list.append(img_ids)

                # offset groups
                offset += num_imgs
                j += 1

            self.scenes = scenes
            self.sceneids = sceneids
            self.images = images
            self.start_img_ids = start_img_ids
            self.scene_img_list = scene_img_list

            self.invalid_scenes = {scene: False for scene in self.scenes}

            # st() # save all required stuffs to the json, avoid re-calculating all the stuffs during loading

            with open(osp.join(self.ROOT, f'pre-calculated-loaddata-{self.num_views}.pkl'), 'wb') as f:
                pickle.dump(
                    dict(scenes=self.scenes,
                        sceneids=self.sceneids,
                        images=images,
                        start_img_ids=start_img_ids,
                        scene_img_list=scene_img_list,
                        invalid_scenes=self.invalid_scenes), 
                    f,
                )
            # st()
            # pass

    def __len__(self):
        return len(self.start_img_ids)

    def get_image_num(self):
        return len(self.images)

    def _get_views(self, idx, resolution, rng, num_views):
        invalid_seq = True
        scene, start_id = self.start_img_ids[idx]

        while invalid_seq:
            while self.invalid_scenes[scene]:
                idx = rng.integers(low=0, high=len(self.start_img_ids))
                scene, start_id = self.start_img_ids[idx]

            all_image_ids = self.scene_img_list[self.sceneids[start_id]]
            pos, ordered_video = self.get_seq_from_start_id(
                num_views, start_id, all_image_ids, rng, max_interval=self.max_interval
            )
            image_idxs = np.array(all_image_ids)[pos]

            views = []
            for view_idx in image_idxs:
                scene_id = self.sceneids[view_idx]
                scene_dir = osp.join(self.ROOT, self.scenes[scene_id])
                rgb_dir = osp.join(scene_dir, "rgb")
                cam_dir = osp.join(scene_dir, "cam")

                basename = self.images[view_idx]

                try:
                    # Load RGB image
                    rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".jpg"))
                    # Load depthmap, no depth, set to all ones
                    depthmap = np.ones_like(rgb_image[..., 0], dtype=np.float32)
                    cam = load_file(osp.join(cam_dir, basename + ".safetensor"))
                    camera_pose = cam["pose"]
                    intrinsics = np.eye(3)
                    intrinsics[0, 0] = cam["intrinsics"][0, 0]
                    intrinsics[1, 1] = cam["intrinsics"][0, 0]
                    intrinsics[0, 2] = cam["intrinsics"][1, 1]
                    intrinsics[1, 2] = cam["intrinsics"][0, 2]
                except:
                    print(f"Error loading {scene} {basename}, skipping")
                    self.invalid_scenes[scene] = True
                    break

                rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, rng=rng, info=view_idx
                )

                views.append(
                    dict(
                        img=rgb_image,
                        depthmap=depthmap.astype(np.float32),
                        camera_pose=camera_pose.astype(np.float32),
                        camera_intrinsics=intrinsics.astype(np.float32),
                        dataset="MVImgnet",
                        label=self.scenes[scene_id] + "_" + basename,
                        instance=f"{str(idx)}_{str(view_idx)}",
                        is_metric=self.is_metric,
                        is_video=ordered_video,
                        quantile=np.array(0.98, dtype=np.float32),
                        img_mask=True,
                        ray_mask=False,
                        camera_only=True,
                        depth_only=False,
                        single_view=False,
                        reset=False,
                    )
                )
            if len(views) == num_views:
                invalid_seq = False
        return views
