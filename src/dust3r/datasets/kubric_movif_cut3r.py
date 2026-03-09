import os
import os.path as osp
import cv2
import numpy as np
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))

# from dust3r.datasets.base.base_stereo_view_dataset import BaseMultiViewDataset
from dust3r.datasets.base.base_multiview_dataset import BaseMultiViewDataset
from dust3r.utils.image import imread_cv2


class Kubric_Tracks(BaseMultiViewDataset):
    def __init__(self, *args, ROOT, **kwargs):
        super().__init__(*args, **kwargs)
        self.ROOT = ROOT
        self.video = True
        self.is_metric = True
        self.max_interval = 4
        assert self.split in ["train", "test"]

        if self.split == "train":
            self.ROOT = os.path.join(self.ROOT, "trackings")
        elif self.split == "test":
            self.ROOT = os.path.join(self.ROOT, "trackings_val")
        
        self.M_Blender2Opencv = np.array(
            [[1,  0,  0,  0],
              [0, -1,  0,  0],
              [0,  0, -1,  0],
              [0,  0,  0,  1]]
        )

        self.loaded_data = self._load_data()

    def _load_data(self):
        self.sceneids = os.listdir(self.ROOT)

    def __len__(self):
        return len(self.sceneids)

    def get_image_num(self):
        return len(self.sceneids * self.num_views)

    def _get_views(self, idx, resolution, rng, num_views):
        scene_id = self.sceneids[idx]
        scene_dir = osp.join(self.ROOT, scene_id)
        depth_range = np.load(osp.join(scene_dir, scene_id + ".npy"), allow_pickle=True).item()["depth_range"].astype(np.float32)
        depth_min = depth_range[0]
        depth_max = depth_range[1]

        is_reverse = self._rng.choice([True, False])
        if is_reverse:
            cam = np.load(osp.join(scene_dir, scene_id + "_dense_reverse.npy"), allow_pickle=True).item()
        else:
            cam = np.load(osp.join(scene_dir, scene_id + "_dense.npy"), allow_pickle=True).item()

        views = []

        total_frames = cam["intrinsics"].shape[0]
        idxs = [i for i in range(total_frames)]

        # sample num_views frames from the total frames sequentially
        if is_reverse:
            idxs = idxs[::-1]
        first = idxs[0]
        rest = idxs[1:]
        if self._rng.random() < 0.8:
            # randown sample sequentially
            sampled = self._rng.choice(rest, size=num_views-1, replace=False)
            sampled_sorted = sorted(sampled, key=lambda x: idxs.index(x))
        else:
            # sample first num_views frames
            sampled_sorted = rest[:num_views-1]
        idxs = [first] + sampled_sorted

        for i in idxs:
            basename = str(i).zfill(3)
            rgb_dir = osp.join(scene_dir, "frames")
            depth_dir = osp.join(scene_dir, "depths")

            rgb_image = imread_cv2(osp.join(rgb_dir, basename + ".png"))
            depth = cv2.imread(osp.join(depth_dir, basename + ".png"), cv2.IMREAD_ANYDEPTH).astype(np.float32)

            depthmap = depth_min + depth * (depth_max-depth_min) / 65535.0

            intrinsics = cam["intrinsics"][i]
            h, w = rgb_image.shape[:2]
            intrinsics = np.array([
                [ intrinsics[0][0] * w,  0.   , w / 2  ],
                [ 0.   , -intrinsics[1][1] * h, h / 2  ],
                [ 0.   ,  0.   , 1.   ]
            ])

            # the depth map in kubric is the distance from point to optical center
            # but we need to use the depth where the distance is along z axis
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]
            u, v = np.meshgrid(np.arange(w), np.arange(h))
            depthmap /= np.sqrt(1 + ((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2)

            # the camera system in kubric is blender, while we use opencv here
            camera_pose = cam["matrix_world"][i] @ self.M_Blender2Opencv

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics, resolution, rng=rng, info=idx
            )

            # generate img mask and raymap mask
            img_mask, ray_mask = self.get_img_and_ray_masks(
                self.is_metric, v, rng, p=[0.75, 0.2, 0.05]
            )

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="Kubric_Tracks",
                    label=scene_id + "_" + basename,
                    instance=osp.join(rgb_dir, basename + ".png"),
                    is_metric=self.is_metric,
                    is_video=True,
                    quantile=np.array(1.0, dtype=np.float32),
                    img_mask=img_mask,
                    ray_mask=ray_mask,
                    camera_only=False,
                    depth_only=False,
                    single_view=False,
                    reset=False,
                )
            )

        assert len(views) == num_views
        return views