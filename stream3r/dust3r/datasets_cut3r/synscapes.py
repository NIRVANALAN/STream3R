import os.path as osp
import pickle
import cv2
import numpy as np
import itertools
import os
import sys

sys.path.append(osp.join(osp.dirname(__file__), "..", ".."))
from tqdm import tqdm
from safetensors.numpy import save_file, load_file
from stream3r.dust3r.datasets_cut3r.base.base_multiview_dataset import BaseMultiViewDataset
from stream3r.dust3r.utils.image import imread_cv2


class SynScapes(BaseMultiViewDataset):

    def __init__(self, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        self.video = False
        self.is_metric = True
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data()

    def _load_data(self):

        if os.path.exists(osp.join(self.ROOT, f'pre-calculated-loaddata-{self.num_views}.pkl')):
            with open(osp.join(self.ROOT, f'pre-calculated-loaddata-{self.num_views}.pkl'),
                      'rb') as f:
                pre_calculated_data = pickle.load(f)
                self.img_names = pre_calculated_data['img_names']
            return

        rgb_dir = osp.join(self.ROOT, "rgb")
        basenames = sorted(
            [f[:-4] for f in os.listdir(rgb_dir) if f.endswith(".png")],
            key=lambda x: int(x),
        )
        self.img_names = basenames  # 25K imgs

        with open(osp.join(self.ROOT, f'pre-calculated-loaddata-{self.num_views}.pkl'),
                  'wb') as f:
            pickle.dump(dict(img_names=self.img_names), f)

    def __len__(self):
        return len(self.img_names)

    def get_image_num(self):
        return len(self.img_names)

    def _get_views(self, idx, resolution, rng, num_views):
        new_seed = rng.integers(0, 2**32) + idx
        new_rng = np.random.default_rng(new_seed)
        img_names = new_rng.choice(self.img_names, num_views, replace=False)

        views = []
        for v, img_name in enumerate(img_names):
            # Load RGB image
            rgb_image = imread_cv2(
                osp.join(self.ROOT, "rgb", f"{img_name}.png"))
            depthmap = np.load(osp.join(self.ROOT, "depth", f"{img_name}.npy"))
            sky_mask = (imread_cv2(
                osp.join(self.ROOT, "sky_mask", f"{img_name}.png"))[..., 0]
                        >= 127)
            depthmap[sky_mask] = -1.0
            depthmap = np.nan_to_num(depthmap, nan=0, posinf=0, neginf=0)
            depthmap[depthmap > 200] = 0.0

            intrinsics = load_file(
                osp.join(self.ROOT, "cam",
                         f"{img_name}.safetensor"))["intrinsics"]
            # camera pose is not provided, placeholder
            camera_pose = np.eye(4)

            rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
                rgb_image,
                depthmap,
                intrinsics,
                resolution,
                rng=rng,
                info=img_name)

            views.append(
                dict(
                    img=rgb_image,
                    depthmap=depthmap.astype(np.float32),
                    camera_pose=camera_pose.astype(np.float32),
                    camera_intrinsics=intrinsics.astype(np.float32),
                    dataset="synscapes",
                    label=img_name,
                    instance=f"{str(idx)}_{img_name}",
                    is_metric=self.is_metric,
                    is_video=False,
                    quantile=np.array(1.0, dtype=np.float32),
                    img_mask=True,
                    ray_mask=False,
                    camera_only=False,
                    depth_only=False,
                    single_view=True,
                    reset=True,
                ))
        assert len(views) == num_views
        return views


if __name__ == "__main__":
    import torch
    import pause
    from torchvision.transforms import ToPILImage
    from stream3r.dust3r.datasets.base.base_stereo_view_dataset import view_name
    from stream3r.dust3r.utils.image import rgb
    from stream3r.dust3r.viz import SceneViz, auto_cam_size
    from IPython.display import display
    from stream3r.dust3r.datasets.utils.transforms import ImgNorm, convert_input_to_pred_format, vis_track
    from stream3r.dust3r.utils.geometry import (
        geotrf,
        inv,
    )
    from stream3r.viz.viser_visualizer_track import start_visualization

    def main():
        dataset = SynScapes(
            split="train", allow_repeat=False, ROOT="/mnt/storage/yslan-data/cut3r_processed/processed_synscapes/",
            aug_crop=0, resolution=(512, 384), num_views=20, transform=ImgNorm
        )

        # import random
        # for i in random.sample(range(len(dataset)), 100):
        #     views = dataset[i]
        #     print(i)

        select_idx = 1
        views = dataset[select_idx]
        output = convert_input_to_pred_format(views)

        # save_path = os.path.join("develop/2d_compare/test_data", views[0]['dataset'] + str(select_idx))
        # os.makedirs(save_path, exist_ok=True)
        # for i in range(len(views)):
        #     print(view_name(views[i]))
        #     ToPILImage()(rgb(views[i]["img"])).save(f"{save_path}/{i}.png")

        server = start_visualization(
            output=output,
            min_conf_thr_percentile=0,
            global_conf_thr_value_to_drop_view=1,
            point_size=0.0016,
        )

        # share_url = servers.request_share_url()
        # print(share_url)

        pause.days(1)

    main()