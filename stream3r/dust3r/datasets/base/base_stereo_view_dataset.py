# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# base class for implementing datasets
# --------------------------------------------------------
import numpy as np
import PIL
import torch
import itertools

import stream3r.dust3r.datasets.utils.cropping as cropping
from stream3r.dust3r.datasets.base.easy_dataset import EasyDataset
from stream3r.dust3r.datasets.utils.transforms import ImgNorm
from stream3r.dust3r.utils.geometry import depthmap_to_absolute_camera_coordinates, geotrf, inv


class BaseStereoViewDataset(EasyDataset):
    """Define all basic options.

    Usage:
        class MyDataset (BaseStereoViewDataset):
            def _get_views(self, idx, rng):
                # overload here
                views = []
                views.append(dict(img=, ...))
                return views
    """

    def __init__(
        self,
        *,  # only keyword arguments
        split=None,
        resolution=None,  # square_size or (width, height) or list of [(width,height), ...]
        transform=ImgNorm,
        aug_crop=False,
        seed=None,
    ):
        self.num_views = 24
        self.split = split
        self._set_resolutions(resolution)

        if isinstance(transform, str):
            transform = eval(transform)
        self.transform = transform

        self.aug_crop = aug_crop
        self.seed = seed

    def __len__(self):
        return len(self.scenes)

    def get_stats(self):
        return f"{len(self)} pairs"

    def __repr__(self):
        resolutions_str = "[" + ";".join(f"{w}x{h}" for w, h in self._resolutions) + "]"
        return (
            f"""{type(self).__name__}({self.get_stats()},
            {self.split=},
            {self.seed=},
            resolutions={resolutions_str},
            {self.transform=})""".replace(
                "self.", ""
            )
            .replace("\n", "")
            .replace("   ", "")
        )

    def _get_views(self, idx, resolution, rng):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[
            ar_idx
        ]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        views = self._get_views(idx, resolution, self._rng)
        # if type(self).__name__ not in ["DTU", "SevenScenes", "NRGBD"]:
        #     assert len(views) == self.num_views

        has_valid_pts3d = False
        has_valid_track = False
        # check data-types
        for v, view in enumerate(views):
            assert (
                "pts3d" not in view
            ), f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view["idx"] = (idx, ar_idx, v)

            # encode the image
            width, height = view["img"].size
            view["true_shape"] = np.int32((height, width))
            view["img"] = self.transform(view["img"])

            assert "camera_intrinsics" in view
            if "camera_pose" not in view:
                view["camera_pose"] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(
                    view["camera_pose"]
                ).all(), f"NaN in camera pose for view {view_name(view)}"
            assert "pts3d" not in view
            assert "valid_mask" not in view
            assert np.isfinite(
                view["depthmap"]
            ).all(), f"NaN in depthmap for view {view_name(view)}"
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)

            view["pts3d"] = pts3d
            view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            if np.any(view["valid_mask"]):
                has_valid_pts3d = True

            if "track" in view:
                view["track_valid_mask"] = view["track_valid_mask"] & views[0]["valid_mask"]
                if np.any(view["track_valid_mask"]):
                    has_valid_track = True

                # we use track in the global coordinate system
                camera_pose = view["camera_pose"]
                R_cam2world = camera_pose[:3, :3]
                t_cam2world = camera_pose[:3, 3]

                # Express in absolute coordinates (invalid depth values)
                view["track"] = (
                    np.einsum("ik, vuk -> vui", R_cam2world, view["track"]) + t_cam2world[None, None, :]
                )

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view["camera_intrinsics"]

        if not has_valid_pts3d or ("track" in view and not has_valid_track):
            return self.__getitem__(((idx + 1) % self.__len__(), ar_idx))

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")

            if "is_2d_track" not in view:
                view["is_2d_track"] = False

        return views

    def _set_resolutions(self, resolutions):
        assert resolutions is not None, "undefined resolution"

        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        self._resolutions = []
        for resolution in resolutions:
            if isinstance(resolution, int):
                width = height = resolution
            else:
                (width, height), num_views = resolution
            assert isinstance(
                width, int
            ), f"Bad type for {width=} {type(width)=}, should be int"
            assert isinstance(
                height, int
            ), f"Bad type for {height=} {type(height)=}, should be int"
            assert isinstance(
                num_views, int
            ), f"Bad type for {num_views=} {type(num_views)=}, should be int"
            assert width >= height
            self._resolutions.append(((width, height), num_views))

    def _crop_resize_if_necessary(
        self, image, depthmap, intrinsics, resolution, rng=None, info=None, track=None, track_valid_mask=None, allow_random_transpose=True
    ):
        """This function:
        - first downsizes the image with LANCZOS inteprolation,
          which is better than bilinear interpolation in
        """
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # downscale with lanczos interpolation so that image.size == resolution
        # cropping centered on the principal point
        W, H = image.size
        cx, cy = intrinsics[:2, 2].round().astype(int)
        min_margin_x = min(cx, W - cx)
        min_margin_y = min(cy, H - cy)
        assert min_margin_x > W / 5, f"Bad principal point in view={info}"
        assert min_margin_y > H / 5, f"Bad principal point in view={info}"
        # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
        l, t = cx - min_margin_x, cy - min_margin_y
        r, b = cx + min_margin_x, cy + min_margin_y
        crop_bbox = (l, t, r, b)

        if track is not None:
            image, depthmap, intrinsics, track, track_valid_mask = cropping.crop_image_depthmap(
                image, depthmap, intrinsics, crop_bbox, track=track, track_valid_mask=track_valid_mask
            )
        else:
            image, depthmap, intrinsics = cropping.crop_image_depthmap(
                image, depthmap, intrinsics, crop_bbox
            )

        # transpose the resolution if necessary
        W, H = image.size  # new size
        assert resolution[0] >= resolution[1]
        if H > 1.1 * W:
            # image is portrait mode
            resolution = resolution[::-1]
        elif 0.9 < H / W < 1.1 and resolution[0] != resolution[1]:
            # image is square, so we chose (portrait, landscape) randomly
            if allow_random_transpose and rng.integers(2):
                resolution = resolution[::-1]

        # high-quality Lanczos down-scaling
        target_resolution = np.array(resolution)
        if self.aug_crop > 1:
            target_resolution += rng.integers(0, self.aug_crop)
        
        if track is not None:
            image, depthmap, intrinsics, track, track_valid_mask = cropping.rescale_image_depthmap(
                image, depthmap, intrinsics, target_resolution, track=track, track_valid_mask=track_valid_mask
            )
        else:
            image, depthmap, intrinsics = cropping.rescale_image_depthmap(
                image, depthmap, intrinsics, target_resolution
            )

        # actual cropping (if necessary) with bilinear interpolation
        intrinsics2 = cropping.camera_matrix_of_crop(
            intrinsics, image.size, resolution, offset_factor=0.5
        )
        crop_bbox = cropping.bbox_from_intrinsics_in_out(
            intrinsics, intrinsics2, resolution
        )
        if track is not None:
            image, depthmap, intrinsics2, track, track_valid_mask = cropping.crop_image_depthmap(
                image, depthmap, intrinsics, crop_bbox, track=track, track_valid_mask=track_valid_mask
            )
        else:
            image, depthmap, intrinsics2 = cropping.crop_image_depthmap(
                image, depthmap, intrinsics, crop_bbox
            )

        if track is not None:
            return image, depthmap, intrinsics2, track, track_valid_mask
        else:
            return image, depthmap, intrinsics2


class BaseStereoDynamicViewDataset(BaseStereoViewDataset):
    def __init__(
        self,
        *,
        split=None,
        resolution=None,
        transform=ImgNorm,
        aug_crop=False,
        seed=None,
        allow_repeat=True,
    ):
        super().__init__(
            split=split,
            resolution=resolution,
            transform=transform,
            aug_crop=aug_crop,
            seed=seed,
        )
        self.allow_repeat = allow_repeat

    @staticmethod
    def blockwise_shuffle(x, rng, block_shuffle):
        if block_shuffle is None:
            return rng.permutation(x).tolist()
        else:
            assert block_shuffle > 0
            blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
            shuffled_blocks = [rng.permutation(block).tolist() for block in blocks]
            shuffled_list = [item for block in shuffled_blocks for item in block]
            return shuffled_list

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        rng,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        """
        args:
            num_views: number of views to return
            id_ref: the reference id (first id)
            ids_all: all the ids
            rng: random number generator
            max_interval: maximum interval between two views
        returns:
            pos: list of positions of the views in ids_all, i.e., index for ids_all
            is_video: True if the views are consecutive
        """
        assert min_interval > 0, f"min_interval should be > 0, got {min_interval}"
        assert (
            min_interval <= max_interval
        ), f"min_interval should be <= max_interval, got {min_interval} and {max_interval}"
        assert id_ref in ids_all
        pos_ref = ids_all.index(id_ref)
        all_possible_pos = np.arange(pos_ref, len(ids_all))

        remaining_sum = len(ids_all) - 1 - pos_ref

        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                assert ids_all[-num_views] == id_ref
                return [pos_ref + i for i in range(num_views)], True, list(range(num_views))
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
            intervals = [
                rng.choice(range(min_interval, max_interval + 1))
                for _ in range(num_views - 1)
            ]

            # if video or collection
            if rng.random() < video_prob:
                # if fixed interval or random
                if rng.random() < fix_interval_prob:
                    # regular interval
                    fixed_interval = rng.choice(
                        range(
                            1,
                            min(remaining_sum // (num_views - 1) + 1, max_interval + 1),
                        )
                    )
                    intervals = [fixed_interval for _ in range(num_views - 1)]
                is_video = True
            else:
                is_video = False

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            pos_candidates = [p for p in all_possible_pos if p not in pos]
            pos = (
                pos
                + rng.choice(
                    pos_candidates, num_views - len(pos), replace=False
                ).tolist()
            )

            pos = (
                sorted(pos)
                if is_video
                else self.blockwise_shuffle(pos, rng, block_shuffle)
            )
        else:
            # assert self.allow_repeat
            uniq_num = remaining_sum
            new_pos_ref = rng.choice(np.arange(pos_ref + 1))
            new_remaining_sum = len(ids_all) - 1 - new_pos_ref
            new_max_interval = min(max_interval, new_remaining_sum // (uniq_num - 1))
            new_intervals = [
                rng.choice(range(1, new_max_interval + 1)) for _ in range(uniq_num - 1)
            ]

            revisit_random = rng.random()
            video_random = rng.random()

            if rng.random() < fix_interval_prob and video_random < video_prob:
                # regular interval
                fixed_interval = rng.choice(range(1, new_max_interval + 1))
                new_intervals = [fixed_interval for _ in range(uniq_num - 1)]
            pos = list(itertools.accumulate([new_pos_ref] + new_intervals))

            is_video = False
            if revisit_random < 0.5 or video_prob == 1.0:  # revisit, video / collection
                is_video = video_random < video_prob
                pos = (
                    self.blockwise_shuffle(pos, rng, block_shuffle)
                    if not is_video
                    else pos
                )
                num_full_repeat = num_views // uniq_num
                pos = (
                    pos * num_full_repeat
                    + pos[: num_views - len(pos) * num_full_repeat]
                )
            elif revisit_random < 0.9:  # random
                pos = rng.choice(pos, num_views, replace=True)
            else:  # ordered
                pos = sorted(rng.choice(pos, num_views, replace=True))
        assert len(pos) == num_views
        pos_ref_zero = [int(x) - pos[0] for x in pos]
        return pos, is_video, pos_ref_zero


# generate tracks for static scenes automatically
class BaseStereoStaticTrackDataset(BaseStereoDynamicViewDataset):
    def __init__(
        self,
        *,
        split=None,
        resolution=None,
        transform=ImgNorm,
        aug_crop=False,
        seed=None,
        allow_repeat=True,
        contain_dynamic=False,
    ):
        super().__init__(
            split=split,
            resolution=resolution,
            transform=transform,
            aug_crop=aug_crop,
            seed=seed,
            allow_repeat=allow_repeat,
        )
        self.contain_dynamic = contain_dynamic

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            # the idx is specifying the aspect-ratio
            idx, ar_idx = idx
        else:
            assert len(self._resolutions) == 1
            ar_idx = 0

        # set-up the rng
        if self.seed:  # reseed for each __getitem__
            self._rng = np.random.default_rng(seed=self.seed + idx)
        elif not hasattr(self, "_rng"):
            seed = torch.initial_seed()  # this is different for each dataloader process
            self._rng = np.random.default_rng(seed=seed)

        # over-loaded code
        resolution = self._resolutions[
            ar_idx
        ]  # DO NOT CHANGE THIS (compatible with BatchedRandomSampler)
        try:
            views = self._get_views(idx, resolution, self._rng)
        except Exception as e:
            print(f"get_views error: {e}")
            return self.__getitem__(((idx + 1) % self.__len__(), ar_idx))

        # if type(self).__name__ not in ["DTU", "SevenScenes", "NRGBD"]:
        #     assert len(views) == self.num_views

        has_valid_pts3d = False
        has_valid_track = False
        # check data-types
        for v, view in enumerate(views):
            assert (
                "pts3d" not in view
            ), f"pts3d should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            assert (
                "track" not in view
            ), f"track should not be there, they will be computed afterwards based on intrinsics+depthmap for view {view_name(view)}"
            view["idx"] = (idx, ar_idx, v)

            # encode the image
            width, height = view["img"].size
            view["true_shape"] = np.int32((height, width))
            view["img"] = self.transform(view["img"])

            assert "camera_intrinsics" in view
            if "camera_pose" not in view:
                view["camera_pose"] = np.full((4, 4), np.nan, dtype=np.float32)
            else:
                assert np.isfinite(
                    view["camera_pose"]
                ).all(), f"NaN in camera pose for view {view_name(view)}"
            assert "valid_mask" not in view
            assert np.isfinite(
                view["depthmap"]
            ).all(), f"NaN in depthmap for view {view_name(view)}"
            pts3d, valid_mask = depthmap_to_absolute_camera_coordinates(**view)
            view["pts3d"] = pts3d
            view["valid_mask"] = valid_mask & np.isfinite(pts3d).all(axis=-1)

            if not self.contain_dynamic:
                view["track_valid_mask"] = views[0]["valid_mask"]
            else:
                if v == 0:
                    view["track_valid_mask"] = views[0]["valid_mask"]
                else:
                    view["track_valid_mask"] = np.zeros_like(views[0]["valid_mask"])

            if v == 0:
                anchor_world_pts3d = view["pts3d"]

            # generate tracks
            # cur_inv_camera_matrix = inv(view["camera_pose"])
            # view["track"] = geotrf(cur_inv_camera_matrix, anchor_world_pts3d)
            view["track"] = anchor_world_pts3d

            if np.any(view["valid_mask"]):
                has_valid_pts3d = True
            if np.any(view["track_valid_mask"]):
                has_valid_track = True

            # check all datatypes
            for key, val in view.items():
                res, err_msg = is_good_type(key, val)
                assert res, f"{err_msg} with {key}={val} for view {view_name(view)}"
            K = view["camera_intrinsics"]

        if not has_valid_pts3d or ("track" in view and not has_valid_track):
            return self.__getitem__(((idx + 1) % self.__len__(), ar_idx))

        # last thing done!
        for view in views:
            # transpose to make sure all views are the same size
            transpose_to_landscape(view)
            # this allows to check whether the RNG is is the same state each time
            view["rng"] = int.from_bytes(self._rng.bytes(4), "big")

            if "is_2d_track" not in view:
                view["is_2d_track"] = False

        return views


def is_good_type(key, v):
    """returns (is_good, err_msg)"""
    if isinstance(v, (str, int, tuple)):
        return True, None
    if v.dtype not in (np.float32, torch.float32, bool, np.int32, np.int64, np.uint8):
        return False, f"bad {v.dtype=}"
    return True, None


def view_name(view, batch_index=None):
    def sel(x):
        return x[batch_index] if batch_index not in (None, slice(None)) else x

    db = sel(view["dataset"])
    label = sel(view["label"])
    instance = sel(view["instance"])
    return f"{db}/{label}/{instance}"


def transpose_to_landscape(view):
    height, width = view["true_shape"]

    if width < height:
        # rectify portrait to landscape
        assert view["img"].shape == (3, height, width)
        view["img"] = view["img"].swapaxes(1, 2)

        assert view["valid_mask"].shape == (height, width)
        view["valid_mask"] = view["valid_mask"].swapaxes(0, 1)

        assert view["depthmap"].shape == (height, width)
        view["depthmap"] = view["depthmap"].swapaxes(0, 1)

        assert view["pts3d"].shape == (height, width, 3)
        view["pts3d"] = view["pts3d"].swapaxes(0, 1)

        if "track" in view:
            assert view["track"].shape == (height, width, 3)
            view["track"] = view["track"].swapaxes(0, 1)
            assert view["track_valid_mask"].shape == (height, width)
            view["track_valid_mask"] = view["track_valid_mask"].swapaxes(0, 1)

        # transpose x and y pixels
        view["camera_intrinsics"] = view["camera_intrinsics"][[1, 0, 2]]
