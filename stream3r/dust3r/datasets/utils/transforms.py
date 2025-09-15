# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUST3R default transforms
# --------------------------------------------------------
import torchvision.transforms as tvf

from stream3r.dust3r.utils.image import ImgNorm

# define the standard image transforms
ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])

# from CoTracker and DELTA
TrackAug = tvf.Compose([
    tvf.RandomApply([tvf.ColorJitter(0.2, 0.2, 0.2, 0.25/3.14)], p=0.25),
    tvf.RandomApply([tvf.GaussianBlur(11, sigma=(0.1, 2.0))], p=0.05),
    ImgNorm
])