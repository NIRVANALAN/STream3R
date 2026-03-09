import numpy as np
import glob
import torch
import imageio.v3 as imageio
import os
from pathlib import Path

# 13/1/2025
# concat the v=2, v=3, and v=8 videos for comparison.

root_dir = Path('/cpfs01/user/lanyushi.p/Repo/dust3r-follow-up/NoPoSplat/' )
output_dir = Path('outputs/test/re10k_v=k_compare' )

v2_dir = 'outputs/test/re10k_v-2'
v3_dir = 'outputs/test/re10k_v-3'
v8_dir = 'outputs/test/re10k_v-8_xformer'

all_instances = list([instance_dir.stem for instance_dir in (root_dir / v8_dir).glob('*.jpg' )])

# print(all_instances)

for instance in all_instances[:]:
    all_vids = []
    for model in (v2_dir, v3_dir, v8_dir):
        # print(instance)
        video_path = list((root_dir / model / 'video').glob(f'{instance}*.mp4'))[0]
        # print(video_path)
        all_vids.append(imageio.imread(video_path))
    all_vids = np.concatenate(all_vids, axis=-2) # V H W 3
    imageio.imwrite(output_dir / f'{instance}.mp4', all_vids)
