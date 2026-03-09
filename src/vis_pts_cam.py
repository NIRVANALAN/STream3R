#!/usr/bin/env python3
"""
3D Point Cloud Inference and Visualization Script

This script performs inference using the ARCroco3DStereo model and visualizes the
resulting 3D point clouds with the PointCloudViewer. Use the command-line arguments
to adjust parameters such as the model checkpoint path, image sequence directory,
image size, device, etc.

Usage:
    python usage.py [--model_path MODEL_PATH] [--seq_path SEQ_PATH] [--size IMG_SIZE]
                            [--device DEVICE] [--vis_threshold VIS_THRESHOLD] [--output_dir OUT_DIR]

Example:
    python usage.py --model_path src/cut3r_512_dpt_4_64.pth \
        --seq_path examples/001 --device cuda --size 512
"""

import os
import sys

# sys.path.insert(0, '.')
sys.path.insert(0, './src')

import numpy as np
import torch
import time
import glob
from pdb import set_trace as st
import random
import cv2
import argparse
import tempfile
import shutil
from copy import deepcopy
# from add_ckpt_path import add_path_to_dust3r
import imageio.v2 as iio

import copy
from src.dust3r.inference import inference, inference_recurrent
from src.viser_utils import PointCloudViewer

# from src.dust3r.utils.camera import pose_encoding_to_camera
from src.model.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.dust3r.post_process import estimate_focal_knowing_depth
from src.dust3r.utils.geometry import geotrf

# ! loading model using hydra
from omegaconf import DictConfig, OmegaConf
# from src.dust3r.model import ARCroco3DStereo

from src.misc.weight_modify import checkpoint_filter_fn
from src.config import load_typed_root_config
from pathlib import Path
import hydra
from src.global_cfg import set_cfg
from src.loss import get_losses
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_dust3r_wrapper import ModelWrapperMust3R
from src.model.model_dust3r_with_render_wrapper import ModelWrapperMust3RWithRender

# Set random seed for reproducibility.
random.seed(42)

from .demo import load_model, prepare_input, prepare_output, parse_seq_path, load_output_from_dir


def run_visualization(cfg, args):
    """
    Execute the full inference and visualization pipeline.

    Args:
        args: Parsed command-line arguments.
    """
    # Set up the computation device.
    device = 'cuda'
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available. Switching to CPU.")
        device = "cpu"
    
    # Load and prepare the model.
    # print(f"Loading model from {args.model_path}...")
    # model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    # model.eval()

    # model = load_model(cfg, args).to(device)
    # model = model.eval()

    # Add the checkpoint path (required for model imports in the dust3r package).
    # add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.

    # Prepare image file paths.
    # img_paths, tmpdirname = parse_seq_path(args.seq_path)
    # if not img_paths:
    #     print(f"No images found in {args.seq_path}. Please verify the path.")
    #     return

    # print(f"Found {len(img_paths)} images in {args.seq_path}.")
    # img_mask = [True] * len(img_paths)

    # img_paths.reverse() # ! check performance

    # Prepare input views.
    print("Preparing input views...")
    # views = prepare_input(
    #     img_paths=img_paths,
    #     img_mask=img_mask,
    #     size=args.size,
    #     revisit=1,
    #     update=True,
    # )
    # if tmpdirname is not None:
    #     shutil.rmtree(tmpdirname)

    # Run inference.
    # print("Running inference...")
    # start_time = time.time()
    # outputs = inference(views, model, device)
    # total_time = time.time() - start_time
    # per_frame_time = total_time / len(views)
    # print(
    #     f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    # )

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    pts3ds_to_vis, colors_to_vis, conf, cam_dict = load_output_from_dir(
        # outputs, args.output_dir, 1, True
        args.output_dir, False
    )
    # st()

    # Convert tensors to numpy arrays for visualization.
    # pts3ds_to_vis = [p.cpu().numpy() for p in pts3ds_other] # shape of item: 1 288 512 3
    # colors_to_vis = [c.cpu().numpy() for c in colors] # 1 288 512 3
    edge_colors = [None] * len(pts3ds_to_vis)
    # st()

    # Create and run the point cloud viewer.
    print("Launching point cloud viewer...")
    viewer = PointCloudViewer(
        # model,
        None,
        # state_args,
        pts3ds_to_vis,
        colors_to_vis,
        conf,
        cam_dict,
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        size = args.size,
        port=9001,
    )
    viewer.run()


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def main(cfg_dict: DictConfig):

    cfg = load_typed_root_config(cfg_dict) #! map it to old pixelSplat configs
    set_cfg(cfg_dict)

    # args = parse_args()
    # if not cfg_dict.seq_path:
    #     print(
    #         "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
    #     )
    #     return
    # else:
    run_visualization(cfg, cfg_dict)


if __name__ == "__main__":
    main()
