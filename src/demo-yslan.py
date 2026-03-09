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

from src.visualization.color_map import apply_color_map_to_image, colorize_depth

# sys.path.insert(0, '.')
sys.path.insert(0, './src')

import numpy as np
import trimesh
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
from src.dust3r.utils.geometry import geotrf, inv

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

# def parse_args():
#     """Parse command-line arguments."""
#     parser = argparse.ArgumentParser(
#         description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
#     )
#     parser.add_argument(
#         "--model_path",
#         type=str,
#         default="src/cut3r_512_dpt_4_64.pth",
#         help="Path to the pretrained model checkpoint.",
#     )
#     parser.add_argument(
#         "--seq_path",
#         type=str,
#         default="",
#         help="Path to the directory containing the image sequence.",
#     )
#     # parser.add_argument(
#     #     "--device",
#     #     type=str,
#     #     default="cuda",
#     #     help="Device to run inference on (e.g., 'cuda' or 'cpu').",
#     # )
#     parser.add_argument(
#         "--size",
#         type=int,
#         default="512",
#         help="Shape that input images will be rescaled to; if using 224+linear model, choose 224 otherwise 512",
#     )
#     parser.add_argument(
#         "--vis_threshold",
#         type=float,
#         default=1.5,
#         help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="./demo_tmp",
#         help="value for tempfile.tempdir",
#     )

#     return parser.parse_args()


# @hydra.main(
#     version_base=None,
#     config_path="../config",
#     config_name="main",
# )
def load_model(cfg, cfg_dict):

    # cfg = load_typed_root_config(cfg_dict) #! map it to old pixelSplat configs
    # set_cfg(cfg_dict)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    # Load the encoder weights.
    # if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
    assert cfg.model.encoder.pretrained_weights

    weight_path = cfg.model.encoder.pretrained_weights
    ckpt_weights = torch.load(weight_path, map_location='cpu')
    print('loading from: ', cfg.model.encoder.pretrained_weights)
    # st()
    if 'model' in ckpt_weights:
        ckpt_weights = ckpt_weights['model']
        ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)

        # missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        # ! some k, v not matched (conf dim) for multiview model.
        encoder_state_dict = encoder.state_dict()
        missing_keys, unexpected_keys = [], []
        for k, v in ckpt_weights.items():
            if k in encoder_state_dict:
                # try:
                if encoder_state_dict[k].shape == v.shape:
                    encoder_state_dict[k] = v

                else:
                    if 'dpt.head.4' in k:
                        encoder_state_dict[k][:v.shape[0]] = v  # e.g., conf
                    else:
                        raise ValueError(f"unexpected keys: {k}")

            else:
                unexpected_keys.append(k)

        for k in encoder_state_dict.keys():
            if k not in ckpt_weights:
                missing_keys.append(k)

        # st()
        encoder.load_state_dict(encoder_state_dict, strict=True)
        if len(unexpected_keys):
            print('unexpected_keys: ', unexpected_keys)

        if len(missing_keys):
            print('missing_keys: ', missing_keys)

    elif 'state_dict' in ckpt_weights:
        ckpt_weights = ckpt_weights['state_dict']
        ckpt_weights = {
            k[8:]: v
            for k, v in ckpt_weights.items() if k.startswith('encoder.')
        }
        # ! load dpt_self weights

        # ! load the dpt weights from cross to self as initialization
        if 'downstream_head_self.dpt.act_postprocess.0.1.weight' in encoder.state_dict(
        ) and 'downstream_head_self.dpt.act_postprocess.0.1.weight' not in ckpt_weights:
            new_ckpt_weights = copy.deepcopy(ckpt_weights)
            for k, v in ckpt_weights.items():
                if k.startswith('downstream_head1'):
                    new_ckpt_weights[k.replace(
                        'downstream_head1',
                        'downstream_head_self')] = v.clone()
            print('initialize downstream_head_self with downstream_head1')
        else:
            new_ckpt_weights = ckpt_weights

        missing_keys, unexpected_keys = encoder.load_state_dict(
            new_ckpt_weights, strict=False)

        if len(unexpected_keys):
            print('unexpected_keys: ', unexpected_keys)

        if len(missing_keys):
            print('missing_keys: ', missing_keys)
    else:
        raise ValueError(f"Invalid checkpoint format: {weight_path}")

    if cfg_dict.enable_joint_render_and_3d_sup:
        ModelClass = ModelWrapperMust3RWithRender
    else:
        ModelClass = ModelWrapperMust3R

    model_wrapper = ModelClass(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker=None,
        distiller=None,
    )

    return model_wrapper


def prepare_input(img_paths,
                  img_mask,
                  size,
                  raymaps=None,
                  raymap_mask=None,
                  revisit=1,
                  update=True):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    from src.dust3r.utils.image import load_images

    images = load_images(img_paths, size=size)
    views = []

    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img":
                images[i]["img"],
                "ray_map":
                torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape":
                torch.from_numpy(images[i]["true_shape"]),
                "idx":
                i,
                "instance":
                str(i),
                "camera_pose":
                torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask":
                torch.tensor(True).unsqueeze(0),
                "ray_mask":
                torch.tensor(False).unsqueeze(0),
                "update":
                torch.tensor(True).unsqueeze(0),
                "reset":
                torch.tensor(False).unsqueeze(0),
            }
            views.append(view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(
            raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (images[j]["img"] if img_mask[i] else torch.full_like(
                    images[0]["img"], torch.nan)),
                "ray_map": (raymaps[k] if raymap_mask[i] else torch.full_like(
                    raymaps[0], torch.nan)),
                "true_shape":
                (torch.from_numpy(images[j]["true_shape"]) if img_mask[i] else
                 torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))),
                "idx":
                i,
                "instance":
                str(i),
                "camera_pose":
                torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0),
                "img_mask":
                torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask":
                torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update":
                torch.tensor(img_mask[i]).unsqueeze(0),
                "reset":
                torch.tensor(False).unsqueeze(0),
            }
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views


def prepare_output(outputs, outdir, revisit=1, use_pose=True):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    pts3ds_self_ls = [
        output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
    ]
    pts3ds_other = [
        output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
    ]

    pts3ds_other_backup = [
        output["pts3d_in_other_view"].cpu() for output in outputs["pred"]
    ] # for save.
    pts3ds_self_ls_backup = [
        output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
    ]

    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    B, H, W, _ = pts3ds_self.shape

    # Recover camera poses.
    pr_poses, pr_intrinsics = zip(*[
        pose_encoding_to_extri_intri(pred["camera_pose"][-1].clone(), (H, W))
        for pred in outputs["pred"]
    ])

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    # assert not use_pose
    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    pp = torch.tensor([W // 2, H // 2],
                      device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self,
                                         pp,
                                         focal_mode="weiszfeld")
    # TODO, actually we also directly predict focal here

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0)
        for output in outputs["views"]
    ]

    # st()
    cam_dict = {
        "focal": focal.cpu().numpy(),  # ! in absolute scale, e.g., 436. torch.Size([B])
        "pp": pp.cpu().numpy(), # B, 2
        "R": R_c2w.cpu().numpy(), # B, 3, 3
        "t": t_c2w.cpu().numpy(), # B, 3
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat([
        0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0)
        for output in outputs["views"]
    ])  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (torch.eye(3).unsqueeze(0).repeat(
        cam2world_tosave.shape[0], 1, 1))  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "pts3d"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    for f_id in range(len(pts3ds_self)):
        depth = depths_tosave[f_id].cpu().numpy()
        conf = conf_self_tosave[f_id].cpu().numpy()
        color = colors_tosave[f_id].cpu().numpy()
        c2w = cam2world_tosave[f_id].cpu().numpy()
        intrins = intrinsics_tosave[f_id].cpu().numpy()
        # ! save img also

        # save pts3d
        pts3d_other = pts3ds_other_backup[f_id].cpu().numpy()
        pts3d_self = pts3ds_self_ls_backup[f_id].cpu().numpy()

        np.savez(os.path.join(outdir, "pts3d", f"{f_id:06d}.npz"),
                 pts3d_other=pts3d_other,
                 pts3d_self=pts3d_self,
                 )

        # st()
        depth_img = colorize_depth(depth, cmap='Spectral_r').astype(np.uint8)
        iio.imwrite(
            os.path.join(outdir, "depth", f"{f_id:06d}.png"),
            depth_img,
        )

        # conf_img = (conf - conf.min()) / (conf.max() - conf.min()) # 0-1
        conf_img = colorize_depth(conf, cmap='Spectral_r').astype(np.uint8)
        iio.imwrite(
            os.path.join(outdir, "conf", f"{f_id:06d}.png"),
            conf_img,
        )


        np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
        np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
        iio.imwrite(
            os.path.join(outdir, "color", f"{f_id:06d}.png"),
            (color * 255).astype(np.uint8),
        )
        np.savez(
            os.path.join(outdir, "camera", f"{f_id:06d}-pose_intrinsics.npz"),
            pose=c2w,
            intrinsics=intrins,
        )
        np.savez(os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
                 **cam_dict)  # for easier loading

    return pts3ds_other, colors, conf_other, cam_dict


def load_output_from_dir(outdir, use_pose=False):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """

    # if use_pose:
    #     transformed_pts3ds_other = []
    #     for pose, pself in zip(pr_poses, pts3ds_self):
    #         transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
    #     pts3ds_other = transformed_pts3ds_other
    #     conf_other = conf_self

    # ! load data
    pts3d_path = os.path.join(outdir, "pts3d")
    rgb_path = os.path.join(outdir, "color")
    cam_path = os.path.join(outdir, "camera")
    V = len(os.listdir(pts3d_path))  # length of the input views

    colors = []
    pts3ds_other = []
    cam_dict_list = []

    h, w = 256, 256

    for v_idx in range(V):
        # st()
        colors.append(iio.imread(os.path.join(
            rgb_path, f'{v_idx}.jpg')))  # how to normalize?
        if h is None:
            h, w = colors[0].shape[:2]
        cam_dict_list.append(dict(np.load(os.path.join(cam_path, f'{v_idx}.npz'))))
        # pts3ds_other.append(
        #     trimesh.load(os.path.join(pts3d_path, f'{v_idx}.obj'))) # type: ignore
        pts3ds_other.append(
            np.load(os.path.join(pts3d_path, f'{v_idx}.npy'))) # type: ignore

    colors = [color / 255.0 for color in colors]  # make in [0, 1]
    # pts3ds_other = [pts3d.vertices for pts3d in pts3ds_other] # PointCloud to np array

    # ! scale and normalize
    # in_camera1 = inv(cam_dict_list[0]["camera_pose"])
    # pts3ds_other = [geotrf(in_camera1, pts3d) for pts3d in pts3ds_other]  # world
    # cam_dict['camera_pose'] = np.matmul(in_camera1[None], cam_dict['camera_pose'])

    cam_dict = {  # make B ...
        k: np.stack([cam[k] for cam in cam_dict_list])
        for k in cam_dict_list[0].keys()
    }

    # ! scale pts and cam with a scale factor, since default scale, camera is invisible?
    # scale_factor = 1/28
    scale_factor = 1
    pts3ds_other = [pts * scale_factor for pts in pts3ds_other]
    cam_dict['t'] *= scale_factor

    conf_other = [np.ones(shape=(1, h, w))*100 for pts3d in pts3ds_other] # 1 h w 3. must be >0

    return pts3ds_other, colors, conf_other, cam_dict


def parse_seq_path(p):
    if os.path.isdir(p):
        img_paths = sorted(glob.glob(f"{p}/*"))
        tmpdirname = None
    else:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {p}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if video_fps == 0:
            cap.release()
            raise ValueError(f"Error: Video FPS is 0 for {p}")
        frame_interval = 1
        frame_indices = list(range(0, total_frames, frame_interval))
        print(
            f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}"
        )
        img_paths = []
        tmpdirname = tempfile.mkdtemp()
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(tmpdirname, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            img_paths.append(frame_path)
        cap.release()
    return img_paths, tmpdirname


def run_inference(cfg, args):
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

    model = load_model(cfg, args).to(device)
    model = model.eval()

    # Add the checkpoint path (required for model imports in the dust3r package).
    # add_path_to_dust3r(args.model_path)

    # Import model and inference functions after adding the ckpt path.

    # Prepare image file paths.
    img_paths, tmpdirname = parse_seq_path(args.seq_path)
    if not img_paths:
        print(f"No images found in {args.seq_path}. Please verify the path.")
        return
    
    total_size = 10
    divide_size = len(img_paths)//total_size
    img_paths = img_paths[::divide_size]

    print(f"Found {len(img_paths)} images in {args.seq_path}.")
    img_mask = [True] * len(img_paths)

    # img_paths.reverse() # ! check performance

    # Prepare input views.
    print("Preparing input views...")
    views = prepare_input(
        img_paths=img_paths,
        img_mask=img_mask,
        size=args.size,
        revisit=1,
        update=True,
    )
    if tmpdirname is not None:
        shutil.rmtree(tmpdirname)

    # Run inference.
    print("Running inference...")
    start_time = time.time()
    outputs = inference(views, model, device)
    total_time = time.time() - start_time
    per_frame_time = total_time / len(views)
    print(
        f"Inference completed in {total_time:.2f} seconds (average {per_frame_time:.2f} s per frame)."
    )

    # Process outputs for visualization.
    print("Preparing output for visualization...")
    pts3ds_other, colors, conf, cam_dict = prepare_output(
        # outputs, args.output_dir, 1, True
        outputs,
        args.output_dir,
        1,
        False)
        # True)

    # Convert tensors to numpy arrays for visualization.
    pts3ds_to_vis = [p.cpu().numpy()
                     for p in pts3ds_other]  # shape of item: 1 288 512 3
    colors_to_vis = [c.cpu().numpy() for c in colors]  # 1 288 512 3
    edge_colors = [None] * len(pts3ds_to_vis)
    # st()

    # Create and run the point cloud viewer.
    print("Launching point cloud viewer...")
    viewer = PointCloudViewer(
        model,
        # state_args,
        pts3ds_to_vis,
        colors_to_vis,
        conf,
        cam_dict,
        device=device,
        edge_color_list=edge_colors,
        show_camera=True,
        vis_threshold=args.vis_threshold,
        size=args.size,
        port=9001,
    )
    viewer.run()


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def main(cfg_dict: DictConfig):

    cfg = load_typed_root_config(cfg_dict)  #! map it to old pixelSplat configs
    set_cfg(cfg_dict)

    # args = parse_args()
    if not cfg_dict.seq_path:
        print(
            "No inputs found! Please use our gradio demo if you would like to iteractively upload inputs."
        )
        return
    else:
        run_inference(cfg, cfg_dict)


if __name__ == "__main__":
    main()
