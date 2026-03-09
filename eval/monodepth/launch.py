import torch
import numpy as np
import cv2
import glob
import argparse
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from scipy.optimize import minimize
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from collections import defaultdict
from eval.monodepth.metadata import dataset_metadata
# from add_ckpt_path import add_path_to_dust3r

# ===========================================
# to load PL, hydra, megaconf stuffs.
import hydra
from omegaconf import DictConfig, OmegaConf
from pdb import set_trace as st

from src.config import load_typed_root_config
from src.dataset.data_module import DataModuleDust3r, DataModule, build_dataset
from src.global_cfg import set_cfg
from src.loss import get_losses
from src.misc.LocalLogger import LocalLogger
from src.misc.step_tracker import StepTracker
from src.misc.wandb_tools import update_checkpoint_path
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
# from src.model.model_wrapper import ModelWrapper
from src.model.model_dust3r_wrapper import ModelWrapperMust3R
from src.model.model_dust3r_with_render_wrapper import ModelWrapperMust3RWithRender
from src.dust3r.utils.image import load_images_for_eval as load_images
# from src.misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag
from src.visualization.color_map import apply_color_map_to_image, colorize_depth

torch.backends.cuda.matmul.allow_tf32 = True

# avoid high cpu usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
# ===========================================

from src.utils import load_encoder_ckpt


def get_args_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--weights", type=str, help="path to the model weights", default=""
    # )

    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="pytorch device")
    parser.add_argument("--output_dir",
                        type=str,
                        default="",
                        help="value for outdir")
    parser.add_argument("--no_crop",
                        type=bool,
                        default=True,
                        help="whether to crop input data")
    parser.add_argument("--full_seq",
                        type=bool,
                        default=False,
                        help="whether to use all seqs")
    parser.add_argument("--seq_list", default=None)

    parser.add_argument("--eval_dataset",
                        type=str,
                        default="nyu",
                        choices=list(dataset_metadata.keys()))
    return parser


def eval_mono_depth_estimation(args, model, device):
    metadata = dataset_metadata.get(args.eval_dataset)
    if metadata is None:
        raise ValueError(f"Unknown dataset: {args.eval_dataset}")

    img_path = metadata.get("img_path")
    if "img_path_func" in metadata:
        img_path = metadata["img_path_func"](args)

    process_func = metadata.get("process_func")
    if process_func is None:
        raise ValueError(
            f"No processing function defined for dataset: {args.eval_dataset}")

    for filelist, save_dir in process_func(args, img_path):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        eval_mono_depth(args, model, device, filelist, save_dir=save_dir)


def eval_mono_depth(args, model, device, filelist, save_dir=None):
    model.eval()
    load_img_size = 512
    for file in tqdm(filelist):
        # construct the "image pair" for the single image
        file = [file]
        images = load_images(file,
                             size=load_img_size,
                             verbose=False,
                             crop=not args.no_crop)
        views = []
        num_views = len(images)

        for i in range(num_views):
            view = {
                "img":
                images[i]["img"],
                # "ray_map":
                # torch.full(
                #     (
                #         images[i]["img"].shape[0],
                #         6,
                #         images[i]["img"].shape[-2],
                #         images[i]["img"].shape[-1],
                #     ),
                #     torch.nan,
                # ),
                "true_shape":
                torch.from_numpy(images[i]["true_shape"]),
                "idx":
                i,
                "instance":
                str(i),
                "camera_pose":
                torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(0),
                "img_mask":
                torch.tensor(True).unsqueeze(0),
                "ray_mask":
                torch.tensor(False).unsqueeze(0),
                "update":
                torch.tensor(True).unsqueeze(0),
                "reset":
                torch.tensor(False).unsqueeze(0),
            }
            views.append({
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in view.items()
            }) # to cuda

        views = views * 2  # for monocular

        # st()

        # ! move data to pixelsplat format

        model.eval()
        with torch.inference_mode():
            visualization_dump = {
                'ignore_gs': True,
            }  # ! since imposing depth / pts3d loss here
            batch, V = model._preprocess_views(views)  # type: ignore
            _ = model.encoder(
                batch["context"],
                0,
                visualization_dump=visualization_dump,
            )

            outputs, all_gts = model._prep_pred_and_gt_mv(views,
                                                        visualization_dump)

        outputs = [outputs[0]] # just use first view

        # outputs, state_args = inference(views, model, device)

        pts3ds_self = [
            # output["pts3d_in_self_view"].cpu() for output in outputs["pred"]
            output["pts3d_in_self_view"].cpu() for output in outputs
        ]
        depth_map = pts3ds_self[0][..., -1].mean(dim=0)

        if save_dir is not None:
            # save the depth map to the save_dir as npy
            np.save(
                f"{save_dir}/{file[0].split('/')[-1].replace('.png','depth.npy')}",
                depth_map.cpu().numpy(),
            )
            # also save the png
            # depth_map = (depth_map - depth_map.min()) / (depth_map.max() -
            #                                              depth_map.min())
            # depth_map = (depth_map * 255).cpu().numpy().astype(np.uint8)
            depth_map = colorize_depth(depth_map)
            
            cv2.imwrite(
                f"{save_dir}/{file[0].split('/')[-1].replace('.png','depth.jpg')}",
                depth_map,
            )
        # st()


@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def main(args: DictConfig):
    # args = get_args_parser()
    # args = args.parse_args()

    # st()

    if args.eval_dataset == "sintel":
        args.full_seq = True
    else:
        args.full_seq = False

    # ! create model using hydra config
    cfg = load_typed_root_config(args)  #! map it to old pixelSplat configs
    set_cfg(args)

    if args.enable_joint_render_and_3d_sup:
        ModelClass = ModelWrapperMust3RWithRender
    else:
        ModelClass = ModelWrapperMust3R

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    load_encoder_ckpt(cfg, encoder)
    # st()

    model = ModelClass(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker=None,
        distiller=None,
    ).to(args.device)

    # st()

    # add_path_to_dust3r(args.weights)
    # from dust3r.inference import inference
    # from dust3r.model import ARCroco3DStereo

    # model = ARCroco3DStereo.from_pretrained(args.weights).to(args.device)
    eval_mono_depth_estimation(args, model, args.device)


if __name__ == "__main__":
    main()
