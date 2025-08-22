import torch
import numpy as np
import matplotlib
import numpy as np
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import os
import sys

from stream3r.models.stream3r import STream3R
from stream3r.dust3r.utils.device import collate_with_cat
from stream3r.dust3r.utils.image import load_images_for_eval as load_images
from stream3r.utils.utils import ImgDust3r2Stream3r

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.monodepth.metadata import dataset_metadata


torch.backends.cuda.matmul.allow_tf32 = True

# avoid high cpu usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
# ===========================================


def colorize_depth(depth: np.ndarray,
                   mask: np.ndarray = None,
                   normalize: bool = True,
                   cmap: str = 'Spectral') -> np.ndarray:
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        depth = np.where((depth > 0) & mask, depth, np.nan)
    disp = 1 / depth
    if normalize:
        min_disp, max_disp = np.nanquantile(disp,
                                            0.001), np.nanquantile(disp, 0.99)
        disp = (disp - min_disp) / (max_disp - min_disp)
    colored = np.nan_to_num(matplotlib.colormaps[cmap](1.0 - disp)[..., :3], 0)
    colored = np.ascontiguousarray((colored.clip(0, 1) * 255).astype(np.uint8))
    return colored


def get_args_parser():
    parser = argparse.ArgumentParser()

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
    for file in tqdm(filelist):
        file = [file]
        images = load_images(
            file,
            size=518,
            verbose=True,
            crop=False,
            patch_size=14,
        )

        images = collate_with_cat([tuple(images)])
        images = torch.stack([view["img"] for view in images], dim=1)
        images = ImgDust3r2Stream3r(images).to(device)

        with torch.no_grad():
            predictions = model(images)

        depth_map = predictions['depth'][0,0].squeeze(-1).cpu()

        if save_dir is not None:
            # save the depth map to the save_dir as npy
            np.save(
                f"{save_dir}/{file[0].split('/')[-1].replace('.png','depth.npy')}",
                depth_map.cpu().numpy(),
            )
            depth_map = colorize_depth(depth_map)
            cv2.imwrite(
                f"{save_dir}/{file[0].split('/')[-1].replace('.png','depth.jpg')}",
                depth_map,
            )


def main():
    args = get_args_parser()
    args = args.parse_args()

    if args.eval_dataset == "sintel":
        args.full_seq = True
    else:
        args.full_seq = False

    model = STream3R.from_pretrained("yslan/STream3R").to(args.device)
    model.eval()

    eval_mono_depth_estimation(args, model, args.device)


if __name__ == "__main__":
    main()
