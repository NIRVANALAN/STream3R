import os
import sys
import numpy as np
import torch
import argparse
from accelerate import PartialState
from tqdm import tqdm
from PIL import Image
import imageio.v2 as iio

from stream3r.models.stream3r import STream3R
from stream3r.stream_session import StreamSession
from stream3r.dust3r.utils.image import load_images_for_eval as load_images
from stream3r.dust3r.utils.device import collate_with_cat
from stream3r.utils.utils import ImgDust3r2Stream3r

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from eval.video_depth.metadata import dataset_metadata
from eval.video_depth.utils import colorize


device = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = True

# avoid high cpu usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)
# ===========================================


def save_depth_maps(pts3ds_self, path, conf_self=None, depth_maps=None):
    if depth_maps is None:
        depth_maps = torch.stack([pts3d_self[..., -1] for pts3d_self in pts3ds_self], 0)
    min_depth = depth_maps.min()  # float(torch.quantile(out, 0.01))
    max_depth = depth_maps.max()  # float(torch.quantile(out, 0.99))
    colored_depth = colorize(
        depth_maps,
        cmap_name="Spectral_r",
        range=(min_depth, max_depth),
        append_cbar=True,
    )
    images = []

    if conf_self is not None:
        conf_selfs = torch.concat(conf_self, 0)
        min_conf = torch.log(conf_selfs.min())  # float(torch.quantile(out, 0.01))
        max_conf = torch.log(conf_selfs.max())  # float(torch.quantile(out, 0.99))
        colored_conf = colorize(
            torch.log(conf_selfs),
            cmap_name="jet",
            range=(min_conf, max_conf),
            append_cbar=True,
        )

    for i, depth_map in enumerate(colored_depth):
        # Apply color map to depth map
        img_path = f"{path}/frame_{(i):04d}.png"
        if conf_self is None:
            to_save = (depth_map * 255).detach().cpu().numpy().astype(np.uint8)
        else:
            to_save = torch.cat([depth_map, colored_conf[i]], dim=1)
            to_save = (to_save * 255).detach().cpu().numpy().astype(np.uint8)
        iio.imwrite(img_path, to_save)
        images.append(Image.open(img_path))
        np.save(f"{path}/frame_{(i):04d}.npy", depth_maps[i].detach().cpu().numpy())

    return depth_maps


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="value for outdir",
    )
    parser.add_argument(
        "--no_crop", type=bool, default=True, help="whether to crop input data"
    )

    parser.add_argument(
        "--eval_dataset",
        type=str,
        default="sintel",
        choices=list(dataset_metadata.keys()),
    )
    parser.add_argument("--size", type=int, default="512")

    parser.add_argument(
        "--pose_eval_stride", default=1, type=int, help="stride for pose evaluation"
    )
    parser.add_argument(
        "--full_seq",
        action="store_true",
        default=False,
        help="use full sequence for pose evaluation",
    )
    parser.add_argument(
        "--seq_list",
        nargs="+",
        default=None,
        help="list of sequences for pose evaluation",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="path to the checkpoint directory",
    )
    return parser


def eval_pose_estimation(args, model, save_dir=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    img_path = metadata["img_path"]
    mask_path = metadata["mask_path"]

    ate_mean, rpe_trans_mean, rpe_rot_mean = eval_pose_estimation_dist(
        args, model, save_dir=save_dir, img_path=img_path, mask_path=mask_path)
    return ate_mean, rpe_trans_mean, rpe_rot_mean


def eval_pose_estimation_dist(args,
                              model,
                              img_path,
                              save_dir=None,
                              mask_path=None):
    metadata = dataset_metadata.get(args.eval_dataset)
    model.eval()

    seq_list = args.seq_list

    if seq_list is None:
        if metadata.get("full_seq", False):
            args.full_seq = True
        else:
            seq_list = metadata.get("seq_list", [])
        if args.full_seq:
            seq_list = os.listdir(img_path)
            seq_list = [
                seq for seq in seq_list
                if os.path.isdir(os.path.join(img_path, seq))
            ]
        seq_list = sorted(seq_list)

    if save_dir is None:
        save_dir = args.output_dir

    distributed_state = PartialState()
    model.to(distributed_state.device)
    device = distributed_state.device

    with distributed_state.split_between_processes(seq_list) as seqs:
        error_log_path = f"{save_dir}/_error_log_{distributed_state.process_index}.txt"  # Unique log file per process
        for seq in tqdm(seqs):
            try:
                dir_path = metadata["dir_path_func"](img_path, seq)

                # Handle skip_condition
                skip_condition = metadata.get("skip_condition", None)
                if skip_condition is not None and skip_condition(
                        save_dir, seq):
                    continue

                mask_path_seq_func = metadata.get("mask_path_seq_func",
                                                  lambda mask_path, seq: None)
                mask_path_seq = mask_path_seq_func(mask_path, seq)

                filelist = [
                    os.path.join(dir_path, name)
                    for name in os.listdir(dir_path)
                ]
                filelist.sort()
                filelist = filelist[::args.pose_eval_stride]

                images = load_images(
                    filelist,
                    size=518,
                    verbose=True,
                    crop=False,
                    patch_size=14,
                )

                images = collate_with_cat([tuple(images)])
                images = torch.stack([view["img"] for view in images], dim=1)
                images = ImgDust3r2Stream3r(images).to(device)

                with torch.no_grad():
                    session = StreamSession(model, mode="causal")
                    for i in range(images.shape[1]):
                        image = images[:, i:i+1]
                        predictions = session.forward_stream(image)

                print(
                    f"Finished depth estmation of {len(filelist)} images"
                )

                os.makedirs(f"{save_dir}/{seq}", exist_ok=True)
                save_depth_maps(None,
                                f"{save_dir}/{seq}",
                                conf_self=None, 
                                depth_maps=predictions['depth'].squeeze().cpu())

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(error_log_path, "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                        e) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(error_log_path, "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(
                        f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception
    return None, None, None


def main():
    args = get_args_parser()
    args = args.parse_args()

    if args.eval_dataset == "sintel":
        args.full_seq = True
    else:
        args.full_seq = False
    args.no_crop = True

    model = STream3R.from_pretrained("yslan/STream3R").to(args.device)
    model.eval()

    eval_pose_estimation(args, model, save_dir=args.output_dir)


if __name__ == "__main__":
    main()
