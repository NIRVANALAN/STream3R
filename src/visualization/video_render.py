from typing import Protocol, runtime_checkable

import cv2
import torch
from einops import repeat, pack
from jaxtyping import Float
from torch import Tensor

from .camera_trajectory.interpolation import interpolate_extrinsics, interpolate_intrinsics
from .camera_trajectory.wobble import generate_wobble, generate_wobble_transformation
from .layout import vcat
from ..dataset.types import BatchedExample
from ..misc.image_io import save_video
from ..misc.utils import vis_depth_map
from ..model.decoder import Decoder
from ..model.types import Gaussians


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


def render_video_wobble(
        gaussians: Gaussians,
        decoder: Decoder,
        batch: BatchedExample,
        num_frames: int = 60,
        smooth: bool = True,
        loop_reverse: bool = True,
        add_depth: bool = False,
) -> Tensor:
    # Two views are needed to get the wobble radius，use the first and the last view
    _, v, _, _ = batch["context"]["extrinsics"].shape

    def trajectory_fn(t):
        origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
        origin_b = batch["context"]["extrinsics"][:, -1, :3, 3]
        delta = (origin_a - origin_b).norm(dim=-1)
        extrinsics = generate_wobble(
            batch["context"]["extrinsics"][:, 0],
            delta * 0.25,
            t,
        )
        intrinsics = repeat(
            batch["context"]["intrinsics"][:, 0],
            "b i j -> b v i j",
            v=t.shape[0],
        )
        return extrinsics, intrinsics

    return render_video_generic(gaussians, decoder, batch, trajectory_fn, num_frames, smooth, loop_reverse, add_depth)


def render_video_interpolation(
        gaussians: Gaussians,
        decoder: Decoder,
        batch: BatchedExample,
        num_frames: int = 60,
        smooth: bool = True,
        loop_reverse: bool = True,
        add_depth: bool = False,
) -> Tensor:
    _, v, _, _ = batch["context"]["extrinsics"].shape

    def trajectory_fn(t):
        extrinsics = interpolate_extrinsics(
            batch["context"]["extrinsics"][0, 0],
            batch["context"]["extrinsics"][0, -1],
            t,
        )
        intrinsics = interpolate_intrinsics(
            batch["context"]["intrinsics"][0, 0],
            batch["context"]["intrinsics"][0, -1],
            t,
        )
        return extrinsics[None], intrinsics[None]

    return render_video_generic(gaussians, decoder, batch, trajectory_fn, num_frames, smooth, loop_reverse, add_depth)


def render_video_interpolation_exaggerated(
        gaussians: Gaussians,
        decoder: Decoder,
        batch: BatchedExample,
        num_frames: int = 300,
        smooth: bool = False,
        loop_reverse: bool = False,
        add_depth: bool = False,
) -> Tensor:
    # Two views are needed to get the wobble radius.
    _, v, _, _ = batch["context"]["extrinsics"].shape

    def trajectory_fn(t):
        origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
        origin_b = batch["context"]["extrinsics"][:, -1, :3, 3]
        delta = (origin_a - origin_b).norm(dim=-1)
        tf = generate_wobble_transformation(
            delta * 0.5,
            t,
            5,
            scale_radius_with_t=False,
        )
        extrinsics = interpolate_extrinsics(
            batch["context"]["extrinsics"][0, 0],
            batch["context"]["extrinsics"][0, -1],
            t * 5 - 2,
        )
        intrinsics = interpolate_intrinsics(
            batch["context"]["intrinsics"][0, 0],
            batch["context"]["extrinsics"][0, -1],
            t * 5 - 2,
        )
        return extrinsics @ tf, intrinsics[None]

    return render_video_generic(gaussians, decoder, batch, trajectory_fn, num_frames, smooth, loop_reverse, add_depth)


def render_video_generic(
        gaussians: Gaussians,
        decoder: Decoder,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
        add_depth: bool = False,
) -> Tensor:
    device = gaussians.means.device

    t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=device)
    if smooth:
        t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

    extrinsics, intrinsics = trajectory_fn(t)

    _, _, _, h, w = batch["context"]["image"].shape

    near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
    far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
    output = decoder.forward(
        gaussians, extrinsics, intrinsics, near, far, (h, w), "depth"
    )
    images = [
        vcat(rgb, depth) if add_depth else rgb
        for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
    ]

    video = torch.stack(images)
    # video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
    if loop_reverse:
        # video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        video = pack([video, video.flip(dims=(0,))[1:-1]], "* c h w")[0]

    return video
