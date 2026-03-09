import torch
import numpy as np
import matplotlib
from colorspacious import cspace_convert
from einops import rearrange
from jaxtyping import Float
from matplotlib import cm
from torch import Tensor


def apply_color_map(
    x: Float[Tensor, " *batch"],
    color_map: str = "inferno",
) -> Float[Tensor, "*batch 3"]:
    cmap = cm.get_cmap(color_map)

    # Convert to NumPy so that Matplotlib color maps can be used.
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]

    # Convert back to the original format.
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(
    image: Float[Tensor, "*batch height width"],
    color_map: str = "inferno",
) -> Float[Tensor, "*batch 3 height with"]:
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")


def apply_color_map_2d(
    x: Float[Tensor, "*#batch"],
    y: Float[Tensor, "*#batch"],
) -> Float[Tensor, "*batch 3"]:
    red = cspace_convert((189, 0, 0), "sRGB255", "CIELab")
    blue = cspace_convert((0, 45, 255), "sRGB255", "CIELab")
    white = cspace_convert((255, 255, 255), "sRGB255", "CIELab")
    x_np = x.detach().clip(min=0, max=1).cpu().numpy()[..., None]
    y_np = y.detach().clip(min=0, max=1).cpu().numpy()[..., None]

    # Interpolate between red and blue on the x axis.
    interpolated = x_np * red + (1 - x_np) * blue

    # Interpolate between color and white on the y axis.
    interpolated = y_np * interpolated + (1 - y_np) * white

    # Convert to RGB.
    rgb = cspace_convert(interpolated, "CIELab", "sRGB1")
    return torch.tensor(rgb, device=x.device, dtype=torch.float32).clip(min=0,
                                                                        max=1)


def colorize_depth_maps(depth_map,
                        min_depth,
                        max_depth,
                        cmap="Spectral",
                        valid_mask=None):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


# https://github.com/microsoft/MoGe/blob/a8c37341bc0325ca99b9d57981cc3bb2bd3e255b/moge/utils/vis.py#L7
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
