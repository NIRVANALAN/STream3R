from einops import rearrange

from .projection import sample_image_grid, get_local_rays
from ..misc.sht import rsh_cart_2, rsh_cart_4, rsh_cart_6, rsh_cart_8


def get_intrinsic_embedding(context, degree=0, downsample=1, merge_hw=False):
    assert degree in [0, 2, 4, 8]

    b, v, _, h, w = context["image"].shape
    device = context["image"].device
    tgt_h, tgt_w = h // downsample, w // downsample
    xy_ray, _ = sample_image_grid((tgt_h, tgt_w), device)
    xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)  # [b, v, h, w, 2]
    directions = get_local_rays(xy_ray, rearrange(context["intrinsics"], "b v i j -> b v () () i j"),)

    if degree == 2:
        directions = rsh_cart_2(directions)
    elif degree == 4:
        directions = rsh_cart_4(directions)
    elif degree == 8:
        directions = rsh_cart_8(directions)

    if merge_hw:
        directions = rearrange(directions, "b v h w d -> b v (h w) d")
    else:
        directions = rearrange(directions, "b v h w d -> b v d h w")

    return directions
