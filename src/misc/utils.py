import torch

from src.visualization.color_map import apply_color_map_to_image


def inverse_normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor.mul(std).add(mean)


# Color-map the result.
def vis_depth_map(result):
    far = result.view(-1)[:16_000_000].quantile(0.99).log()
    try:
        near = result[result > 0][:16_000_000].quantile(0.01).log()
    except:
        print("No valid depth values found.")
        near = torch.zeros_like(far)
    result = result.log()
    result = 1 - (result - near) / (far - near)
    return apply_color_map_to_image(result, "turbo")


def confidence_map(result):
    # far = result.view(-1)[:16_000_000].quantile(0.99).log()
    # try:
    #     near = result[result > 0][:16_000_000].quantile(0.01).log()
    # except:
    #     print("No valid depth values found.")
    #     near = torch.zeros_like(far)
    # result = result.log()
    # result = 1 - (result - near) / (far - near)
    result = result / result.view(-1).max()
    return apply_color_map_to_image(result, "magma")


def get_overlap_tag(overlap):
    if 0.05 <= overlap <= 0.3:
        overlap_tag = "small"
    elif overlap <= 0.55:
        overlap_tag = "medium"
    elif overlap <= 0.8:
        overlap_tag = "large"
    else:
        overlap_tag = "ignore"

    return overlap_tag
