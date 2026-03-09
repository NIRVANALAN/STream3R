import torch
from jaxtyping import Int
from torch import Tensor
import numpy as np


def add_third_context_index(
    indices: Int[Tensor, "*batch 2"]
) -> Int[Tensor, "*batch 3"]:
    left, right = indices.unbind(dim=-1)
    return torch.stack((left, (left + right) // 2, right), dim=-1)


def add_k_context_indices(
    indices: Int[Tensor, "*batch 2"],
    context_views: Int,
    middle_context_view=True,
) -> Int[Tensor, "*batch 3"]:
    left, right = indices.unbind(dim=-1)
    # return torch.stack((left, (left + right) // 2, right), dim=-1)
    # return torch.arange(left, right, t)
    ctx_views = np.linspace(left, right, num=context_views, endpoint=True, dtype=np.uint64).tolist()
    if middle_context_view: # move the middle frame as the canonical (first) frame.
        middle_idx = len(ctx_views)//2
        ctx_views = [ctx_views[middle_idx]] + ctx_views[:middle_idx] + ctx_views[middle_idx+1:]
    ctx_views = torch.Tensor(ctx_views).to(torch.int64)
    return ctx_views
