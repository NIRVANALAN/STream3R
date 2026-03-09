import logging
from typing import List, Dict

import math
import torch
from torch import nn as nn
import torch.nn.functional as F


_logger = logging.getLogger(__name__)


def resample_patch_embed(
        patch_embed,
        new_size: List[int],
        interpolation: str = 'bicubic',
        antialias: bool = True,
        verbose: bool = False,
):
    """Resample the weights of the patch embedding kernel to target resolution.
    We resample the patch embedding kernel by approximately inverting the effect
    of patch resizing.

    Code based on:
      https://github.com/google-research/big_vision/blob/b00544b81f8694488d5f36295aeb7972f3755ffe/big_vision/models/proj/flexi/vit.py

    With this resizing, we can for example load a B/8 filter into a B/16 model
    and, on 2x larger input image, the result will match.

    Args:
        patch_embed: original parameter to be resized.
        new_size (tuple(int, int): target shape (height, width)-only.
        interpolation (str): interpolation for resize
        antialias (bool): use anti-aliasing filter in resize
        verbose (bool): log operation
    Returns:
        Resized patch embedding kernel.
    """
    import numpy as np
    try:
        import functorch
        vmap = functorch.vmap
    except ImportError:
        if hasattr(torch, 'vmap'):
            vmap = torch.vmap
        else:
            assert False, "functorch or a version of torch with vmap is required for FlexiViT resizing."

    assert len(patch_embed.shape) == 4, "Four dimensions expected"
    assert len(new_size) == 2, "New shape should only be hw"
    old_size = patch_embed.shape[-2:]
    if tuple(old_size) == tuple(new_size):
        return patch_embed

    if verbose:
        _logger.info(f"Resize patch embedding {patch_embed.shape} to {new_size}, w/ {interpolation} interpolation.")

    def resize(x_np, _new_size):
        x_tf = torch.Tensor(x_np)[None, None, ...]
        x_upsampled = F.interpolate(
            x_tf, size=_new_size, mode=interpolation, antialias=antialias)[0, 0, ...].numpy()
        return x_upsampled

    def get_resize_mat(_old_size, _new_size):
        mat = []
        for i in range(np.prod(_old_size)):
            basis_vec = np.zeros(_old_size)
            basis_vec[np.unravel_index(i, _old_size)] = 1.
            mat.append(resize(basis_vec, _new_size).reshape(-1))
        return np.stack(mat).T

    resize_mat = get_resize_mat(old_size, new_size)
    resize_mat_pinv = torch.tensor(np.linalg.pinv(resize_mat.T), device=patch_embed.device)

    def resample_kernel(kernel):
        resampled_kernel = resize_mat_pinv @ kernel.reshape(-1)
        return resampled_kernel.reshape(new_size)

    v_resample_kernel = vmap(vmap(resample_kernel, 0, 0), 1, 1)
    orig_dtype = patch_embed.dtype
    patch_embed = patch_embed.float()
    patch_embed = v_resample_kernel(patch_embed)
    patch_embed = patch_embed.to(orig_dtype)
    return patch_embed


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))

            # instead of assigning the same weight to all channels, we can assign higher weight for original RGB channels
            # conv_weight[:, :3, :, :] = conv_weight[:, :3, :, :] * 0.5
            # conv_weight[:, 3:, :, :] = conv_weight[:, 3:, :, :] * 0.5 * (3 / float(in_chans - 3))

    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def adapt_head_conv(conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape

    conv_weight_new = torch.chunk(conv_weight, 6, dim=1)
    conv_weight_new = [conv_weight_new.mean(dim=1, keepdim=True) for conv_weight_new in conv_weight_new]
    conv_weight_new = torch.cat(conv_weight_new, dim=1) * 0.5
    conv_weight = torch.cat([conv_weight, conv_weight_new], dim=1)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def adapt_linear(conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I = conv_weight.shape

    conv_weight_new = torch.tensor_split(conv_weight, 81, dim=1)
    conv_weight_new = [conv_weight_new.mean(dim=1, keepdim=True) for conv_weight_new in conv_weight_new]
    conv_weight_new = torch.cat(conv_weight_new, dim=1)
    # conv_weight = torch.cat([conv_weight, conv_weight_new], dim=1)
    conv_weight = torch.cat([conv_weight * 0.5, conv_weight_new * 0.5], dim=1)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: nn.Module,
        interpolation: str = 'bicubic',
        antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    # state_dict = state_dict.get('model', state_dict)
    # state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.backbone.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.backbone.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
            if v.shape[1] != I:
                v = adapt_input_conv(I, v)
        # elif 'downstream_head1.dpt.head.0.weight' in k or 'downstream_head2.dpt.head.0.weight' in k:
        #     v = adapt_head_conv(v)

        elif 'decoder_embed.weight' in k:
            O, I = model.backbone.decoder_embed.weight.shape
            if v.shape[1] != I:
                v = adapt_linear(v)

        out_dict[k] = v

    # add prefix to make our model happy
    prefix = 'backbone.'
    out_dict = {prefix + k if 'downstream_head' not in k else k: v for k, v in out_dict.items()}

    # # remove the conf head weights
    out_dict['downstream_head1.dpt.head.4.weight'] = out_dict['downstream_head1.dpt.head.4.weight'][0:3]
    out_dict['downstream_head1.dpt.head.4.bias'] = out_dict['downstream_head1.dpt.head.4.bias'][0:3]
    out_dict['downstream_head2.dpt.head.4.weight'] = out_dict['downstream_head2.dpt.head.4.weight'][0:3]
    out_dict['downstream_head2.dpt.head.4.bias'] = out_dict['downstream_head2.dpt.head.4.bias'][0:3]

    return out_dict
