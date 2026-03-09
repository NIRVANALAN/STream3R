# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# Main encoder/decoder blocks
# --------------------------------------------------------
# References:
# timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/helpers.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/mlp.py
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/patch_embed.py

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

try:
    from torch.nn.attention.flex_attention import flex_attention
    FLEXATTN_AVAILABLE = True
except ImportError:
    print("FlexAttention is not available")
    FLEXATTN_AVAILABLE = False

from itertools import repeat
import collections.abc

from typing import Optional, Callable

try:
    import xformers
    import xformers.ops
    from xformers.ops import memory_efficient_attention
    from xformers.ops import MemoryEfficientAttentionFlashAttentionOp, MemoryEfficientAttentionCutlassOp
    # from xformers.ops import RMSNorm

    XFORMERS_AVAILABLE = True
except ImportError:
    # logger.warning("xFormers not available")
    print("xFormers not available")
    XFORMERS_AVAILABLE = False

assert (XFORMERS_AVAILABLE)

try:
    from apex.normalization import fused_layer_norm_cuda  # check whether cuda compiled
    from apex.normalization import FusedRMSNorm as RMSNorm  # ! requires compilation
except:
    from src.model.norm import RMSNorm

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x,
              drop_prob: float = 0.,
              training: bool = False,
              scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 bias=True,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 rope=None,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(
                head_dim,
                elementwise_affine=True) if qk_norm else nn.Identity()
            self.k_norm = RMSNorm(
                head_dim,
                elementwise_affine=True) if qk_norm else nn.Identity()

    def forward(self, x, xpos):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)

        if self.rope is not None:
            with torch.amp.autocast(device_type="cuda", enabled=False):
                q = self.rope(q, xpos)
                k = self.rope(k, xpos)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # ! mismatched. debug later.

        # q = q.transpose(1,2) # for xformer
        # k = k.transpose(1,2)
        # v = v.transpose(1,2)
        # attn_output = xformers.ops.memory_efficient_attention(q, k, v, scale=self.scale, attn_bias=None)
        # x2 = attn_output.reshape(B, N, C)
        # (Pdb) p (x2-x).max()
        # tensor(0.0014, device='cuda:0'

        # st()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):

    def forward(self, x, xpos):
        if not XFORMERS_AVAILABLE:
            return super().forward(x, xpos)

        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        # q,k,v = qkv.unbind(2)  # make torchscript happy (cannot use tensor as tuple)

        if self.rope is not None:  # rope only supports fp32
            dtype = q.dtype
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                q = self.rope(q.float(), xpos).to(dtype)
                k = self.rope(k.float(), xpos).to(dtype)

        q = q.transpose(1, 2)  # for xformer
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.qk_norm:
            q, k = self.q_norm(q.contiguous()), self.k_norm(k.contiguous())

        # x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=MemoryEfficientAttentionFlashAttentionOp).reshape(B, N, C)
        x = xformers.ops.memory_efficient_attention(q, k, v,
                                                    attn_bias=None).reshape(
                                                        B, N, C)

        # Reshape and project the output
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossAttention(nn.Module):

    def __init__(self,
                 dim,
                 rope=None,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 use_xformers=False,
                 qk_norm=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.qk_norm = qk_norm

        if qk_norm:
            self.q_norm = RMSNorm(
                head_dim,
                elementwise_affine=True) if qk_norm else nn.Identity()
            self.k_norm = RMSNorm(
                head_dim,
                elementwise_affine=True) if qk_norm else nn.Identity()

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope
        self.use_xformers = use_xformers

    def forward(self, query, key, value, qpos, kpos):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, self.num_heads,
                                    C // self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)  # 3005

        # q = q.transpose(1,2) # for xformer
        # k = k.transpose(1,2)
        # v = v.transpose(1,2)

        # x2 = memory_efficient_attention(q, k, v, scale=self.scale).reshape([B, Nq, C])
        # st()
        # (Pdb) p (x - x2).max()
        # tensor(0.0005, device='cuda:0')

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffCrossAttention(CrossAttention):
    # for cross attention, where context serves as k and v
    def __init__(self,
                 dim,
                 rope=None,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0,
                 qk_norm=False):
        super().__init__(dim, rope, num_heads, qkv_bias, attn_drop, proj_drop)

    def forward(self, query, key, value, qpos, kpos):
        if not XFORMERS_AVAILABLE:
            # assert attn_bias is None, "xFormers is required for nested tensors usage"
            return super().forward(query, key, value, qpos, kpos)

        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, self.num_heads,
                                    C // self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        if self.qk_norm:
            q, k = self.q_norm(q.contiguous()), self.k_norm(k.contiguous())

        if self.rope is not None:
            dtype = q.dtype
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                q = self.rope(q.float(), qpos).to(dtype)
                k = self.rope(k.float(), kpos).to(dtype)

        # ! is this required? rope copmat.
        q = q.transpose(1, 2)  # for xformer
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # x = memory_efficient_attention(q, k, v, scale=self.scale, op=MemoryEfficientAttentionFlashAttentionOp).reshape([B, Nq, C]) # memory, 3303
        x = memory_efficient_attention(q, k, v, scale=self.scale).reshape(
            [B, Nq, C])  # memory, 3303

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CausalFlexAttention(nn.Module):
    """Memory efficient attention with causal masking using PyTorch FlexAttention.
    Works with input shape [B, V, L, C] where:
    - B is batch size
    - V is number of views/frames
    - L is number of tokens per view
    - C is embedding dimension
    
    For views 0 and 1: allows bidirectional attention between them
    For views 2 and beyond: enforces causal masking (each view can only attend to itself and previous views)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Separate projections for Q, K, V like in CrossAttention
        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope
        self.qk_norm = qk_norm

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True)

    def _create_causal_mask_fn(self, num_tokens_per_view: int) -> Callable:
        """Create a causal mask function for FlexAttention that masks at view level.
        
        The function must take exactly 5 arguments as required by flex_attention:
        - score: A scalar tensor representing the attention score
        - batch: Batch index tensor
        - head: Head index tensor
        - q_idx: Query token index tensor
        - k_idx: Key/value token index tensor

        For views 0 and 1: implements cross attention where:
        - View 0 can only attend to View 1
        - View 1 can only attend to View 0
        For views 2+: enforces causal masking (each view can only attend to previous views)
        """

        def causal_mask(score, batch, head, q_idx, k_idx):
            # Convert token indices to view indices by integer division
            q_view = q_idx // num_tokens_per_view
            kv_view = k_idx // num_tokens_per_view

            # For views 0 and 1: implement cross attention
            # View 0 can only attend to View 1, and View 1 can only attend to View 0
            cross_attention = ((q_view == 0) & (kv_view == 1)) | ((q_view == 1) & (kv_view == 0))

            # For views 2+: enforce causal masking
            later_views = (q_view > 1) | (kv_view > 1)
            causal_ok = q_view >= kv_view

            # Combine the conditions:
            # 1. If views are 0 and 1, allow cross attention (each view attends to the other)
            # 2. If either view is > 1, enforce causal masking
            mask = torch.where(
                cross_attention | (later_views & causal_ok),
                score,
                float('-inf')
            )

            return mask

        return causal_mask

    def forward(self,
                x: torch.Tensor,
                xpos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, V, L, C = x.shape

        # Generate Q, K, V projections separately
        q = self.projq(x).reshape(B, V*L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.projk(x).reshape(B, V*L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.projv(x).reshape(B, V*L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply RoPE if available
        if self.rope is not None and xpos is not None:
            # RoPE typically works better in fp32
            dtype = q.dtype
            with torch.autocast(device_type="cuda",
                                dtype=torch.float32,
                                enabled=False):
                q_float = q.float()
                k_float = k.float()
                q = self.rope(q_float, xpos).to(dtype)
                k = self.rope(k_float, xpos).to(dtype)

        # Apply QK normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Create causal mask function with current L
        causal_mask = self._create_causal_mask_fn(L)

        # Apply causal attention using FlexAttention
        # FlexAttention expects (batch, num_heads, seq_len, head_dim)
        attn_output = flex_attention(
            q,
            k,
            v,
            score_mod=causal_mask,
            enable_gqa=False  # Set to True if using grouped-query attention
        )

        # Dropout on attention weights (if needed, can be applied in score_mod)
        if self.training and self.attn_drop.p > 0:
            attn_output = self.attn_drop(attn_output)

        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).reshape(B, V, L, C)

        # Final projection
        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x


class CausalMemEffAttention(MemEffAttention):
    """Memory efficient attention with causal masking between frames."""

    def forward(self, x, xpos):
        if not XFORMERS_AVAILABLE:
            return super().forward(x, xpos)

        B, N, C = x.shape
        # Store number of tokens per frame for masking
        self.num_tokens_per_frame = N // self.num_frames

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]

        if self.rope is not None:  # rope only supports fp32
            dtype = q.dtype
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                q = self.rope(q.float(), xpos).to(dtype)
                k = self.rope(k.float(), xpos).to(dtype)

        q = q.transpose(1, 2)  # for xformer
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.qk_norm:
            q, k = self.q_norm(q.contiguous()), self.k_norm(k.contiguous())

        # Create frame-level causal mask
        frame_mask = torch.zeros(N, N, device=x.device, dtype=torch.bool)
        for i in range(self.num_frames):
            for j in range(i + 1):
                start_q = i * self.num_tokens_per_frame
                end_q = (i + 1) * self.num_tokens_per_frame
                start_k = j * self.num_tokens_per_frame
                end_k = (j + 1) * self.num_tokens_per_frame
                frame_mask[start_q:end_q, start_k:end_k] = True

        # Apply causal attention
        x = xformers.ops.memory_efficient_attention(q, k, v,
                                                    attn_bias=frame_mask).reshape(
                                                        B, N, C)

        # Reshape and project the output
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class EnhancedCausalFlexAttention(CausalFlexAttention):
    """Enhanced version with additional features like sliding window attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        window_size: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        self.window_size = window_size
        super().__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop,
                         proj_drop, rope)

        # Override causal mask if using sliding window
        if window_size is not None:
            self.causal_mask = self._create_sliding_window_mask_fn()

    def _create_sliding_window_mask_fn(self) -> Callable:
        """Create a sliding window causal mask function."""

        def sliding_window_causal_mask(b, h, q_idx, kv_idx):
            # Causal constraint: can only attend to current and previous positions
            causal_ok = q_idx >= kv_idx

            # Window constraint: only attend within window_size
            if self.window_size is not None:
                window_ok = (q_idx - kv_idx) <= self.window_size
                return causal_ok & window_ok
            return causal_ok

        return sliding_window_causal_mask


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 rope=None,
                 qk_norm=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn = MemEffAttention(dim,
                                    rope=rope,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    attn_drop=attn_drop,
                                    proj_drop=drop,
                                    qk_norm=qk_norm)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x, xpos):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 norm_mem=True,
                 rope=None,
                 qk_norm=False,
                 causal=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if causal:
            self.attn = CausalMemEffAttention(dim,
                                              rope=rope,
                                              num_heads=num_heads,
                                              qkv_bias=qkv_bias,
                                              attn_drop=attn_drop,
                                              proj_drop=drop,
                                              qk_norm=qk_norm)

            if FLEXATTN_AVAILABLE:
                causal_blk = CausalFlexAttention
            else:
                causal_blk = CausalMemEffAttention

            self.cross_attn = causal_blk(dim,
                                         rope=rope,
                                         num_heads=num_heads,
                                         qkv_bias=qkv_bias,
                                         attn_drop=attn_drop,
                                         proj_drop=drop,
                                         qk_norm=qk_norm)
        else:
            self.attn = MemEffAttention(dim,
                                        rope=rope,
                                        num_heads=num_heads,
                                        qkv_bias=qkv_bias,
                                        attn_drop=attn_drop,
                                        proj_drop=drop,
                                        qk_norm=qk_norm)
            self.cross_attn = MemEffCrossAttention(dim,
                                                   rope=rope,
                                                   num_heads=num_heads,
                                                   qkv_bias=qkv_bias,
                                                   attn_drop=attn_drop,
                                                   proj_drop=drop,
                                                   qk_norm=qk_norm)

        # st()
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()

    def forward(self, x, y, xpos, ypos, has_sa=True):
        if has_sa:
            x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        y_ = self.norm_y(y)
        x = x + self.drop_path(
            self.cross_attn(self.norm2(x), y_, y_, xpos, ypos))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x, y



# ! R and W block
# https://github.com/facebookresearch/PointInfinity/blob/main/modules.py


class Read_Block(nn.Module):

    def __init__(self,
                 z_dim,
                 x_dim,
                 num_heads=16,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_x = norm_layer(x_dim)
        self.norm_z1 = norm_layer(z_dim)
        assert z_dim == x_dim
        self.attn = MemEffCrossAttention(z_dim,
                                         rope=None,
                                         num_heads=num_heads,
                                         qkv_bias=qkv_bias,
                                         attn_drop=attn_drop,
                                         proj_drop=drop,
                                         qk_norm=True)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm_z2 = norm_layer(z_dim)
        mlp_hidden_dim = int(z_dim * mlp_ratio)
        self.mlp = Mlp(in_features=z_dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, z, x, xpos=None, ypos=None):
        x_ = self.norm_x(x)
        z = z + self.drop_path(self.attn(self.norm_z1(z), x_, x_, xpos, ypos))
        z = z + self.drop_path(self.mlp(self.norm_z2(z)))
        return z


class Write_Block(nn.Module):

    def __init__(self,
                 z_dim,
                 x_dim,
                 num_heads=16,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_z = norm_layer(z_dim)
        self.norm_x1 = norm_layer(x_dim)
        assert z_dim == x_dim
        self.attn = MemEffCrossAttention(x_dim,
                                         rope=None,
                                         num_heads=num_heads,
                                         qkv_bias=qkv_bias,
                                         attn_drop=attn_drop,
                                         proj_drop=drop,
                                         qk_norm=True)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm_x2 = norm_layer(x_dim)
        mlp_hidden_dim = int(x_dim * mlp_ratio)
        self.mlp = Mlp(in_features=x_dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, z, x, xpos=None, ypos=None):
        z_ = self.norm_z(z)
        x = x + self.drop_path(self.attn(self.norm_x1(x), z_, z_, xpos, ypos))
        x = x + self.drop_path(self.mlp(self.norm_x2(x)))
        return x


# patch embedding
class PositionGetter(object):
    """ return positions of patches """

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        if not (h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h, w] = torch.cartesian_prod(y,
                                                              x)  # (h, w, 2)
        pos = self.cache_positions[h, w].view(1, h * w, 2).expand(b, -1,
                                                                  2).clone()
        return pos


class PatchEmbed(nn.Module):
    """ just adding _init_weights + position getter compared to timm.models.layers.patch_embed.PatchEmbed"""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.position_getter = PositionGetter()

    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(
            H == self.img_size[0],
            f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        )
        torch._assert(
            W == self.img_size[1],
            f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        )
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos

    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))


class CausalPyTorchAttention(nn.Module):
    """Memory efficient attention with causal masking using PyTorch's native attention.
    Works with input shape [B, V, L, C] where:
    - B is batch size
    - V is number of views/frames
    - L is number of tokens per view
    - C is embedding dimension
    
    For views 0 and 1: implements cross attention where:
    - View 0 can only attend to View 1
    - View 1 can only attend to View 0
    For views 2 and beyond: enforces causal masking (each view can only attend to previous views)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        rope: Optional[nn.Module] = None,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Separate projections for Q, K, V like in CrossAttention
        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope
        self.qk_norm = qk_norm

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim, elementwise_affine=True)
            self.k_norm = RMSNorm(self.head_dim, elementwise_affine=True)

    def _create_causal_mask(self, B: int, V: int, L: int, device: torch.device) -> torch.Tensor:
        """Create a causal mask tensor for attention.
        
        Args:
            B: Batch size
            V: Number of views
            L: Number of tokens per view
            device: Device to create the mask on
            
        Returns:
            Float mask tensor of shape [B, H, V*L, V*L] where:
            - 0.0 indicates positions that can be attended to
            - -inf indicates positions that should be masked
            H is the number of attention heads
        """
        # Create base mask for all positions (without batch and head dimensions)
        mask = torch.zeros(V*L, V*L, device=device, dtype=torch.float32)
        
        # For each query view
        for v_q in range(V):
            # For each key view
            for v_k in range(V):
                # Get the token ranges for these views
                q_start = v_q * L
                q_end = (v_q + 1) * L
                k_start = v_k * L
                k_end = (v_k + 1) * L
                
                # For views 0 and 1: allow bidirectional attention
                if v_q <= 1 and v_k <= 1:
                    # Allow attention in both directions between views 0 and 1
                    mask[q_start:q_end, k_start:k_end] = 0.0
                # For views 2+: enforce causal masking
                else:
                    # Allow attention only to current and previous views
                    if v_q >= v_k:
                        mask[q_start:q_end, k_start:k_end] = 0.0
                    else:
                        mask[q_start:q_end, k_start:k_end] = float('-inf')
        
        # Expand to include batch and head dimensions
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
        
        return mask

    def forward(self,
                x: torch.Tensor,
                xpos: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, V, L, C = x.shape

        # Generate Q, K, V projections separately
        q = self.projq(x).reshape(B, V*L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.projk(x).reshape(B, V*L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.projv(x).reshape(B, V*L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply RoPE if available
        if self.rope is not None and xpos is not None:
            # RoPE typically works better in fp32
            dtype = q.dtype
            with torch.autocast(device_type="cuda",
                                dtype=torch.float32,
                                enabled=False):
                q_float = q.float()
                k_float = k.float()
                q = self.rope(q_float, xpos).to(dtype)
                k = self.rope(k_float, xpos).to(dtype)

        # Apply QK normalization if enabled
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Create causal mask
        attn_bias = self._create_causal_mask(B, V, L, x.device)

        # Debug prints
        # print("\nDebug PyTorch attention:")
        # print(f"Input shape: {x.shape}")
        # print(f"Q shape: {q.shape}")
        # print(f"K shape: {k.shape}")
        # print(f"V shape: {v.shape}")
        # print(f"Mask shape: {attn_bias.shape}")
        # print(f"Sample mask values:\n{attn_bias[:5, :5]}")
        # st()
        # np.save('attn_bias.npy', attn_bias[0][0].detach().cpu().numpy())

        # Apply attention using PyTorch's native attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale
        )

        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).reshape(B, V, L, C)

        # Final projection
        x = self.proj(attn_output)
        x = self.proj_drop(x)

        return x

class DecoderBlockBVL(nn.Module):
    """Decoder block that works with input shape B*V*L*C where:
    - B is batch size
    - V is number of views/frames
    - L is number of tokens per view
    - C is embedding dimension
    
    The block:
    1. Performs self-attention within each view independently
    2. Performs causal attention between views (each view can only attend to previous views)
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 norm_mem=True,
                 rope=None,
                 qk_norm=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()
        
        # Self attention within views
        self.attn = MemEffAttention(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm
        )

        # if FLEXATTN_AVAILABLE:
        #     causal_blk = CausalFlexAttention
        # else:
        causal_blk = CausalPyTorchAttention
        
        # Causal attention between views
        self.cross_attn = causal_blk(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, x, xpos=None):
        """
        Args:
            x: Input tensor of shape [B, V, L, C] where:
               - B is batch size
               - V is number of views/frames
               - L is number of tokens per view
               - C is embedding dimension
            xpos: Optional positional embeddings
        """
        B, V, L, C = x.shape
        
        # First apply self-attention within each view
        # Reshape to [B*V, L, C] for batch processing
        x_reshaped = x.reshape(B*V, L, C)
        
        # Process all views in one batch
        x_reshaped = self.norm1(x_reshaped)
        x_reshaped = x_reshaped + self.drop_path(self.attn(x_reshaped, xpos))
        
        # Reshape back to [B, V, L, C]
        x = x_reshaped.view(B, V, L, C)
        
        # Then apply causal attention between views
        x = self.norm2(x)
        x = x + self.drop_path(self.cross_attn(x, xpos))
        # st()
        
        # Final MLP
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        
        return x

