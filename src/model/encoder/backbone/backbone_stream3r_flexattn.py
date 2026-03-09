from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from torch import nn
from torch.utils.checkpoint import checkpoint

from .croco.blocks import DecoderBlockBVL, Mlp
from .croco.croco import CroCoNet
from .croco.misc import fill_default_args, freeze_all_params, transpose_to_landscape, is_symmetrized, interleave, \
    make_batch_symmetric
from .croco.patch_embed import get_patch_embed
from .croco.pos_embed import get_1d_rotary_pos_embed
from .backbone import Backbone
from ....geometry.camera_emb import get_intrinsic_embedding

from pdb import set_trace as st
from diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid

from .backbone_croco import BackboneCrocoCfg, croco_params
from .backbone_croco_multiview import AsymmetricCroCoMulti
from .backbone_croco_multiview_cut3r import AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn_frame0_as_memory

from .backbone_croco_must3r import Must3r_reimpl
from .backbone_croco_must3r_noinj import Must3r_reimpl_noinj

from torch.nn.attention.flex_attention import flex_attention


class Stream3r_flexattn(Must3r_reimpl_noinj):
    """Stream3r implementation using true causal attention."""

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        super().__init__(cfg, d_in)
        """changelog:
        DONE:
        1. switch to true causal attention training
        2. use DecoderBlockBVL for view-level self-attention and causal attention
        """

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads,
                     dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        """Set up decoder with DecoderBlockBVL."""
        super()._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec)

        # Create decoder blocks
        # st()
        self.dec_blocks = nn.ModuleList([
            DecoderBlockBVL(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer,
                rope=self.decoder_rope,
                qk_norm=True
                # qk_norm=False, # align with the original implementation
            ) for _ in range(dec_depth)
        ])
        
        # Final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def _forward_decoder_step(self, feat, pos, ret_state=False):
        '''Implements true parallel causal attention.
        Each view can only attend to previous views and itself.
        '''
        ress = []
        v = feat.shape[1]
        assert v >= 2  # this model works on at least 2 views

        # Add learnable tokens to all features
        # feat[:, 0] gets frame0 token, all others get B token
        feat_with_tokens = []
        for i in range(v):
            if i == 0:
                feat_with_tokens.append(feat[:, i] +
                                        self.learnable_frame0_token)
            else:
                feat_with_tokens.append(feat[:, i] + self.learnable_B_token)

        # Stack all features: [B, V, L, C]
        all_feat = torch.stack(feat_with_tokens, dim=1)

        # Apply causal attention through decoder blocks
        final_output = [all_feat]

        for idx, blk in enumerate(self.dec_blocks):
            if (self.gradient_checkpointing and self.training
                    and torch.is_grad_enabled()):

                def causal_block_forward(x):
                    return blk(x, pos)

                output = checkpoint(causal_block_forward,
                                    final_output[-1],
                                    use_reentrant=False)
            else:
                output = blk(final_output[-1], pos)

            final_output.append(output)

        # Apply final norm
        # st()
        final_output[-1] = self.dec_norm(final_output[-1])

        # Split into individual view outputs
        for i in range(v):
            view_output = []
            for layer_idx in range(len(final_output)):
                view_output.append(final_output[layer_idx][:, i])
            ress.append(view_output)

        # For compatibility, return memory as concatenated features
        memory = [
            torch.cat([ress[i][j] for i in range(v)], dim=1)
            for j in range(len(ress[0]))
        ]

        return ress, memory

    def _rnn_decoder_causal(self, mem_feat, f_i, *args, **kwargs):
        # This method is no longer needed with parallel causal attention
        # but keeping for compatibility - it should not be called
        raise NotImplementedError(
            "_rnn_decoder_causal is replaced by parallel causal attention in _forward_decoder_step"
        )
