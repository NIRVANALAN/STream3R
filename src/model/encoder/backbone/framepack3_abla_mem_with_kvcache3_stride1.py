from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Literal

import torch
from einops import rearrange, repeat
from torch import nn
from torch.utils.checkpoint import checkpoint

from .croco.blocks import DecoderBlock, Mlp, Read_Block, Write_Block
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

from .backbone_croco_framepack import FramePack3R
from .backbone_croco_framepack3 import FramePack3Rv3
# from .backbone_croco_framepack2 import


class FramePack3Rv3_Abla_Mem_with_KVCache3_stride1(FramePack3Rv3):

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        super().__init__(cfg, d_in)
        """changelog:
        TODO:
        1. has a learnable memmory [z]
        2. full autodecoder CA (w/o [z])
        3. use [z] to write back to previous tokens
        4. has a "cutoff" range for the write back, make it 2 for now.
        5. avoid using [f0] as the memory token
        6. if context size > self.cutoff_window, use memory tokens + recent self.cutoff_window instead (infinite)
        """

        # self.cutoff_window_k = 1  # only attend to & update frames within recent k pack
        # self.pack_size = 3

        # ! for ablation
        self.cutoff_window_k = 3 # ablate performance
        self.pack_size = 1

        # # ! be compatible with the previous model
        # for _, param in self.global_3d_feedback_Inj.named_parameters():
        #     param.requires_grad_(True)

    def _rnn_decoder_causal_pack(self, mem_feat, f_nextpack, ress, f_state,
                                 *args, **kwargs):
        # ! perform memory-based attention

        final_output = [f_nextpack + self.learnable_B_token.unsqueeze(1)]
        final_mem_state = [f_state]

        b, v, l, c = f_nextpack.shape

        ress_v = len(ress)
        if ress_v < self.pack_size * self.cutoff_window_k:
        # if True:
            prev_k_pack_indices = list(range(1, ress_v))  # will add [0] later
        else:
            prev_k_pack_indices = list(
                range(
                    ress_v - self.pack_size * self.cutoff_window_k,
                    ress_v,
                ))

        memory_indices = [0] + prev_k_pack_indices

        # st()

        for idx, blk_siamese in enumerate(
                self.dec_blocks):  # ! 12 blocks of decoder.

            # Memory read from incoming frames
            f_state, _ = blk_siamese(
                final_mem_state[-1],  # b l c
                rearrange(final_output[-1], 'b v l c -> b (v l) c'),
                None,
                None,
            )

            # Prepare memory to read from
            kvcache_mem = rearrange(mem_feat[idx],
                                    'b (v l) c -> b v l c',
                                    v=ress_v)[:, memory_indices]
            kvcache_mem = rearrange(
                kvcache_mem,
                'b v l c -> b (v l) c',
            )

            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # First transformer operation
                        f_state, _ = module(
                            inputs[0],  # final_mem_state[-1]
                            inputs[1],  # rearranged final_output[-1]
                            None,
                            None
                        )
                        # Second transformer operation
                        f_nextpack, _ = module(
                            inputs[2],  # rearranged final_output[-1]
                            inputs[3],  # concatenated memory
                            None,
                            None
                        )
                        return f_state, f_nextpack
                    return custom_forward

                f_state, f_nextpack = checkpoint(
                    create_custom_forward(blk_siamese),
                    final_mem_state[-1],
                    rearrange(final_output[-1], 'b v l c -> b (v l) c'),
                    rearrange(final_output[-1], 'b v l c -> b (v l) c'),
                    torch.cat([kvcache_mem, final_mem_state[-1]], dim=1),
                    use_reentrant=False
                )
            else:
                # Memory read from incoming frames
                f_state, _ = blk_siamese(
                    final_mem_state[-1],  # b l c
                    rearrange(final_output[-1], 'b v l c -> b (v l) c'),
                    None,
                    None)

                # New frames read information from memory
                f_nextpack, _ = blk_siamese(
                    rearrange(final_output[-1], 'b v l c -> b (v l) c'),
                    torch.cat([kvcache_mem, final_mem_state[-1]], dim=1), None,
                    None)

            final_output.append(
                rearrange(f_nextpack, 'b (v l) c -> b v l c', b=b, v=v))
            final_mem_state.append(f_state)

        final_output[-1] = self.dec_norm(final_output[-1])

        # decouple final_output into a list of v length lists, each with 13 layers
        decoupled_output = []
        for view_idx in range(v):
            view_features = []
            for layer_idx in range(len(final_output)):
                view_features.append(
                    final_output[layer_idx][:, view_idx])  # [b, l, c]
            decoupled_output.append(
                view_features)  # list of v lists, each with 13 layers

        return decoupled_output, final_mem_state[-1]  # no zip required
