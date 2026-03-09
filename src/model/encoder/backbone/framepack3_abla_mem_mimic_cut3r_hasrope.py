from ast import Not
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

from .croco.pos_embed import get_2d_sincos_pos_embed, RoPE2D


class FramePack3Rv3_Abla_Mem_cut3r_hasrope(FramePack3Rv3):

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        super().__init__(cfg, d_in)
        """changelog:
        """

        # self.cutoff_window_k = 1  # only attend to & update frames within recent k pack
        # self.pack_size = 3

        # ! for ablation
        self.cutoff_window_k = 1
        self.pack_size = 1

        # # ! be compatible with the previous model
        for _, param in self.global_3d_feedback_Inj.named_parameters():
            param.requires_grad_(False)

    # '''
    def _rnn_decoder_initialize(self, f_state, f1, pos_1, pos_fstate, *args, **kwargs):
        # ! initializing causal inference from paired views

        # final_output = [(f1 + self.learnable_frame0_token,
        #                  f2 + self.learnable_B_token)]
        # b = f1.shape[0]
        final_output = [(f_state, f1)]
        # final_output = [(f1+0, f2 + self.learnable_B_token)]

        for idx, blk_siamese in enumerate(
                self.dec_blocks):  # ! 12 blocks of decoder.

            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                def create_custom_forward(module):
                    def custom_forward(inputs, pos_fstate, pos_1):
                        f_state, _ = module(*inputs[::+1], pos_fstate, pos_1)
                        f1, _ = module(*inputs[::-1], pos_1, pos_fstate)
                        return f_state, f1
                    return custom_forward

                f_state, f1 = checkpoint(
                    create_custom_forward(blk_siamese),
                    final_output[-1],
                    pos_fstate,
                    pos_1,
                    use_reentrant=False
                )

            else:
                f_state, _ = blk_siamese(*final_output[-1][::+1], pos_fstate, pos_1)
                f1, _ = blk_siamese(*final_output[-1][::-1], pos_1, pos_fstate)

            # store the result
            final_output.append((f_state, f1))

        # del final_output[1] # duplicate with final_output[0]

        final_output[-1] = (self.dec_norm_for_state(final_output[-1][0]), 
                            self.dec_norm(final_output[-1][1]))

        return zip(*final_output)



    def _forward_decoder_step(
            self,
            # views,
            feat,
            pos,
            # pos_fstate,
            ret_state=False):
        # basically, the state_feat (global memory) stays the same, while the second CA sees all the previous hidden states.
        # state_feat, state_pos = feat[:, 0,].clone(), pos[:, 0,].clone() # remove intrinsics embedding here

        ress = []

        b, v = feat.shape[:2]
        assert v >= 2  # this model works on at least 2 views

        # ! initialize the memory
        # f_state = self.memory_token.repeat(b, 1, 1) # ! gradient will not flow back
        f_state = self.memory_token.expand(b, -1, -1) # ! gradient will flow back
        ress = []


        # add register tokens, though perhaps can be removed.
        feat[:, 0] = feat[:, 0] + self.learnable_frame0_token
        feat[:, 1:] = feat[:, 1:] + self.learnable_B_token.unsqueeze(1)

        if self.state_pe == "2d": # ! default setting here.
            width = int(self.memory_token.shape[1]**0.5)
            width = width + 1 if width % 2 == 1 else width
            pos_fstate = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.memory_token.shape[1])],
                    dtype=pos.dtype,
                    device=pos.device,
                )[None]
                .expand(b, -1, -1)
                .contiguous()
            )
        else:
            raise NotImplementedError(f"state_pe={self.state_pe} not implemented")

        for i in range(v):
            f_state_all, output = self._rnn_decoder_initialize(f_state, feat[:, i], pos[:, i].contiguous(), pos_fstate)
            f_state = f_state_all[-1]
            ress.append(output)
    
        return ress, f_state # , memory
    # '''

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads,
                     dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):

        super()._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads,
                             dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec)
        
        self.dec_norm_for_state = norm_layer(dec_embed_dim)
        # st()
        pass


        pos_embed = 'RoPE100'
        freq = float(pos_embed[len('RoPE'):])
        self.decoder_rope = RoPE2D(freq=freq)

        # transformer for the decoder, state
        # init from self.dec_blocks
        # self.dec_state_blocks = nn.ModuleList([
        #     DecoderBlock(dec_embed_dim,
        #                  dec_num_heads,
        #                  mlp_ratio=mlp_ratio,
        #                  qkv_bias=True,
        #                  norm_layer=norm_layer,
        #                  norm_mem=norm_im2_in_dec,
        #                  rope=self.decoder_rope) for i in range(dec_depth)
        # ])

        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim,
                         dec_num_heads,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=True,
                         norm_layer=norm_layer,
                         norm_mem=norm_im2_in_dec,
                         rope=self.decoder_rope) for i in range(dec_depth)
        ])

        self.state_pe = "2d"
        # self.state_size = self.memory_token.shape[1]
        # st()


        pass


