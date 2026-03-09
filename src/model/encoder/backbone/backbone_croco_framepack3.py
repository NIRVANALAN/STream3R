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
# from .backbone_croco_framepack2 import


class FramePack3Rv3(FramePack3R):

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
        self.cutoff_window_k = 3
        self.pack_size = 1

        # ! be compatible with the previous model
        for _, param in self.global_3d_feedback_Inj.named_parameters():
            param.requires_grad_(True)

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads,
                     dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        super()._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads,
                             dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec)

        self.memory_token = nn.Parameter(
            torch.empty(1, 768, self.dec_embed_dim)
        )  # 768 = tokens for [512, 384] resolution inputs.
        torch.nn.init.normal_(self.memory_token, mean=0.0,
                              std=0.02)  # init as normal distribution

        # ! setup the read and write block to communicate with the (learnable) memory tokens
        # ! directly reuse the existing blocks
        # self.mem_read_blk = Read_Block(z_dim=dec_embed_dim,
        #                                x_dim=dec_embed_dim,
        #                                num_heads=dec_num_heads,
        #                                mlp_ratio=mlp_ratio,
        #                                qkv_bias=True,
        #                                norm_layer=norm_layer)
        # self.mem_write_blk = Write_Block(z_dim=dec_embed_dim,
        #                                  x_dim=dec_embed_dim,
        #                                  num_heads=dec_num_heads,
        #                                  mlp_ratio=mlp_ratio,
        #                                  qkv_bias=True,
        #                                  norm_layer=norm_layer)

    def _decoder(self, feat, pose, extra_embed=None):
        b, v, l, c = feat.shape
        final_output = [feat]  # before projection
        if extra_embed is not None:
            feat = torch.cat((feat, extra_embed), dim=-1)

        # project to decoder dim
        f = rearrange(feat, "b v l c -> (b v) l c")
        f = self.decoder_embed(f)
        f = rearrange(f, "(b v) l c -> b v l c", b=b, v=v).contiguous()

        # ! RNN rollout
        # ! len(rnn_rollout_dec): v
        # ! len(rnn_rollout_dec[0]): 13 (length of decoder + 1)

        rnn_rollout_dec, memory = self._forward_decoder_step(
            f, pose, ret_state=False
        )  #  len(rnn_rollout_dec)==V, len(rnn_rollout_dec[0])==13

        rnn_rollout_dec = [
            torch.stack([dec_tuple[i] for dec_tuple in rnn_rollout_dec], dim=1)
            for i in range(len(rnn_rollout_dec[0]))
        ]  # stack tuple of last layer output to B V L C
        final_output += rnn_rollout_dec

        del final_output[
            1]  # duplicate with final_output[0]. just linear projection.
        
        # st()

        return final_output

    def _dropout_memory_token(self, memory_feat, sample_length=1):
        # ! always drop? Yes.
        if not self.training:
            return memory_feat
        # assert self.memory_dropout > 0

        # ! dynamic dropout rate based on token length
        # if already_merged:
        #     batch_token_length = memory_feat[0].shape[1] /
        # else:
        batch_token_length = memory_feat[0].shape[1] / sample_length
        drop_rate = (batch_token_length -
                     self.token_length_range[0]) * self.memory_dropout[1] + (
                         -batch_token_length +
                         self.token_length_range[1]) * self.memory_dropout[0]
        drop_rate = drop_rate / (self.token_length_range[1] -
                                 self.token_length_range[0])

        assert drop_rate >= self.memory_dropout[
            0] and drop_rate <= self.memory_dropout[1]

        # self.token_length_range

        valid_mask = torch.empty_like(memory_feat[0]).bernoulli(1 - drop_rate)
        # st()
        memory_feat = [valid_mask * mem
                       for mem in memory_feat]  # same mask in each layer
        return memory_feat

    def _rnn_decoder_causal_pack(self, mem_feat, f_nextpack, ress, f_state,
                                 *args, **kwargs):
        # ! perform memory-based attention

        final_output = [f_nextpack + self.learnable_B_token.unsqueeze(1)]
        final_mem_state = [f_state]

        b, v, l, c = f_nextpack.shape
        # st()

        ress_v = len(ress)
        # if ress_v < self.pack_size * self.cutoff_window_k:
        if True:
            prev_k_pack_indices = list(range(1, ress_v))  # will add [0] later
        # else:
        #     prev_k_pack_indices = list(
        #         range(
        #             ress_v - self.pack_size * self.cutoff_window_k,
        #             ress_v,
        #         ))

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

    def _forward_decoder_step(
            self,
            # views,
            feat,
            pos,
            ret_state=False):
        '''basically, the state_feat (global memory) stays the same, while the second CA sees all the previous hidden states.
        '''
        # state_feat, state_pos = feat[:, 0,].clone(), pos[:, 0,].clone() # remove intrinsics embedding here

        ress = []

        v = feat.shape[1]
        assert v >= 2  # this model works on at least 2 views

        # ! initialize the memory
        (m1,
         m2), f_state = self._rnn_decoder_initialize(feat[:, 0], None,
                                                     feat[:, 1], None)
        m2 = self._dropout_memory_token(m2)
        ress.append(m1)
        ress.append(m2)

        # st() # check whether inplace modified

        # record all feedback
        feedback_list_all = []

        if self.global_3d_feedback:
            feedback_m1 = self.global_3d_feedback_Inj(
                m1[-2])  # feature of layer L-1
            feedback_list_m1 = [
                feedback_m1 for _ in range(self.dec_depth - 1)
            ] + [0]
            feedback_m2 = self.global_3d_feedback_Inj(
                m2[-2])  # feature of layer L-1
            feedback_list_m2 = [
                feedback_m2 for _ in range(self.dec_depth - 1)
            ] + [0]
            memory = [
                torch.cat(
                    [m1[i] + feedback_list_m1[i], m2[i] + feedback_list_m2[i]],
                    dim=1) for i in range(self.dec_depth)
            ]  # L = 12. concat in sequence dimention.

            feedback_list_all += [feedback_m1, feedback_m2]

        else:
            memory = [
                torch.cat([m1[i], m2[i]], dim=1) for i in range(self.dec_depth)
            ]  # L = 12. concat in sequence dimention.

        # st()
        # for i in range(2, v):  # ! causal decoding
        for i in range(2, v, self.pack_size):  # ! causal decoding

            # dec, f_state = self._rnn_decoder_causal(
            decoupled_output, f_state = self._rnn_decoder_causal_pack(
                memory,  # ! avoid loading feat[:, 0] since already canonical frame
                feat[:, i:i + self.pack_size],
                ress,
                f_state,
            )

            # ! update the memory
            # ress.append(dec)
            ress.extend(decoupled_output)

            for dec in decoupled_output:

                if self.memory_dropout is not None:
                    dec_for_mem = self._dropout_memory_token(dec)
                else:
                    dec_for_mem = dec

                # ! global 3d feedback enhancement
                if self.global_3d_feedback:
                    feedback = self.global_3d_feedback_Inj(
                        dec[-2])  # feature of layer L-1
                    feedback_list = [
                        feedback for _ in range(self.dec_depth - 1)
                    ] + [0]
                    memory = [
                        torch.cat(
                            [memory[i], dec_for_mem[i] + feedback_list[i]],
                            dim=1) for i in range(self.dec_depth)
                    ]  # update the causal memory

                else:  # directly use dec[i] as the memory. like KV Cache.
                    memory = [
                        torch.cat([memory[i], dec_for_mem[i]], dim=1)
                        for i in range(self.dec_depth)
                    ]  # update the causal memory

        return ress, memory

    def _rnn_decoder_initialize(self, f1, pos_1, f2, pos_2, *args, **kwargs):
        # ! initializing causal inference from paired views

        final_output = [(f1 + self.learnable_frame0_token,
                         f2 + self.learnable_B_token)]
        b = f1.shape[0]
        final_state_output = [self.memory_token.expand(b, -1, -1)]
        # final_output = [(f1+0, f2 + self.learnable_B_token)]

        for idx, blk_siamese in enumerate(
                self.dec_blocks):  # ! 12 blocks of decoder.

            f1, _ = blk_siamese(*final_output[-1][::+1], pos_1, pos_2)
            f2, _ = blk_siamese(*final_output[-1][::-1], pos_2, pos_1)

            f_state, _ = blk_siamese(
                final_state_output[-1],
                torch.cat([
                    final_output[-1][0],
                    final_output[-1][1],
                ], dim=1),  # b l c
                None,
                None)

            # store the result
            final_output.append((f1, f2))
            final_state_output.append(f_state)

        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))

        return zip(*final_output), final_state_output[-1]

    def _rnn_decoder_causal(self, mem_feat, f_i, ress, f_state, *args,
                            **kwargs):
        # ! perform memory-based attention

        final_output = [f_i + self.learnable_B_token]
        final_mem_state = [f_state]

        for idx, blk_siamese in enumerate(
                self.dec_blocks):  # ! 12 blocks of decoder.

            # if (self.gradient_checkpointing and self.training
            #         and torch.is_grad_enabled()):
            #     f_i, _ = checkpoint(
            #         blk_siamese,
            #         final_output[-1],
            #         mem_feat[idx],
            #         None,
            #         None,
            #         use_reentrant=False
            #     )  # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint

            # else:

            # ! CA, memory read info from incoming frames
            f_state, _ = blk_siamese(
                final_mem_state[-1],  # b l c
                # rearrange(final_output[-1], 'b v l c -> b (v l) c'),
                final_output[-1],
                None,
                None)

            # ! new frames read information from memory (memory has info from both previous frames and incoming frames)
            # f_i, _ = blk_siamese(final_output[-1], mem_feat[idx], None,
            f_i, _ = blk_siamese(
                final_output[-1],
                torch.cat([final_mem_state[-1], mem_feat[idx]], dim=1), None,
                None)

            final_output.append(f_i)
            final_mem_state.append(f_state)

        final_output[-1] = self.dec_norm(final_output[-1])

        return final_output, final_mem_state[-1]  # no zip required
