from copy import deepcopy
from functools import partial
import random
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


class FramePack3Rv3_Abla_Mem_with_KVCache_TwoDecoders_final(FramePack3Rv3):

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
        self.cutoff_window_k = 1
        self.pack_size = 1

        self.training_ratio = {
            'full_ctx': 0.3,
            'state_only': 0.40,
            'truncated_ctx_with_state': 0.3
        }

        self.random_training_choice_fn = partial(
            random.choices,
            list(self.training_ratio.keys()),
            weights=list(self.training_ratio.values()),
            k=1)

        # for fair ablation with cut3r_mast3rpt
        self.global_3d_feedback = False
        # for _, param in self.global_3d_feedback_Inj.named_parameters():
        #     param.requires_grad_(False)
        #     # param.requires_grad_(True)
        del self.global_3d_feedback_Inj

    def _rnn_decoder_causal_pack(self, mem_feat, f_nextpack, pos_f_nextpack, ress, f_state,
                                 pos_fstate, *args, **kwargs):
        # ! perform memory-based attention

        final_output = [f_nextpack + self.learnable_B_token.unsqueeze(1)]
        final_mem_state = [f_state]

        b, v, l, c = f_nextpack.shape
        # st()

        ress_v = len(ress)
        # if ress_v < self.pack_size * self.cutoff_window_k:
        prev_k_pack_indices_all = list(range(1, ress_v))  # will add [0] later
        # if True:
            # prev_k_pack_indices = prev_k_pack_indices_all 
        # else:

        prev_k_pack_indices = list(
            range(
                ress_v - self.pack_size * self.cutoff_window_k,
                ress_v,
            ))

        # st()

        current_training_mode = self.random_training_choice_fn()[0]
        memory_indices = [0] # ! always include the first frame as the memory token

        # for idx, blk_siamese in enumerate(
        #         self.dec_blocks):  # ! 12 blocks of decoder.

        for idx, (blk_siamese, blk_state) in enumerate(
                zip(self.dec_blocks,
                    self.dec_state_blocks)):  # ! 12 blocks of decoder.

            # Memory read from incoming frames
            # f_state, _ = blk_siamese(
            #     final_mem_state[-1],  # b l c
            #     rearrange(final_output[-1], 'b v l c -> b (v l) c'),
            #     None,
            #     None,
            # )

            # ! select the memmory based on the training mode
            # if current_training_mode == 'full_ctx':
            #     memory_indices += prev_k_pack_indices_all
            #     # Prepare memory to read from
            #     kvcache_mem = rearrange(mem_feat[idx],
            #                             'b (v l) c -> b v l c',
            #                             v=ress_v)[:, memory_indices]
            #     kvcache_mem = rearrange(
            #         kvcache_mem,
            #         'b v l c -> b (v l) c',
            #     )

            if current_training_mode == 'state_only':
                kvcache_mem = final_mem_state[-1]

            # elif current_training_mode == 'truncated_ctx_with_state':
            elif current_training_mode in ['full_ctx', 'truncated_ctx_with_state']: 
                if current_training_mode == 'full_ctx':
                    memory_indices += prev_k_pack_indices_all
                else:
                    memory_indices += prev_k_pack_indices
                kvcache_mem = rearrange(mem_feat[idx],
                                        'b (v l) c -> b v l c',
                                        v=ress_v)[:, memory_indices]

                # merge the memory tokens back
                kvcache_mem = rearrange(
                    kvcache_mem,
                    'b v l c -> b (v l) c',
                )
                kvcache_mem = torch.cat([kvcache_mem, final_mem_state[-1]], dim=1)# ! make sure the "state" always has gradients

            if self.gradient_checkpointing and self.training and torch.is_grad_enabled(
            ):

                def create_custom_forward(module_state, module_siamese):

                    def custom_forward(inputs, pos):
                        # f_state as the q
                        f_state, _ = module_state(
                            inputs[0],  # final_mem_state[-1]
                            inputs[1],  # rearranged final_output[-1]
                            pos[0], # pos of state
                            pos[1])

                        # incoming frames as the q
                        f_nextpack, _ = module_siamese( # ! ignore RoPE here for siamese decoder
                            inputs[2],  # rearranged final_output[-1]
                            inputs[3],  # concatenated memory
                            None,
                            None)
                        return f_state, f_nextpack

                    return custom_forward

                f_state, f_nextpack = checkpoint(
                    create_custom_forward(blk_state, blk_siamese),
                    (final_mem_state[-1],
                    rearrange(final_output[-1], 'b v l c -> b (v l) c'),
                    rearrange(final_output[-1], 'b v l c -> b (v l) c'),
                    kvcache_mem,
                    ),
                    (pos_fstate, rearrange(pos_f_nextpack, 'b v l c -> b (v l) c').contiguous()),
                    use_reentrant=False)
            else:
                # Memory read from incoming frames
                f_state, _ = blk_state(
                    final_mem_state[-1],  # b l c
                    rearrange(final_output[-1], 'b v l c -> b (v l) c'),
                    pos_fstate, 
                    rearrange(pos_f_nextpack, 'b v l c -> b (v l) c').contiguous())

                # New frames read information from memory
                f_nextpack, _ = blk_siamese(
                    rearrange(final_output[-1], 'b v l c -> b (v l) c'),
                    kvcache_mem, 
                    None,
                    None)

            final_output.append(
                rearrange(f_nextpack, 'b (v l) c -> b v l c', b=b, v=v))
            final_mem_state.append(f_state)

        final_output[-1] = self.dec_norm(final_output[-1])
        final_mem_state[-1] = self.dec_norm_for_state(final_mem_state[-1])

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

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads,
                     dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):

        super()._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads,
                             dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec)

        # transformer for the decoder, state
        # init from self.dec_blocks
        self.dec_norm_for_state = norm_layer(dec_embed_dim)

        self.state_pe = "2d"
        pos_embed = 'RoPE100'
        freq = float(pos_embed[len('RoPE'):])
        self.state_decoder_rope = RoPE2D(freq=freq)

        self.dec_state_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim,
                         dec_num_heads,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=True,
                         norm_layer=norm_layer,
                         norm_mem=norm_im2_in_dec,
                         rope=self.state_decoder_rope)
            for i in range(dec_depth)
        ])

        pass

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
        final_state_output[-1] = self.dec_norm_for_state(
            final_state_output[-1])  # ! add norm for state here

        return zip(*final_output), final_state_output[-1]

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

        b, v = feat.shape[0], feat.shape[1]
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

        if self.state_pe == "2d":  # ! default setting here.
            width = int(self.memory_token.shape[1]**0.5)
            width = width + 1 if width % 2 == 1 else width
            pos_fstate = (torch.tensor(
                [[i // width, i % width]
                 for i in range(self.memory_token.shape[1])],
                dtype=pos.dtype,
                device=pos.device,
            )[None].expand(b, -1, -1).contiguous())
        else:
            raise NotImplementedError(
                f"state_pe={self.state_pe} not implemented")

        # st()
        # for i in range(2, v):  # ! causal decoding
        for i in range(2, v, self.pack_size):  # ! causal decoding

            # dec, f_state = self._rnn_decoder_causal(
            decoupled_output, f_state = self._rnn_decoder_causal_pack(
                memory,  # ! avoid loading feat[:, 0] since already canonical frame
                feat[:, i:i + self.pack_size],
                pos[:, i:i + self.pack_size],
                ress,
                f_state,
                pos_fstate=pos_fstate,
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
