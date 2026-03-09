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


class FramePack3Rv3_Abla_FA_no_state(FramePack3Rv3):

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
        self.pack_size = 3

        # # ! be compatible with the previous model
        for _, param in self.global_3d_feedback_Inj.named_parameters():
            param.requires_grad_(False) # unused

        del self.memory_token

    def _rnn_decoder_initialize(self, f1, pos_1, f2, pos_2, *args, **kwargs):
        # ! initializing causal inference from paired views

        # st()
        final_output = [
            torch.cat(
                [
                    (f1 + self.learnable_frame0_token).unsqueeze(1),  # b 1 l c
                    f2 + self.learnable_B_token.unsqueeze(1)  # b v l c
                ],
                dim=1) # b v l c
        ]

        b, v_f2, l, c = f2.shape

        for idx, blk_siamese in enumerate(
                self.dec_blocks):  # ! 12 blocks of decoder.

            f1, _ = blk_siamese(
                final_output[-1][:, 1],
                rearrange(final_output[-1][:, 1:], 'b v l c -> b (v l) c'),
                pos_1, pos_2)
            f2, _ = blk_siamese(
                rearrange(final_output[-1][:, 1:], 'b v l c -> b (v l) c'),
                final_output[-1][:, 1], pos_2, pos_1)

            f2 = rearrange(f2, "b (v l) c -> b v l c", b=b, v=v_f2)

            final_output.append(torch.cat(
                [f1.unsqueeze(1), f2], dim=1))  # 1 1 L C + 1 V-1 L C -> 1 V L C

            # final_state_output.append(f_state)

        final_output[-1] = self.dec_norm(final_output[-1])
        # st()

        # reshape final_output list into a list of tuples
        # final_output_compat = []
        # for v in range(final_output[0].shape[1]):
        #     final_output_compat.append(tuple(item[:, v, :, :] for item in final_output))

        # return zip(*final_output), final_state_output[-1]
        # return zip(*final_output_compat)
        return final_output

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
        ress = self._rnn_decoder_initialize(feat[:, 0], None,
                                                     feat[:, 1:], None)
        # m2 = self._dropout_memory_token(m2)
        # ress.append(m1)
        # ress.append(m2)
        # ress = list(ress)

        # st() # check whether inplace modified

        # record all feedback
        # feedback_list_all = []

        # if self.global_3d_feedback:
        #     feedback_m1 = self.global_3d_feedback_Inj(
        #         m1[-2])  # feature of layer L-1
        #     feedback_list_m1 = [
        #         feedback_m1 for _ in range(self.dec_depth - 1)
        #     ] + [0]
        #     feedback_m2 = self.global_3d_feedback_Inj(
        #         m2[-2])  # feature of layer L-1
        #     feedback_list_m2 = [
        #         feedback_m2 for _ in range(self.dec_depth - 1)
        #     ] + [0]
        #     memory = [
        #         torch.cat(
        #             [m1[i] + feedback_list_m1[i], m2[i] + feedback_list_m2[i]],
        #             dim=1) for i in range(self.dec_depth)
        #     ]  # L = 12. concat in sequence dimention.

        #     feedback_list_all += [feedback_m1, feedback_m2]

        # else:
        #     memory = [
        #         torch.cat([m1[i], m2[i]], dim=1) for i in range(self.dec_depth)
        #     ]  # L = 12. concat in sequence dimention.

        # # st()
        # # for i in range(2, v):  # ! causal decoding
        # for i in range(2, v, self.pack_size):  # ! causal decoding

        #     # dec, f_state = self._rnn_decoder_causal(
        #     decoupled_output, f_state = self._rnn_decoder_causal_pack(
        #         memory,  # ! avoid loading feat[:, 0] since already canonical frame
        #         feat[:, i:i + self.pack_size],
        #         ress,
        #         f_state,
        #     )

        #     # ! update the memory
        #     # ress.append(dec)
        #     ress.extend(decoupled_output)

        #     for dec in decoupled_output:

        #         if self.memory_dropout is not None:
        #             dec_for_mem = self._dropout_memory_token(dec)
        #         else:
        #             dec_for_mem = dec

        #         # ! global 3d feedback enhancement
        #         if self.global_3d_feedback:
        #             feedback = self.global_3d_feedback_Inj(
        #                 dec[-2])  # feature of layer L-1
        #             feedback_list = [
        #                 feedback for _ in range(self.dec_depth - 1)
        #             ] + [0]
        #             memory = [
        #                 torch.cat(
        #                     [memory[i], dec_for_mem[i] + feedback_list[i]],
        #                     dim=1) for i in range(self.dec_depth)
        #             ]  # update the causal memory

        #         else:  # directly use dec[i] as the memory. like KV Cache.
        #             memory = [
        #                 torch.cat([memory[i], dec_for_mem[i]], dim=1)
        #                 for i in range(self.dec_depth)
        #             ]  # update the causal memory

        return ress, None

    def _decoder(self, feat, pose, extra_embed=None):
        b, v, l, c = feat.shape
        # final_output = [feat]  # before projection
        if extra_embed is not None:
            feat = torch.cat((feat, extra_embed), dim=-1)

        # project to decoder dim
        f = rearrange(feat, "b v l c -> (b v) l c")
        f = self.decoder_embed(f)
        f = rearrange(f, "(b v) l c -> b v l c", b=b, v=v).contiguous()

        # ! RNN rollout
        # ! len(rnn_rollout_dec): v
        # ! len(rnn_rollout_dec[0]): 13 (length of decoder + 1)
        # st()

        final_output, memory = self._forward_decoder_step(
            f, pose, ret_state=False
        )  #  len(rnn_rollout_dec)==V, len(rnn_rollout_dec[0])==13
        # st()
        final_output = [feat] + final_output

        # final_output = [
        #     torch.stack([dec_tuple[i] for dec_tuple in final_output], dim=1)
        #     for i in range(len(final_output))
        # ]  # stack tuple of per-layer output to B V L C
        # # final_output += rnn_rollout_dec
        # st()

        del final_output[
            1]  # duplicate with final_output[0]. just linear projection.

        return final_output # the output is a list of B V L C tensors (length==12)

