from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from torch import nn
from torch.utils.checkpoint import checkpoint

from .croco.blocks import DecoderBlock, Mlp
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


class Must3r_reimpl(
        AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn_frame0_as_memory):

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        super().__init__(cfg, d_in)
        """changelog:
        # DONE:
        1. siamese decoder
        2. causal memory (same with current implementation), as KVCache, different Mi for different layers. 
        3. learnable [B] token to denote canonical space
        4. Online memory update
        5. no RoPE
        # TODO:
        7. INJ3D global feedback (after current version stable training and data verified), later.
        6. pre-train with pair data (log-depth) supervision first.
        8. predict both local GS and global GS. (after debugging dataset)
        """

        # siamese decoder
        del self.dec_norm_state

        # [B] to denote canonical coordinate
        self.learnable_B_token = nn.Parameter(
            torch.empty(1, 1, self.dec_embed_dim))  # to add to [B, L C]
        torch.nn.init.normal_(self.learnable_B_token, mean=0.0,
                              std=0.02)  # init as normal distribution
        self.learnable_B_token.requires_grad_(True)  # trainable

        # ! for frame0
        self.learnable_frame0_token = nn.Parameter(
            torch.empty(1, 1, self.dec_embed_dim))  # to add to [B, L C]
        torch.nn.init.normal_(self.learnable_frame0_token, mean=0.0,
                              std=0.02)  # init as normal distribution
        self.learnable_frame0_token.requires_grad_(True)  # trainable

        # self.global_3d_feedback = False  # ! disable in the paired view pre-training stage
        self.global_3d_feedback = True
        if self.global_3d_feedback:
            self.global_3d_feedback_Inj = nn.Sequential(
                nn.LayerNorm(self.dec_embed_dim),
                Mlp(self.dec_embed_dim,
                    hidden_features=self.dec_embed_dim * 4,
                    act_layer=nn.GELU,
                    drop=0.))
            pass

        # self.memory_dropout = None
        # self.memory_dropout = 0.05  # randomly remove memory tokens
        self.memory_dropout = [
            0.05, 0.10
            # 0.1, 0.2
        ]  # following must3r, add higher dropout rate for high-res img
        # self.token_length_range = [(256 / 16)**2 + 1, (512 * 384 / 16 / 16 + 1)
        self.token_length_range = [(224 / 16)**2 + 1, (512 * 384 / 16 / 16 + 1)
                                   ]  # range of tokens

        # self.gradient_checkpointing = False
        self.gradient_checkpointing = True  # ! debug and support this

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

        assert drop_rate >= self.memory_dropout[0] and drop_rate <= self.memory_dropout[1]

        # self.token_length_range

        valid_mask = torch.empty_like(memory_feat[0]).bernoulli(1 - drop_rate)
        # st()
        memory_feat = [valid_mask * mem
                       for mem in memory_feat]  # same mask in each layer
        return memory_feat

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
        m1, m2 = self._rnn_decoder_initialize(feat[:, 0], None, feat[:, 1],
                                              None)
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
        for i in range(2, v):  # ! causal decoding

            dec = self._rnn_decoder_causal(
                memory,  # ! avoid loading feat[:, 0] since already canonical frame
                feat[:, i],
            )

            # ! update the memory
            ress.append(dec)

            if self.memory_dropout is not None:
                dec_for_mem = self._dropout_memory_token(dec)
            else:
                dec_for_mem = dec

            # ! global 3d feedback enhancement
            if self.global_3d_feedback:
                feedback = self.global_3d_feedback_Inj(
                    dec[-2])  # feature of layer L-1
                feedback_list = [feedback
                                 for _ in range(self.dec_depth - 1)] + [0]
                memory = [
                    torch.cat([memory[i], dec_for_mem[i] + feedback_list[i]],
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
        # final_output = [(f1+0, f2 + self.learnable_B_token)]

        for idx, blk_siamese in enumerate(
                self.dec_blocks):  # ! 12 blocks of decoder.

            f1, _ = blk_siamese(*final_output[-1][::+1], pos_1, pos_2)
            f2, _ = blk_siamese(*final_output[-1][::-1], pos_2, pos_1)

            # store the result
            final_output.append((f1, f2))

        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))

        return zip(*final_output)

    def _rnn_decoder_causal(self, mem_feat, f_i, *args, **kwargs):
        # ! perform memory-based attention

        final_output = [f_i + self.learnable_B_token]

        for idx, blk_siamese in enumerate(
                self.dec_blocks):  # ! 12 blocks of decoder.

            if (self.gradient_checkpointing and self.training
                    and torch.is_grad_enabled()):
                f_i, _ = checkpoint(
                    blk_siamese,
                    final_output[-1],
                    mem_feat[idx],
                    None,
                    None,
                    use_reentrant=False
                )  # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint

            else:

                f_i, _ = blk_siamese(final_output[-1], mem_feat[idx], None,
                                     None)

            final_output.append(f_i)

        final_output[-1] = self.dec_norm(final_output[-1])

        return final_output  # no zip required

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads,
                     dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec, causal=False):

        # ! disable RoPE in decoder
        self.decoder_rope = None

        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        enc_embed_dim = enc_embed_dim + self.intrinsics_embed_decoder_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim,
                         dec_num_heads,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=True,
                         norm_layer=norm_layer,
                         norm_mem=norm_im2_in_dec,
                         rope=self.decoder_rope,
                         causal=causal) for i in range(dec_depth) # ! not enabling qk_norm here...
        ])
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)
