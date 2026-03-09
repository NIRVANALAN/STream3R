from copy import deepcopy
from dataclasses import dataclass
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


class FramePack3Rv2(FramePack3R):

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

        self.cutoff_window_k = 1  # only attend to & update frames within recent k pack
        self.pack_size = 3

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
        self.mem_read_blk = Read_Block(z_dim=dec_embed_dim,
                                       x_dim=dec_embed_dim,
                                       num_heads=dec_num_heads,
                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=True,
                                       norm_layer=norm_layer)
        # self.mem_write_blk = Write_Block(z_dim=dec_embed_dim,
        #                                  x_dim=dec_embed_dim,
        #                                  num_heads=dec_num_heads,
        #                                  mlp_ratio=mlp_ratio,
        #                                  qkv_bias=True,
        #                                  norm_layer=norm_layer)

    def _rnn_decoder_causal(self,
                            f_state,
                            ress,
                            f_nextpack,
                            return_ress=False,
                            *args,
                            **kwargs):
        # ! perform memory-based attention
        '''return_ress: directly return updated ress
        '''

        final_output = [f_nextpack + self.learnable_B_token.unsqueeze(1)
                        ]  # read a pack sequentially
        final_mem_state = [f_state]

        # mem_length = len(ress)
        b, ress_v = ress[0].shape[:2]
        use_memory_token_for_retrieval = False

        prev_k_pack_indices = list(
            range(
                ress_v - self.pack_size * self.cutoff_window_k,
                ress_v,
            ))

        # if ress_v > 1 + self.pack_size * self.cutoff_window_k:
        if True: # force using memory features
            # involves frame_0 as the memory
            # memory_indices = [0] + list(
            #     range(ress_v - self.pack_size * self.cutoff_window_k, ress_v,
            #           1))
            memory_indices = [0] + prev_k_pack_indices
            use_memory_token_for_retrieval = True
            # st()

        else:
            # directly use kv cache for the memory
            memory_indices = list(range(ress_v))

        # mem_v = len(memory_indices)

        # other_indices = [  # get indices for all other tokens
        #     i for i in list(range(ress_v)) if i not in memory_indices
        # ]

        # ! if < self.cuttoff_window, use all tokens available. Otherwise, use memory instead.
        mem_feat = [
            rearrange(ress[i][:, memory_indices], 'b v l c -> b (v l) c')
            for i in range(self.dec_depth)
        ]  # list of B, mem_v, L, C

        # updated_mem = []
        updated_prev_packs = []

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
            updated_fstate, _ = blk_siamese(
                final_mem_state[-1],  # b l c
                rearrange(final_output[-1], 'b v l c -> b (v l) c'),
                None,
                None)
            final_mem_state.append(updated_fstate)

            # ! CA, new frames read information from memory
            mem_to_read_from = mem_feat[idx]
            if use_memory_token_for_retrieval:
                mem_to_read_from = torch.cat(
                    [final_mem_state[-1], mem_to_read_from], dim=1)

            f_nextpack_i, _ = blk_siamese(
                rearrange(final_output[-1], 'b v l c -> b (v l) c'),
                mem_to_read_from, None, None)  # b l c

            final_output.append(
                rearrange(f_nextpack_i,
                          'b (v l) c -> b v l c',
                          b=b,
                          v=final_output[-1].shape[1]))

            # ! CA, update prev k packs
            # st()
            # ! TODO
            # f_pack_i, _ = blk_siamese(
            #     rearrange(ress[idx][:, prev_k_pack_indices],
            #               'b v l c -> b (v l) c',
            #               v=self.pack_size * self.cutoff_window_k),
            #     final_mem_state[-1],
            #     # rearrange(mem_feat[idx][:, 1:], 'b n l c -> (b n) l c'),
            #     None,  # (BxN) L C, where N is the number of packs
            #     None)
            # updated_prev_packs.append(
            #     rearrange(f_pack_i,
            #               'b (v l) c -> b v l c',
            #               v=self.pack_size * self.cutoff_window_k))
            # ! TODO, remove memory update for now
            updated_prev_packs.append(ress[idx][:, prev_k_pack_indices])

        # TODO, merge all updated tokens into the ress?

        final_output[-1] = self.dec_norm(final_output[-1])

        # if return_ress:  # ! concat final_output, append to ress, and return the new_ress

        # final_output, updated_mem, updated_within_pack
        new_ress = [ress[0]] + [
            torch.empty_like(layer_ress) for layer_ress in ress[1:]
        ]  # ress[0] is just raw encoder output tokens

        # merge mem,other pack frames; and cat incoming frame packs
        for idx in range(0, len(ress) - 1):  # 12 layers
            # new_ress[idx + 1][:, memory_indices] = updated_mem[idx]
            new_ress[idx + 1][:, prev_k_pack_indices] = updated_prev_packs[idx]

        for idx in range(0, len(ress)):  # final_output has 13 layers
            new_ress[idx] = torch.cat([new_ress[idx], final_output[idx]],
                                      dim=1)  # merge incoming packs

        return new_ress, final_mem_state[-1]

        # else:  # return updated stuffs
        #     return final_output, final_mem_state, updated_prev_packs  # no zip required

    def _forward_decoder_step(
            self,
            # views,
            feat,
            pos,
            ret_state=False):
        '''basically, the state_feat (global memory) stays the same, while the second CA sees all the previous hidden states.
        '''

        ress = []
        # memory = []

        # v = feat.shape[1]
        b, v, l, c = feat.shape  # already projected
        assert v >= 2  # this model works on at least 2 views

        # ! first, do CA between f_0 and f_[1, window_size]
        ress = self._rnn_decoder_initialize(feat[:, 0], None,
                                            feat[:,
                                                 1:self.pack_size + 1], None)

        # st()

        # memory = [
        #     torch.cat([ret[i] for ret in ress], dim=1)
        #     for i in range(self.dec_depth)
        # ]

        if v > 1 + self.pack_size:
            #     return ress

            # else:
            # raise NotImplementedError()

            # ! init f_state with a read block
            # st()
            f_state = self.mem_read_blk(
                repeat(self.memory_token, '1 l c -> b l c', b=b),
                rearrange(ress[-1], 'b v l c -> b (v l) c'), None, None)

            for i in range(1 + self.pack_size, v,
                           self.pack_size):  # ! causal decoding

                ress, f_state = self._rnn_decoder_causal(
                    # memory,  # ! avoid loading feat[:, 0] since already canonical frame
                    f_state,
                    ress,
                    feat[:, i:i + self.pack_size],
                    return_ress=True)

                # st()

                # ! update the memory
                # ress.append(dec)

                # if self.memory_dropout is not None:
                #     dec_for_mem = self._dropout_memory_token(dec)
                # else:
                # dec_for_mem = dec

                # # ! global 3d feedback enhancement
                # if self.global_3d_feedback:
                #     feedback = self.global_3d_feedback_Inj(
                #         dec[-2])  # feature of layer L-1
                #     feedback_list = [feedback
                #                     for _ in range(self.dec_depth - 1)] + [0]
                #     memory = [
                #         torch.cat([memory[i], dec_for_mem[i] + feedback_list[i]],
                #                 dim=1) for i in range(self.dec_depth)
                #     ]  # update the causal memory

                # else:  # directly use dec[i] as the memory. like KV Cache.
                # memory = [
                #     torch.cat([memory[i], dec_for_mem[i]], dim=1)
                #     for i in range(self.dec_depth)
                # ]  # update the causal memory

        return ress

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

        rnn_rollout_dec = self._forward_decoder_step(
            f, pose, ret_state=False
        )  #  len(rnn_rollout_dec)==V, len(rnn_rollout_dec[0])==13

        # rnn_rollout_dec = [
        #     torch.stack([dec_tuple[i] for dec_tuple in rnn_rollout_dec], dim=1)
        #     for i in range(len(rnn_rollout_dec[0]))
        # ]  # stack tuple of last layer output to B V L C
        final_output += rnn_rollout_dec

        del final_output[
            1]  # duplicate with final_output[0]. just linear projection.

        return final_output

    # def on_after_backward(self):
    #     for name, param in self.named_parameters():
    #         # if param.grad is None:
    #         if param.grad is None and param.requires_grad:
    #             print(name)
    #     st()
    #     pass

