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

from .backbone_croco_must3r import Must3r_reimpl

from src.model.encoder.backbone.croco.blocks import Write_Block


class HybridCausalFA(Must3r_reimpl):

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        super().__init__(cfg, d_in)
        """changelog:
        1. add a function to build memory from ress online
        2. add new cross attention for (write back)
        3. fix m2 memory update position
        """

        # set the

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads,
                     dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        super()._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads,
                             dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec)

        # ! add self.write_blocks
        self.write_blocks = nn.ModuleList([
            Write_Block(z_dim=dec_embed_dim,
                        x_dim=dec_embed_dim,
                        num_heads=dec_num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer) for _ in range(dec_depth)
        ])

    def _build_mem_online(self, ress, feedback_list_all):
        # convert ress to memory by concatenating all tokens on L dim; 
        # add random dropout and inject feedback 

        memory = []

        for j in range(
                self.dec_depth):  # add per layer output to the memory list

            memory.append(
                torch.cat([
                    ress[i][j] +
                    (feedback_list_all[i] if j != self.dec_depth - 1 else 0)
                    for i in range(len(ress))
                ],
                          dim=1))  # L = 12. concat in sequence dimention.

        # st()
        memory = self._dropout_memory_token(memory, sample_length=len(ress))

        return memory

    def _build_layerwise_ress(self, ress):
        # concat ress on L dim

        layerwise_ress = []

        for j in range(len(ress[0])):  # add per layer output to the memory list

            layerwise_ress.append(
                torch.cat([ress[i][j] for i in range(len(ress))],
                          dim=1))  # L = 12. concat in sequence dimention.

        return layerwise_ress
    
    def _layerwise_ress_to_ress(self, layerwise_ress, v):
        """
        Args:
            layerwise_ress: list of tensors of shape [B, total_T, D], len = L (layers)
            v: how many views here
        Returns:
            ress: list of N elements, each is a list of L tensors of shape [B, T_i, D]
        """
        ress = [[] for _ in range(v)]

        for j in range(len(layerwise_ress)):  # for each layer
            split_layer = torch.chunk(layerwise_ress[j], chunks=v, dim=1)  # split along sequence dim
            for i in range(v):
                ress[i].append(split_layer[i])

        return ress

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
        ress.append(m1)
        ress.append(m2)

        # m2 = self._dropout_memory_token(m2)

        # st()  # check whether inplace modified

        # record all feedback
        feedback_list_all = []

        if self.global_3d_feedback:
            feedback_m1 = self.global_3d_feedback_Inj(
                m1[-2])  # feature of layer L-1
            feedback_m2 = self.global_3d_feedback_Inj(
                m2[-2])  # feature of layer L-1
            # feedback_list_m1 = [
            #     feedback_m1 for _ in range(self.dec_depth - 1)
            # ] + [0]
            # feedback_list_m2 = [
            #     feedback_m2 for _ in range(self.dec_depth - 1)
            # ] + [0]
            # memory = [
            #     torch.cat(
            #         [m1[i] + feedback_list_m1[i], m2[i] + feedback_list_m2[i]],
            #         dim=1) for i in range(self.dec_depth)
            # ]  # L = 12. concat in sequence dimention.

            feedback_list_all += [feedback_m1, feedback_m2]
            memory = self._build_mem_online(ress, feedback_list_all)
            layerwise_ress = self._build_layerwise_ress(ress) # list of B, v*L, C
            # ress_2 = self._layerwise_ress_to_ress(layerwise_ress, v=2)
            # inverted right

        else:
            raise NotImplementedError()

        # else:
        #     memory = [
        #         torch.cat([m1[i], m2[i]], dim=1) for i in range(self.dec_depth)
        #     ]  # L = 12. concat in sequence dimention.

        for i in range(2, v):  # ! causal decoding

            dec, update_past_f = self._rnn_decoder_hybrid(
                memory,  # ! avoid loading feat[:, 0] since already canonical frame
                feat[:, i],
                layerwise_ress
            )

            # ! update the memory
            # ress.append(dec)

            # ! use the updated past latents instead
            # update_past_f.append(dec)
            ress = self._layerwise_ress_to_ress(update_past_f, v=i) + [dec]

            # if self.memory_dropout is not None:
            #     dec_for_mem = self._dropout_memory_token(dec)
            # else:
            #     dec_for_mem = dec

            # ! global 3d feedback enhancement
            if self.global_3d_feedback:
                feedback = self.global_3d_feedback_Inj(
                    dec[-2])  # feature of layer L-1

                feedback_list_all.append(feedback) # TODO: reuse existing feedback, is it ok?
                memory = self._build_mem_online(ress, feedback_list_all)

            else:  # directly use dec[i] as the memory. like KV Cache.
                raise NotImplementedError()

            layerwise_ress = self._build_layerwise_ress(ress)

        return ress, memory

    def _rnn_decoder_hybrid(self, mem_feat, f_i, past_f, *args, **kwargs):
        # ! perform memory-based attention

        final_output = [f_i + self.learnable_B_token]
        update_past_f = [past_f[0]] # the first position will not be used.

        for idx, blk_siamese in enumerate(
                self.dec_blocks):  # ! 12 blocks of decoder.

            if (self.gradient_checkpointing and self.training
                    and torch.is_grad_enabled()):
                f_i, _ = checkpoint(
                    blk_siamese, # a SA + CA, layerwise
                    final_output[-1],
                    mem_feat[idx],
                    None,
                    None,
                    use_reentrant=False
                )  # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint

                # update history latents
                past_f_i = checkpoint(
                    self.write_blocks[idx], # a CA
                    final_output[-1], # z
                    past_f[idx+1], # x
                    None, 
                    None,
                    use_reentrant=False
                )  # https://pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint

            else:

                f_i, _ = blk_siamese(final_output[-1], mem_feat[idx], None,
                                     None)
                past_f_i = self.write_blocks[idx](final_output[-1], past_f[idx+1])

            final_output.append(f_i)
            update_past_f.append(past_f_i)

        final_output[-1] = self.dec_norm(final_output[-1])

        # ! decouple update_past_f into chunks

        return final_output, update_past_f  # no zip required
