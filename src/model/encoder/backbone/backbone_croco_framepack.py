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


class FramePack3R(Must3r_reimpl):

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        super().__init__(cfg, d_in)
        """changelog:
        TODO:
        1. merge from AsymmetricCroCoMulti_RW, using the [B] and single CA version
        2. same behaviour with v=2
        3. supports AR decoding
        4. parallel operation (window=3, e.g.)
        5. trained with random "write back"?
        6. supports global_3d_feedback?
        DONE:
        """

        self.pack_size = 3
        # del self.global_3d_feedback_Inj # not used
        # ! disable it for now.
        for _, param in self.global_3d_feedback_Inj.named_parameters():
            param.requires_grad_(False)

    def _rnn_decoder_initialize(self, f1, pos_1, f2, pos_2, *args, **kwargs):
        # ! initializing causal inference from paired views

        final_output = [(f1 + self.learnable_frame0_token,
                         f2 + self.learnable_B_token.unsqueeze(1))]
        b, ctx_v = f2.shape[:2]

        for idx, blk_siamese in enumerate(
                self.dec_blocks):  # ! 12 blocks of decoder.

            feat_current = final_output[-1]

            # frame_0 attend to incoming frames
            f1, _ = blk_siamese(
                feat_current[0],
                rearrange(feat_current[1], 'b v l c -> b (v l) c'), pos_1,
                pos_2)

            # all incoming frames CA to frame_0
            # f2, _ = blk_siamese(
            #     rearrange(feat_current[1], 'b v l c -> (b v) l c'),
            #     repeat(feat_current[0], 'b l c -> (b v) l c', v=ctx_v), pos_2,
            #     pos_1)
            f2, _ = blk_siamese(
                rearrange(feat_current[1], 'b v l c -> b (v l) c'),
                feat_current[0],  # b l c
                pos_2,
                # repeat(feat_current[0], 'b l c -> (b v) l c', v=ctx_v), pos_2,
                pos_1)

            # store the result
            final_output.append(
                (f1, rearrange(f2, 'b (v l) c -> b v l c', b=b, v=ctx_v)))
            # (f1, rearrange(f2, '(b v) l c -> b v l c', b=b, v=ctx_v)))

        # st() # check what format to return? shall be b v l c
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))

        # final_output = [torch.cat([output[0].unsqueeze(1), rearrange(output[1], '(b v) l c -> b v l c', v=ctx_v)], dim=1) for output in final_output]
        final_output = [
            torch.cat([output[0].unsqueeze(1), output[1]], dim=1)
            for output in final_output
        ]

        return final_output

    def _rnn_decoder_causal(self,
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

        # st()

        # mem_length = len(ress)
        b, ress_v = ress[0].shape[:2]

        # ! currently select 0 + middle token for each pack
        memory_indices = [0] + list(
            range(1 + self.pack_size // 2, ress_v, self.pack_size))
        mem_v = len(memory_indices)

        other_indices = [  # get indices for all other tokens
            i for i in list(range(ress_v)) if i not in memory_indices
        ]

        mem_feat = [ress[i][:, memory_indices]
                    for i in range(self.dec_depth)]  # list of B, mem_v, L, C

        updated_mem = []
        updated_within_pack = []

        # st()
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

            # ! CA, memory read info from new frames
            updated_mem_i, _ = blk_siamese(
                rearrange(mem_feat[idx], 'b v l c -> b (v l) c'),
                rearrange(final_output[-1], 'b v l c -> b (v l) c'), None,
                None)
            updated_mem.append(  # ! layerwise updates
                rearrange(updated_mem_i, 'b (v l) c -> b v l c', b=b, v=mem_v))

            # ! CA, write frame_0 memory into new frames
            f_nextpack_i, _ = blk_siamese(
                rearrange(final_output[-1], 'b v l c -> b (v l) c'),
                mem_feat[idx][:, 0], None, None)  # b l c

            final_output.append(
                rearrange(f_nextpack_i,
                          'b (v l) c -> b v l c',
                          b=b,
                          v=final_output[-1].shape[1]))

            # ! CA, write middle frame memory into its own packs
            # N = B * num_+pack
            # v = self.pack_size - 1
            f_pack_i, _ = blk_siamese(
                rearrange(ress[idx][:, other_indices],
                          'b (n v) l c -> (b n) (v l) c',
                          v=self.pack_size - 1),  # (BxN) (V L) C
                rearrange(mem_feat[idx][:, 1:], 'b n l c -> (b n) l c'),
                None,  # (BxN) L C, where N is the number of packs
                None)
            updated_within_pack.append(
                rearrange(f_pack_i,
                          '(b n) (v l) c -> b (n v) l c',
                          b=b,
                          v=self.pack_size - 1))

        # TODO, merge all updated tokens into the ress?

        final_output[-1] = self.dec_norm(final_output[-1])

        if return_ress:  # ! concat final_output, append to ress, and return the new_ress

            # final_output, updated_mem, updated_within_pack
            new_ress = [ress[0]] + [
                torch.empty_like(layer_ress) for layer_ress in ress[1:]
            ]  # ress[0] is just raw encoder output tokens

            # merge mem,other pack frames; and cat incoming frame packs
            for idx in range(0, len(ress) - 1):  # 12 layers
                new_ress[idx + 1][:, memory_indices] = updated_mem[idx]
                new_ress[idx + 1][:, other_indices] = updated_within_pack[idx]

            for idx in range(0, len(ress)):  # final_output has 13 layers
                new_ress[idx] = torch.cat([new_ress[idx], final_output[idx]],
                                          dim=1)  # merge incoming packs

            return new_ress

        else:  # return updated stuffs
            return final_output, updated_mem, updated_within_pack  # no zip required

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

            for i in range(1 + self.pack_size, v,
                           self.pack_size):  # ! causal decoding

                ress = self._rnn_decoder_causal(
                    # memory,  # ! avoid loading feat[:, 0] since already canonical frame
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
