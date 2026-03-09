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

class Must3r_reimpl_noinj(
        Must3r_reimpl):

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
        # del self.dec_norm_state

        # # [B] to denote canonical coordinate
        # self.learnable_B_token = nn.Parameter(
        #     torch.empty(1, 1, self.dec_embed_dim))  # to add to [B, L C]
        # torch.nn.init.normal_(self.learnable_B_token, mean=0.0,
        #                       std=0.02)  # init as normal distribution
        # self.learnable_B_token.requires_grad_(True)  # trainable

        # # ! for frame0
        # self.learnable_frame0_token = nn.Parameter(
        #     torch.empty(1, 1, self.dec_embed_dim))  # to add to [B, L C]
        # torch.nn.init.normal_(self.learnable_frame0_token, mean=0.0,
        #                       std=0.02)  # init as normal distribution
        # self.learnable_frame0_token.requires_grad_(True)  # trainable

        self.global_3d_feedback = False  # ! disable in the paired view pre-training stage
        # st()
        # self.global_3d_feedback = True
        del self.global_3d_feedback_Inj
        # if self.global_3d_feedback:
        #     self.global_3d_feedback_Inj = nn.Sequential(
        #         nn.LayerNorm(self.dec_embed_dim),
        #         Mlp(self.dec_embed_dim,
        #             hidden_features=self.dec_embed_dim * 4,
        #             act_layer=nn.GELU,
        #             drop=0.))
        #     pass

        # # self.memory_dropout = None
        # # self.memory_dropout = 0.05  # randomly remove memory tokens
        # self.memory_dropout = [
        #     0.05, 0.10
        #     # 0.1, 0.2
        # ]  # following must3r, add higher dropout rate for high-res img
        # # self.token_length_range = [(256 / 16)**2 + 1, (512 * 384 / 16 / 16 + 1)
        # self.token_length_range = [(224 / 16)**2 + 1, (512 * 384 / 16 / 16 + 1)
        #                            ]  # range of tokens

        # # self.gradient_checkpointing = False
        # self.gradient_checkpointing = True  # ! debug and support this
