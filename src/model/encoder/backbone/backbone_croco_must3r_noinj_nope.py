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
from .backbone_croco_must3r_noinj import Must3r_reimpl_noinj

class Must3r_reimpl_noinj_nope(
        Must3r_reimpl_noinj):

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

        # ! set the PE token to zero
        self.learnable_B_token.data.zero_()
        self.learnable_B_token.requires_grad_(False)

        self.learnable_frame0_token.data.zero_()
        self.learnable_frame0_token.requires_grad_(False)
