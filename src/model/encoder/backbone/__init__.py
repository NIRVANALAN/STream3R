from typing import Any
import torch.nn as nn

from .backbone import Backbone
from .backbone_croco_multiview import AsymmetricCroCoMulti, AsymmetricCroCoMulti_RW
# from .backbone_croco_multiview_cut3r import AsymmetricCroCoMulti_RW_CUT3R, AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn
from .backbone_croco_multiview_cut3r import *
from .backbone_dino import BackboneDino, BackboneDinoCfg
from .backbone_resnet import BackboneResnet, BackboneResnetCfg
from .backbone_croco import AsymmetricCroCo, BackboneCrocoCfg
from .backbone_croco_must3r import Must3r_reimpl
from .backbone_croco_hybridattn import HybridCausalFA
from .backbone_croco_framepack import FramePack3R
from .backbone_croco_framepack2 import FramePack3Rv2
from .backbone_croco_framepack3 import FramePack3Rv3

# ablation
from .framepack3_abla_mem_with_kvcache import FramePack3Rv3_Abla_Mem_with_KVCache
from .framepack3_abla_mem_with_kvcache0 import FramePack3Rv3_Abla_Mem_with_KVCache0
from .framepack3_abla_mem_with_kvcache3_stride1 import FramePack3Rv3_Abla_Mem_with_KVCache3_stride1
from .framepack3_abla_fa_nostate import FramePack3Rv3_Abla_FA_no_state
from .framepack3_abla_fa_wstate import FramePack3Rv3_Abla_FA_with_state
from .framepack3_abla_mem_with_kvcache3_stride3 import FramePack3Rv3_Abla_Mem_with_KVCache3_stride3
from .framepack3_abla_mem_with_kvcache_twodecoders import FramePack3Rv3_Abla_Mem_with_KVCache_TwoDecoders

from .framepack3_abla_mem_mimic_cut3r import FramePack3Rv3_Abla_Mem_cut3r
from .framepack3_abla_mem_mimic_cut3r_blkstate import FramePack3Rv3_Abla_Mem_cut3r_blkstate
from .framepack3_abla_mem_mimic_cut3r_hasrope import FramePack3Rv3_Abla_Mem_cut3r_hasrope

from .framepack3_abla_mem_mimic_cut3r_v2init import FramePack3Rv3_Abla_Mem_cut3r_v2init
from .framepack3_abla_mem_mimic_cut3r_v2init_withf0 import FramePack3Rv3_Abla_Mem_cut3r_v2init_withf0
from .framepack3_abla_mem_mimic_cut3r_blkstate_nope import FramePack3Rv3_Abla_Mem_cut3r_blkstate_nope
from .framepack3_abla_mem_with_kvcache_twodecoders_final import FramePack3Rv3_Abla_Mem_with_KVCache_TwoDecoders_final
from .backbone_croco_must3r_noinj_nope import Must3r_reimpl_noinj_nope

from .backbone_croco_must3r_noinj import Must3r_reimpl_noinj
from .backbone_stream3r_flexattn import Stream3r_flexattn

BACKBONES: dict[str, Backbone[Any]] = {
    "resnet": BackboneResnet,
    "dino": BackboneDino,
    "croco": AsymmetricCroCo,
    "croco_multi": AsymmetricCroCoMulti,
    "croco_multi_rw": AsymmetricCroCoMulti_RW,
    "croco_multi_rw_cut3r": AsymmetricCroCoMulti_RW_CUT3R,
    "croco_multi_rw_ar_causal_v1":
    AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn,
    "croco_multi_rw_ar_causal_v1.5_frame0Mem":
    AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn_frame0_as_memory,
    "croco_multi_rw_ar_causal_v2_oneCABlk":
    AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn_OneCABlk,
    "croco_multi_rw_ar_causal_v1.6_frame0Mem":
    AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn_frame0_as_memory_changeAttn,
    # must3r
    'must3r_reimpl': Must3r_reimpl,
    'hybrid_attn': HybridCausalFA,
    'FramePack3R': FramePack3R,
    'FramePack3Rv2': FramePack3Rv2,
    'FramePack3Rv3': FramePack3Rv3,
    'FramePack3Rv3_Abla_Mem_with_KVCache0': FramePack3Rv3_Abla_Mem_with_KVCache0, # state only
    'FramePack3Rv3_Abla_Mem_with_KVCache': FramePack3Rv3_Abla_Mem_with_KVCache, # state + all prev kvcache
    'FramePack3Rv3_Abla_Mem_with_KVCache3_stride1': FramePack3Rv3_Abla_Mem_with_KVCache3_stride1, # state + 3 prev kvcache
    'FramePack3Rv3_Abla_Mem_with_KVCache3_stride3': FramePack3Rv3_Abla_Mem_with_KVCache3_stride3,
    'FramePack3Rv3_Abla_FA_no_state': FramePack3Rv3_Abla_FA_no_state,
    'FramePack3Rv3_Abla_FA_with_state': FramePack3Rv3_Abla_FA_with_state,
    'FramePack3Rv3_Abla_Mem_with_KVCache_TwoDecoders': FramePack3Rv3_Abla_Mem_with_KVCache_TwoDecoders,
    'FramePack3Rv3_Abla_Mem_cut3r': FramePack3Rv3_Abla_Mem_cut3r,
    'FramePack3Rv3_Abla_Mem_cut3r_blkstate': FramePack3Rv3_Abla_Mem_cut3r_blkstate,
    'FramePack3Rv3_Abla_Mem_cut3r_hasrope': FramePack3Rv3_Abla_Mem_cut3r_hasrope,
    'FramePack3Rv3_Abla_Mem_cut3r_v2init': FramePack3Rv3_Abla_Mem_cut3r_v2init,
    'FramePack3Rv3_Abla_Mem_cut3r_v2init_withf0': FramePack3Rv3_Abla_Mem_cut3r_v2init_withf0,
    'FramePack3Rv3_Abla_Mem_cut3r_v2init_nope': FramePack3Rv3_Abla_Mem_cut3r_blkstate_nope,
    'FramePack3Rv3_Abla_Mem_cut3r_twodecoders_final': FramePack3Rv3_Abla_Mem_with_KVCache_TwoDecoders_final,
    'Must3r_reimpl_noinj': Must3r_reimpl_noinj,
    'Must3r_reimpl_noinj_nope': Must3r_reimpl_noinj_nope,
    'Stream3r_flexattn': Stream3r_flexattn,
}

BackboneCfg = BackboneResnetCfg | BackboneDinoCfg | BackboneCrocoCfg


def get_backbone(cfg: BackboneCfg, d_in: int = 3) -> nn.Module:
    return BACKBONES[cfg.name](cfg, d_in)
