from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from torch import nn

from .croco.blocks import DecoderBlock
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

from src.model.vggt.utils.helper_fn import slice_expand_and_flatten

inf = float('inf')

# croco_params = {
#     'ViTLarge_BaseDecoder': {
#         'enc_depth': 24,
#         'dec_depth': 12,
#         'enc_embed_dim': 1024,
#         'dec_embed_dim': 768,
#         'enc_num_heads': 16,
#         'dec_num_heads': 12,
#         'pos_embed': 'RoPE100',
#         'img_size': (512, 512),
#         'qk_norm': False,
#     },
# }

# default_dust3r_params = {
#     'enc_depth': 24,
#     'dec_depth': 12,
#     'enc_embed_dim': 1024,
#     'dec_embed_dim': 768,
#     'enc_num_heads': 16,
#     'dec_num_heads': 12,
#     'pos_embed': 'RoPE100',
#     'patch_embed_cls': 'PatchEmbedDust3R',
#     'img_size': (512, 512),
#     'head_type': 'dpt',
#     'output_mode': 'pts3d',
#     'depth_mode': ('exp', -inf, inf),
#     'conf_mode': ('exp', 1, inf)
# }

# @dataclass
# class BackboneCrocoCfg:
#     name: Literal["croco"]
#     model: Literal["ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]  # keep interface for the last two models, but they are not supported
#     patch_embed_cls: str = 'PatchEmbedDust3R'  # PatchEmbedDust3R or ManyAR_PatchEmbed
#     asymmetry_decoder: bool = True
#     intrinsics_embed_loc: Literal["encoder", "decoder", "none"] = 'none'
#     intrinsics_embed_degree: int = 0
#     intrinsics_embed_type: Literal["pixelwise", "linear", "token"] = 'token'  # linear or dpt
#     enable_temporal_pe: bool = False # whether to enable temporal PE when using v>2 as the input
#     qk_norm: bool = False # for stable bf16 training


class AsymmetricCroCoMulti(CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:

        self.cfg = cfg
        self.intrinsics_embed_loc = cfg.intrinsics_embed_loc
        self.intrinsics_embed_degree = cfg.intrinsics_embed_degree
        self.intrinsics_embed_type = cfg.intrinsics_embed_type
        self.intrinsics_embed_encoder_dim = 0
        self.intrinsics_embed_decoder_dim = 0
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_encoder_dim = (
                self.intrinsics_embed_degree +
                1)**2 if self.intrinsics_embed_degree > 0 else 3
        elif self.intrinsics_embed_loc == 'decoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_decoder_dim = (
                self.intrinsics_embed_degree +
                1)**2 if self.intrinsics_embed_degree > 0 else 3

        self.patch_embed_cls = cfg.patch_embed_cls
        self.croco_args = fill_default_args(croco_params[cfg.model],
                                            CroCoNet.__init__)

        super().__init__(**croco_params[cfg.model])

        if cfg.asymmetry_decoder:
            self.dec_blocks2 = deepcopy(
                self.dec_blocks)  # This is used in DUSt3R and MASt3R

        if 'linear' in self.intrinsics_embed_type or self.intrinsics_embed_type == 'token':
            if self.intrinsics_embed_loc in ['encoder', 'decoder']:
                self.intrinsic_encoder = nn.Linear(9, 1024)
            else:
                self.intrinsic_encoder = nn.Linear(9, 768)  # for dpt

        # temporal PE
        self.enable_temporal_pe = cfg.enable_temporal_pe
        if self.enable_temporal_pe:
            embed_dim_temporal = self.dec_embed_dim
            # temporal_interpolation_scale = 1.0
            # https://github.com/huggingface/diffusers/blob/edb8c1bce67e81f0de90a7e4c16b2f6537d39f2d/src/diffusers/models/embeddings.py#L859C1-L862C10
            # temporal_size = 8 # TODO, hard coded now.
            temporal_size = 9  # TODO, hard coded now.
            middle_idx = temporal_size // 2
            # grid_t = torch.arange(temporal_size, dtype=torch.float32) / temporal_interpolation_scale
            grid_t = list(range(0, temporal_size))
            grid_t = torch.Tensor([grid_t[middle_idx]] + grid_t[:middle_idx] +
                                  grid_t[middle_idx + 1:]).to(torch.float32)
            pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(
                embed_dim_temporal, grid_t,
                output_type="pt")  # shape: temporal_size * 768

            # ! TODO, hard coded. middle-frame f0 enabled here.
            self.pos_embed_temporal = pos_embed_temporal.to(
                torch.float32).unsqueeze(0)  # type: ignore

        # ! predicting camera also
        self.enable_camera_token = cfg.enable_camera_token

        if self.enable_camera_token:
            # https://github.com/facebookresearch/vggt/blob/a16b0760f617cd3da27ae56aa78857148f1250b7/vggt/models/aggregator.py#L125C8-L125C74
            self.camera_token = nn.Parameter(
                torch.randn(1, 2, 1, self.enc_embed_dim)
            )  # ! separate tokens for the first and other frames.
            # Initialize parameters with small values
            nn.init.normal_(self.camera_token, std=1e-6)
        else:
            self.camera_token = None

        # self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        # self.set_freeze(freeze)

    def _set_patch_embed(self,
                         img_size=224,
                         patch_size=16,
                         enc_embed_dim=768,
                         in_chans=3):
        in_chans = in_chans + self.intrinsics_embed_encoder_dim
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size,
                                           patch_size, enc_embed_dim, in_chans)

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads,
                     dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
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
                         rope=self.rope) for i in range(dec_depth)
        ])
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        assert freeze in ['none', 'mask',
                          'encoder'], f"unexpected freeze={freeze}"
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_decoder': [
                self.mask_token, self.patch_embed, self.enc_blocks,
                self.enc_norm, self.decoder_embed, self.dec_blocks,
                self.dec_blocks2, self.dec_norm
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def _encode_image(self,
                      image,
                      true_shape,
                      orig_shape,
                      intrinsics_embed=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)
        # st()

        if self.camera_token is not None:  # 'first token here'
            add_pose = pos[:, 0:1, :].clone()
            add_pose[:, :, 0] += (pos[:, -1, 0].unsqueeze(-1) + 1)
            # B = x.shape[0]
            pos = torch.cat((pos, add_pose), dim=1)
            B, V = orig_shape

            camera_token = slice_expand_and_flatten(self.camera_token, B, V)

            x = torch.cat((x, camera_token), dim=1)

        # add K injection then.
        if intrinsics_embed is not None:  # ! if adding [K],

            if self.intrinsics_embed_type == 'linear':
                x = x + intrinsics_embed
            elif self.intrinsics_embed_type == 'token':  # ! this is the case.
                x = torch.cat((x, intrinsics_embed), dim=1)
                add_pose = pos[:, 0:1, :].clone()
                add_pose[:, :, 0] += (pos[:, -1, 0].unsqueeze(-1) + 1)
                pos = torch.cat((pos, add_pose), dim=1)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)
        return x, pos, None

    def _decoder(self, feat, pose, extra_embed=None):
        b, v, l, c = feat.shape
        final_output = [feat]  # before projection
        if extra_embed is not None:
            feat = torch.cat((feat, extra_embed), dim=-1)

        # project to decoder dim
        f = rearrange(feat, "b v l c -> (b v) l c")
        f = self.decoder_embed(f)
        f = rearrange(f, "(b v) l c -> b v l c", b=b, v=v)

        final_output.append(f)

        def generate_ctx_views(x):
            b, v, l, c = x.shape
            ctx_views = x.unsqueeze(1).expand(b, v, v, l, c)
            mask = torch.arange(v).unsqueeze(0) != torch.arange(v).unsqueeze(1)
            ctx_views = ctx_views[:, mask].reshape(b, v, v - 1, l,
                                                   c)  # B, V, V-1, L, C
            ctx_views = ctx_views.flatten(
                2, 3)  # B, V, (V-1)*L, C. concatenate here.
            return ctx_views.contiguous()

        pos_ctx = generate_ctx_views(pose)
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            feat_current = final_output[-1]  # 1, 3, 257, 768
            feat_current_ctx = generate_ctx_views(
                feat_current)  # 1, 3, 514, 768
            # img1 side
            # st()
            f1, _ = blk1(feat_current[:, 0].contiguous(),
                         feat_current_ctx[:, 0].contiguous(),
                         pose[:, 0].contiguous(), pos_ctx[:, 0].contiguous())
            f1 = f1.unsqueeze(1)
            # img2 side
            f2, _ = blk2(
                rearrange(feat_current[:, 1:], "b v l c -> (b v) l c"),
                rearrange(feat_current_ctx[:, 1:], "b v l c -> (b v) l c"),
                rearrange(pose[:, 1:], "b v l c -> (b v) l c"),
                rearrange(pos_ctx[:, 1:], "b v l c -> (b v) l c"))
            f2 = rearrange(f2, "(b v) l c -> b v l c", b=b, v=v - 1)
            # store the result
            final_output.append(torch.cat((f1, f2), dim=1))  # ?

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        last_feat = rearrange(final_output[-1], "b v l c -> (b v) l c")
        last_feat = self.dec_norm(last_feat)
        final_output[-1] = rearrange(last_feat,
                                     "(b v) l c -> b v l c",
                                     b=b,
                                     v=v)
        return final_output

    def forward(
        self,
        context: dict,
        symmetrize_batch=False,
        return_views=False,
    ):
        b, v, _, h, w = context["image"].shape
        images_all = context["image"]

        # camera embedding in the encoder
        # st()
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            intrinsic_embedding = get_intrinsic_embedding(
                context, degree=self.intrinsics_embed_degree)
            images_all = torch.cat((images_all, intrinsic_embedding), dim=2)

        intrinsic_embedding_all = None
        if self.intrinsics_embed_loc == 'encoder' and (
                self.intrinsics_embed_type == 'token'
                or self.intrinsics_embed_type == 'linear'):
            intrinsic_embedding = self.intrinsic_encoder(
                context["intrinsics"].flatten(2))
            intrinsic_embedding_all = rearrange(
                intrinsic_embedding, "b v c -> (b v) c").unsqueeze(1)

        # step 1: encoder input images
        images_all = rearrange(images_all, "b v c h w -> (b v) c h w")
        shape_all = torch.tensor(images_all.shape[-2:])[None].repeat(b * v, 1)

        feat, pose, _ = self._encode_image(images_all, shape_all, (b, v),
                                           intrinsic_embedding_all)

        feat = rearrange(feat, "(b v) l c -> b v l c", b=b, v=v)
        pose = rearrange(pose, "(b v) l c -> b v l c", b=b, v=v)

        # step 2: decoder
        dec_feat = self._decoder(feat, pose)
        shape = rearrange(shape_all, "(b v) c -> b v c", b=b, v=v)
        images = rearrange(images_all, "(b v) c h w -> b v c h w", b=b, v=v)

        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'token':
            dec_feat = list(dec_feat)
            for i in range(len(dec_feat)):
                dec_feat[i] = dec_feat[i][:, :, :-1]

        return dec_feat, shape, images
        # return [feat.float() for feat in dec_feat], shape.float(), images.float() # float type for gs rendering

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024


class AsymmetricCroCoMulti_RW(AsymmetricCroCoMulti):

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        super().__init__(cfg, d_in)

    # ! follow slam3r R & W TX design.
    def _decoder(self, feat, pose, extra_embed=None):
        b, v, l, c = feat.shape
        final_output = [feat]  # before projection
        if extra_embed is not None:
            feat = torch.cat((feat, extra_embed), dim=-1)

        # project to decoder dim
        f = rearrange(feat, "b v l c -> (b v) l c")
        f = self.decoder_embed(f)
        f = rearrange(f, "(b v) l c -> b v l c", b=b, v=v)

        if self.enable_temporal_pe:
            f = f + self.pos_embed_temporal.to(f).unsqueeze(2)

        final_output.append(f)

        # ! remove if the r-w works.
        def generate_ctx_views(x):
            b, v, l, c = x.shape
            ctx_views = x.unsqueeze(1).expand(b, v, v, l, c)
            mask = torch.arange(v).unsqueeze(0) != torch.arange(v).unsqueeze(1)
            ctx_views = ctx_views[:, mask].reshape(b, v, v - 1, l,
                                                   c)  # B, V, V-1, L, C
            ctx_views = ctx_views.flatten(
                2, 3)  # B, V, (V-1)*L, C. concatenate here.
            return ctx_views.contiguous()

        pos_ctx = generate_ctx_views(pose)  # pose: torch.Size([B, 12, 257, 2])

        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            feat_current = final_output[-1]  # 1, 3, 257, 768
            feat_current_ctx = generate_ctx_views(
                feat_current)  # 1, 3, 514, 768
            # img1 side
            f1, _ = blk1(feat_current[:, 0].contiguous(),
                         feat_current_ctx[:, 0].contiguous(),
                         pose[:, 0].contiguous(),
                         pos_ctx[:,
                                 0].contiguous())  # torch.Size([1, 3855, 768])
            f1 = f1.unsqueeze(1)
            # img2 side

            # f2, _ = blk2(rearrange(feat_current[:, 1:], "b v l c -> (b v) l c"),   # 65.3gib vram
            #              rearrange(feat_current_ctx[:, 1:], "b v l c -> (b v) l c"),
            #              rearrange(pose[:, 1:], "b v l c -> (b v) l c"),
            #              rearrange(pos_ctx[:, 1:], "b v l c -> (b v) l c"))

            # ! only cross-attend to frame 0 here. 53.6 gib vram
            # st()
            f2, _ = blk2(
                rearrange(feat_current[:, 1:],
                          "b v l c -> (b v) l c").contiguous(),
                #  rearrange(feat_current_ctx[:, 1:], "b v l c -> (b v) l c"),
                repeat(feat_current[:, 0], "b l c -> (b v) l c",
                       v=v - 1).contiguous(),
                rearrange(pose[:, 1:], "b v l c -> (b v) l c").contiguous(),
                #  rearrange(pos_ctx[:, 1:], "b v l c -> (b v) l c"))
                repeat(pose[:, 0], "b l c -> (b v) l c", v=v - 1).contiguous())

            f2 = rearrange(f2, "(b v) l c -> b v l c", b=b, v=v - 1)
            # store the result
            # st()
            final_output.append(torch.cat(
                (f1, f2), dim=1))  # 1 1 L C + 1 V-1 L C -> 1 V L C

        # normalize last output
        # st()
        del final_output[1]
        last_feat = rearrange(final_output[-1], "b v l c -> (b v) l c")
        last_feat = self.dec_norm(last_feat)
        final_output[-1] = rearrange(last_feat,
                                     "(b v) l c -> b v l c",
                                     b=b,
                                     v=v)
        return final_output
