from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from .croco.blocks import DecoderBlock
from .croco.croco import CroCoNet
from .croco.misc import fill_default_args, freeze_all_params, transpose_to_landscape, is_symmetrized, interleave, \
    make_batch_symmetric
from .croco.patch_embed import get_patch_embed
from .backbone import Backbone
from ....geometry.camera_emb import get_intrinsic_embedding

from pdb import set_trace as st

# from src.model.vggt.utils import slice_expand_and_flatten

inf = float('inf')


croco_params = {
    'ViTLarge_BaseDecoder': {
        'enc_depth': 24,
        'dec_depth': 12,
        'enc_embed_dim': 1024,
        'dec_embed_dim': 768,
        'enc_num_heads': 16,
        'dec_num_heads': 12,
        'pos_embed': 'RoPE100',
        'img_size': (512, 512),
        'qk_norm': False,
    },
}

croco_params['ViTLarge_BaseDecoder_qknorm'] = {
    **croco_params['ViTLarge_BaseDecoder'],
    'qk_norm': True,
}

# st()

default_dust3r_params = {
    'enc_depth': 24,
    'dec_depth': 12,
    'enc_embed_dim': 1024,
    'dec_embed_dim': 768,
    'enc_num_heads': 16,
    'dec_num_heads': 12,
    'pos_embed': 'RoPE100',
    'patch_embed_cls': 'PatchEmbedDust3R',
    'img_size': (512, 512),
    'head_type': 'dpt',
    'output_mode': 'pts3d',
    'depth_mode': ('exp', -inf, inf),
    'conf_mode': ('exp', 1, inf)
}


@dataclass
class BackboneCrocoCfg:
    # name: Literal["croco", "croco_multi", "croco_multi_rw", "croco_multi_rw_cut3r", "croco_multi_rw_ar_causal_v1", "croco_multi_rw_ar_causal_v1.5_oneCABlk", "croco_multi_rw_ar_causal_v2_oneCABlk"]
    name: str
    model: Literal["ViTLarge_BaseDecoder_qknorm", "ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]  # keep interface for the last two models, but they are not supported # !
    patch_embed_cls: str = 'PatchEmbedDust3R'  # PatchEmbedDust3R or ManyAR_PatchEmbed
    asymmetry_decoder: bool = True
    intrinsics_embed_loc: Literal["encoder", "decoder", "none", "dpt"] = 'none'
    intrinsics_embed_degree: int = 0
    intrinsics_embed_type: Literal["pixelwise", "linear", "token", "none", "linear-from-pred"] = 'token'  # linear or dpt
    enable_camera_token: bool = False # ! if predicting camera as in vgg-t/cut3r/flare
    enable_temporal_pe: bool = False # whether to enable temporal PE when using v>2 as the input
    # qk_norm: bool = False # for stable bf16 training


class AsymmetricCroCo(CroCoNet):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    """

    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:

        self.intrinsics_embed_loc = cfg.intrinsics_embed_loc
        self.intrinsics_embed_degree = cfg.intrinsics_embed_degree
        self.intrinsics_embed_type = cfg.intrinsics_embed_type
        self.intrinsics_embed_encoder_dim = 0
        self.intrinsics_embed_decoder_dim = 0
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_encoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3
        elif self.intrinsics_embed_loc == 'decoder' and self.intrinsics_embed_type == 'pixelwise':
            self.intrinsics_embed_decoder_dim = (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3

        self.patch_embed_cls = cfg.patch_embed_cls
        self.croco_args = fill_default_args(croco_params[cfg.model], CroCoNet.__init__)

        super().__init__(**croco_params[cfg.model])

        if cfg.asymmetry_decoder:
            self.dec_blocks2 = deepcopy(self.dec_blocks)  # This is used in DUSt3R and MASt3R

        if self.intrinsics_embed_type == 'linear' or self.intrinsics_embed_type == 'token':
            self.intrinsic_encoder = nn.Linear(9, 1024)

        # self.set_freeze(freeze)

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3):
        in_chans = in_chans + self.intrinsics_embed_encoder_dim
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans)

    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        enc_embed_dim = enc_embed_dim + self.intrinsics_embed_decoder_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
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
        assert freeze in ['none', 'mask', 'encoder'], f"unexpected freeze={freeze}"
        to_be_frozen = {
            'none':     [],
            'mask':     [self.mask_token],
            'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
            'encoder_decoder':  [self.mask_token, self.patch_embed, self.enc_blocks, self.enc_norm, self.decoder_embed, self.dec_blocks, self.dec_blocks2, self.dec_norm],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def _encode_image(self, image, true_shape, intrinsics_embed=None):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # st()

        if intrinsics_embed is not None:

            if self.intrinsics_embed_type == 'linear':
                x = x + intrinsics_embed
            elif self.intrinsics_embed_type == 'token':
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

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2, intrinsics_embed1=None, intrinsics_embed2=None):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0),
                                             torch.cat((intrinsics_embed1, intrinsics_embed2), dim=0) if intrinsics_embed1 is not None else None)
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1, intrinsics_embed1)
            out2, pos2, _ = self._encode_image(img2, true_shape2, intrinsics_embed2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2, force_asym=False):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        intrinsics_embed1 = view1.get('intrinsics_embed', None)
        intrinsics_embed2 = view2.get('intrinsics_embed', None)

        if force_asym or not is_symmetrized(view1, view2):
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2, intrinsics_embed1, intrinsics_embed2)
        else:
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2, extra_embed1=None, extra_embed2=None):
        final_output = [(f1, f2)]  # before projection

        if extra_embed1 is not None:
            f1 = torch.cat((f1, extra_embed1), dim=-1)
        if extra_embed2 is not None:
            f2 = torch.cat((f2, extra_embed2), dim=-1)

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self,
                context: dict,
                symmetrize_batch=False,
                return_views=False,
                ):
        b, v, _, h, w = context["image"].shape
        device = context["image"].device

        view1, view2 = ({'img': context["image"][:, 0]},
                        {'img': context["image"][:, 1]})

        # camera embedding in the encoder
        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'pixelwise':
            intrinsic_emb = get_intrinsic_embedding(context, degree=self.intrinsics_embed_degree)
            view1['img'] = torch.cat((view1['img'], intrinsic_emb[:, 0]), dim=1)
            view2['img'] = torch.cat((view2['img'], intrinsic_emb[:, 1]), dim=1)

        if self.intrinsics_embed_loc == 'encoder' and (self.intrinsics_embed_type == 'token' or self.intrinsics_embed_type == 'linear'):
            intrinsic_embedding = self.intrinsic_encoder(context["intrinsics"].flatten(2))
            view1['intrinsics_embed'] = intrinsic_embedding[:, 0].unsqueeze(1)
            view2['intrinsics_embed'] = intrinsic_embedding[:, 1].unsqueeze(1)

        if symmetrize_batch:
            instance_list_view1, instance_list_view2 = [0 for _ in range(b)], [1 for _ in range(b)]
            view1['instance'] = instance_list_view1
            view2['instance'] = instance_list_view2
            view1['idx'] = instance_list_view1
            view2['idx'] = instance_list_view2
            view1, view2 = make_batch_symmetric(view1, view2)

            # encode the two images --> B,S,D
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2, force_asym=False)
        else:
            # encode the two images --> B,S,D
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2, force_asym=True)

        if self.intrinsics_embed_loc == 'decoder':
            # FIXME: downsample is hardcoded to 16
            intrinsic_emb = get_intrinsic_embedding(context, degree=self.intrinsics_embed_degree, downsample=16, merge_hw=True)
            dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, intrinsic_emb[:, 0], intrinsic_emb[:, 1])
        else:
            dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        if self.intrinsics_embed_loc == 'encoder' and self.intrinsics_embed_type == 'token':
            dec1, dec2 = list(dec1), list(dec2)
            for i in range(len(dec1)):
                dec1[i] = dec1[i][:, :-1]
                dec2[i] = dec2[i][:, :-1]

        if return_views:
            return dec1, dec2, shape1, shape2, view1, view2
        return dec1, dec2, shape1, shape2

    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024
