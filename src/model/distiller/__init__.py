import torch

from .dust3d_backbone import Dust3R


inf = float('inf')


def get_distiller(name):
    assert name == 'dust3r' or name == 'mast3r', f"unexpected name={name}"
    distiller = Dust3R(enc_depth=24, dec_depth=12, enc_embed_dim=1024, dec_embed_dim=768, enc_num_heads=16, dec_num_heads=12, pos_embed='RoPE100', patch_embed_cls='PatchEmbedDust3R', img_size=(512, 512), head_type='dpt', output_mode='pts3d', depth_mode=('exp', -inf, inf), conf_mode=('exp', 1, inf))
    distiller = distiller.eval()

    if name == 'dust3r':
        weight_path = './pretrained_weights/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth'
    elif name == 'mast3r':
        weight_path = './pretrained_weights/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth'
    else:
        raise NotImplementedError(f"unexpected {name=}")
    ckpt_weights = torch.load(weight_path, map_location='cpu')['model']
    missing_keys, unexpected_keys = distiller.load_state_dict(ckpt_weights, strict=False if name == 'mast3r' else True)

    return distiller
