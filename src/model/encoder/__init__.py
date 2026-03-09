from typing import Optional

from .encoder import Encoder
from .encoder_noposplat import EncoderNoPoSplatCfg, EncoderNoPoSplat
from .encoder_noposplat_multi import EncoderNoPoSplatMulti
from .visualization.encoder_visualizer import EncoderVisualizer

ENCODERS = {
    "noposplat": (EncoderNoPoSplat, None),
    "noposplat_multi": (EncoderNoPoSplatMulti, None),
}

EncoderCfg = EncoderNoPoSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
