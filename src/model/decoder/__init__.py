from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg

DECODERS = {
    "splatting_cuda": DecoderSplattingCUDA,
}

DecoderCfg = DecoderSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg)
