from dataclasses import dataclass

from jaxtyping import Float
from pdb import set_trace as st
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    average_over_mask: bool


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):

    def forward(self,
                prediction: DecoderOutput,
                batch: BatchedExample,
                gaussians: Gaussians,
                global_step: int,
                mask=None) -> Float[Tensor, ""]:
        delta_square = (prediction.color - batch["target"]["image"])**2

        if not self.cfg.average_over_mask:
            mse_loss = (delta_square).mean()
        else:
            if mask is not None:
                assert (mask is not None and mask.shape[2] == 1)  # B v 1 h w
                if self.cfg.average_over_mask:
                    mse_loss = (delta_square * mask).sum() / mask.sum()
                else:
                    mse_loss = (delta_square * mask).mean()

        return self.cfg.weight * mse_loss
