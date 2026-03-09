from dataclasses import dataclass

import torch
from pdb import set_trace as st
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int
    average_over_mask: bool
    apply_mask_before_mask: bool
    spatial: bool


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        # if self.cfg.average_over_mask:
        #     self.lpips = LPIPS(net="vgg", spatial=True)
        # else:
        self.lpips = LPIPS(net="vgg", spatial=self.cfg.spatial, verbose=True)
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        mask=None,
        # average_over_mask=True,
    ) -> Float[Tensor, ""]:
        image = batch["target"]["image"]

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step or self.cfg.weight == 0:
            return torch.tensor(0, dtype=torch.float32, device=image.device)
        

        if self.cfg.apply_mask_before_mask:
            assert mask is not None

            prediction.color = prediction.color * mask
            image = image * mask

            # loss = self.lpips.forward(
            #     rearrange(prediction.color, "b v c h w -> (b v) c h w"),
            #     rearrange(image, "b v c h w -> (b v) c h w"),
            #     normalize=True,  # force input in [0,1] here
            # )

        # else:
        
        # st()
        loss = self.lpips.forward(
            rearrange(prediction.color, "b v c h w -> (b v) c h w"),
            rearrange(image, "b v c h w -> (b v) c h w"),
            normalize=True,  # force input in [0,1] here
        )

        if mask is not None and not self.cfg.apply_mask_before_mask:

            assert (mask is not None and mask.shape[2] == 1)  # B v 1 h w
            mask = rearrange(mask, "b v 1 h w -> (b v) 1 h w")
            if self.cfg.average_over_mask:
                # assert mask is not None
                lpips_loss = (loss * mask).sum() / mask.sum()
            else:
                lpips_loss = (loss * mask).mean()
        else:
            # return self.cfg.weight * loss.mean()
            lpips_loss = loss.mean()

        return lpips_loss * self.cfg.weight
