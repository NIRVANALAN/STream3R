from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor
import numpy as np

from .view_sampler import ViewSampler

# def generate_extra_views(index_context_left, index_context_right, num_extra_views):
#     available_indices = torch.arange(index_context_left + 1, index_context_right)
#     shuffled_indices = torch.randperm(len(available_indices))
#     selected_indices = shuffled_indices[:num_extra_views]
#     extra_views = available_indices[selected_indices].tolist()
#     return extra_views
def generate_extra_views(index_context_left, index_context_right, num_extra_views):
    available_indices = np.arange(index_context_left + 1, index_context_right)
    np.random.shuffle(available_indices)
    extra_views = available_indices[:num_extra_views].tolist()
    return extra_views


@dataclass
class ViewSamplerBoundedCfg:
    name: Literal["bounded"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    middle_context_view: bool
    streaming_input: bool # whether video input


class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        num_views, _, _ = extrinsics.shape # >300 frames for dl3dv, ~150 for re10k

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.warm_up_steps > 0: # 9375
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views

        # Pick the gap between the context views.
        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, max_gap)

        # if max_gap < num_views: # e.g., some video only 17 frames
        #     max_gap = 1 # stride=1

        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)
        if max_gap < min_gap:
            # st()
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # if load more than bino-view, change the context_gap
        # if self.cfg.middle_context_view:
        if self.cfg.num_context_views > 2:
            context_gap *= (self.cfg.num_context_views - 1)

        # Pick the left and right context indices.
        try:
            index_context_left = torch.randint(
                num_views if self.cameras_are_circular else max(1, num_views - context_gap),
                size=tuple(),
                device=device,
            ).item()
        except:
            raise ValueError('{} margin wrong'.format(num_views if self.cameras_are_circular else max(1, num_views - context_gap)))
        if self.stage == "test":
            index_context_left = index_context_left * 0
        index_context_right = min(num_views-1, index_context_left + context_gap)

        if self.is_overfitting:
            index_context_left *= 0
            index_context_right *= 0
            index_context_right += max_gap

        # st()
        # Pick the target view indices.
        if self.stage == "test":
        # if self.stage != "train":
            # st()
            # When testing, pick all.
            index_target = torch.arange(
                index_context_left,
                index_context_right + 1,
                device=device,
            )
        # if self.stage == "vis":
        #     # When testing, pick all.
        #     index_target = torch.arange(
        #         index_context_left,
        #         index_context_right,
        #         device=device,
        #     )
        else:
            # When training or validating (visualizing), pick at random.
            # index_target = torch.randint(
            #     index_context_left + self.cfg.min_distance_to_context_views,
            #     index_context_right + 1 - self.cfg.min_distance_to_context_views,
            #     size=(self.cfg.num_target_views,),
            #     device=device,
            # ) # ! sometimes it is duplicated, use perm
            available_indices = torch.arange(index_context_left + self.cfg.min_distance_to_context_views, index_context_right + 1 - self.cfg.min_distance_to_context_views)
            shuffled_indices = torch.randperm(len(available_indices))
            index_target = available_indices[shuffled_indices][:self.cfg.num_target_views]

        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            index_target %= num_views
            index_context_right %= num_views

        # If more than two context views are desired, pick extra context views between
        # the left and right ones.

        fixed_length = False # ! hard coded, to resolve the AR registration failure bug.

        # st()
        # if not fixed_length:
        if False:
            num_context_views = self.schedule(
                2, self.cfg.num_context_views,
            )
            # if num_context_views > 2 and torch.rand(1) > 0.75:
            if num_context_views > 2 and self.global_step % 5 == 0:
                num_context_views = 2 # ! force good registration here

        else:
            num_context_views = self.cfg.num_context_views

        if self.cfg.num_context_views > 2:
            num_extra_views = num_context_views - 2
            extra_views = []
            # while len(set(extra_views)) != num_extra_views: # ! set will cancel duplicated views => infinite loop here.
            #     extra_views = torch.randint(
            #         index_context_left + 1,
            #         index_context_right,
            #         (num_extra_views,),
            #     ).tolist()
            # ! avoid while loop here
            extra_views = generate_extra_views(index_context_left, index_context_right, num_extra_views)
        else:
            extra_views = []
        
        context_indices = [index_context_left, *extra_views, index_context_right]

        if self.cfg.streaming_input:
            context_indices.sort() # follow the causal order

        if self.cfg.middle_context_view: # follow pku slam3r setting.
            middle_idx = len(context_indices)//2
            context_indices = [context_indices[middle_idx]] + context_indices[:middle_idx] + context_indices[middle_idx+1:]

        context_indices = torch.tensor(context_indices)

        overlap = torch.tensor([0.5], dtype=torch.float32, device=device)  # dummy

        return (
            context_indices,
            index_target,
            overlap
        )

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
