import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from dacite import Config, from_dict
from jaxtyping import Float, Int64
from torch import Tensor

from ...evaluation.evaluation_index_generator import IndexEntry
from ...global_cfg import get_cfg
from ...misc.step_tracker import StepTracker
from ..types import Stage
from .three_view_hack import add_third_context_index, add_k_context_indices
from .view_sampler import ViewSampler

from pdb import set_trace as st
import numpy as np


@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Path
    num_context_views: int
    middle_context_view: bool
    # streaming_input: bool # whether video input


class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):
    index: dict[str, IndexEntry | None]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)

        self.cfg = cfg

        dacite_config = Config(cast=[tuple])
        with cfg.index_path.open("r") as f:
            self.index = {
                k: None if v is None else from_dict(IndexEntry, v, dacite_config)
                for k, v in json.load(f).items()
            }
            # pass

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        # return_sliding_window=False,
        # sliding_window_stride=2,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")
        context_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)

        if len(entry.target) == 1 and entry.target[0] == -1: # if to visualize the whole trajectory
            target_indices = torch.arange(context_indices[0], context_indices[1], step=1, dtype=torch.int64, device=device)
        else:
            target_indices = torch.tensor(entry.target, dtype=torch.int64, device=device)

        overlap = entry.overlap if isinstance(entry.overlap, float) else 0.75 if entry.overlap == "large" else 0.25
        overlap = torch.tensor([overlap], dtype=torch.float32, device=device)

        # Handle 2-view index for 3 views.
        v = self.num_context_views
        # st()
        if v > len(context_indices):
            if v == 3:
                context_indices = add_third_context_index(context_indices)
            # if v == 8: # for video input only
            else:
                context_indices = add_k_context_indices(context_indices, context_views=v).to(device)
                # st()
                # pass

        return context_indices, target_indices, overlap

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return 0


    def sample_sliding_window(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        sliding_window_stride=1, # >1 not allowed for GA
        model_window_size=9, # model can process how many frames in a single feedforward
        # overlap=3, # ! whether overlap between windows. shall > 0 for global alignment.
        # return_whole_traj=False, # also return the whole trajectory for inference?
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
        Float[Tensor, " overlap"],  # overlap
    ]:
        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")

        frame_number = extrinsics.shape[0]

        # window_size = sliding_window_stride * (model_window_size) # here, 2 * (9-1) = 16
        # window_step = window_size//2

        # ! debug window size, make it 1/2 or 1/3 of the video number for better visualization

        procrustes_alingment = False        
        # procrustes_alingment = True # split inference on several windows

        if procrustes_alingment:
            # ! window=2
            window_size = frame_number // 2
            window_step = window_size-1 # only one frame overlap here

            # window_size = frame_number // 10
            # window_step = window_size-1 # only one frame overlap here

        else: # full video inference
            window_size = frame_number
            window_step = frame_number + 1 # no overlap here

        # st()

        v = self.num_context_views # 9 here
        # assert v > len(context_indices)

        context_indices_list = []
        target_indices_list = []
        overlap_list = []

        # mons3r pair sampling strategy: https://github.com/Junyi42/monst3r/blob/425422df3798f71b129a3737fc7980923586d31f/dust3r/image_pairs.py#L31
        if procrustes_alingment:
            for window_left in range(0, frame_number-window_size, window_step): # the middle frame is the end of another window

                # Handle 2-view index for 3 views.

                overlap = entry.overlap if isinstance(entry.overlap, float) else 0.75 if entry.overlap == "large" else 0.25
                overlap = torch.tensor([overlap], dtype=torch.float32, device=device)

                context_indices_left_right = torch.Tensor([window_left, window_size+window_left-1])

                context_indices = add_k_context_indices(context_indices_left_right, 
                                                        context_views=v, 
                                                        middle_context_view=self.cfg.middle_context_view).to(device)

                # ! evaluate all indices? yes here.
                target_indices = torch.arange(context_indices_left_right[0], context_indices_left_right[1]+1, step=1, dtype=torch.int64, device=device)

                context_indices_list.append(context_indices)
                target_indices_list.append(target_indices)
                overlap_list.append(overlap)
        else:

                overlap = entry.overlap if isinstance(entry.overlap, float) else 0.75 if entry.overlap == "large" else 0.25
                overlap = torch.tensor([overlap], dtype=torch.float32, device=device)

                context_indices_left_right = torch.Tensor([0, frame_number-1])

                context_indices = add_k_context_indices(context_indices_left_right, 
                                                        context_views=v, 
                                                        middle_context_view=self.cfg.middle_context_view).to(device)

                # ! evaluate all indices? yes here.
                target_indices = torch.arange(context_indices_left_right[0], context_indices_left_right[1]+1, step=1, dtype=torch.int64, device=device)

                context_indices_list.append(context_indices)
                target_indices_list.append(target_indices)
                overlap_list.append(overlap)


        # st()
        # pass

        # return context_indices, target_indices, overlap
        # st()
        return context_indices_list, target_indices_list, overlap_list