# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# global alignment optimization wrapper function
# --------------------------------------------------------
from enum import Enum

from .optimizer import PointCloudOptimizer
from .modular_optimizer import ModularPointCloudOptimizer
from .pair_viewer import PairViewer
# gs-based optimizer
from .gs_optimizer import GSOptimizer


class GlobalAlignerMode(Enum):
    PointCloudOptimizer = "PointCloudOptimizer"
    ModularPointCloudOptimizer = "ModularPointCloudOptimizer"
    PairViewer = "PairViewer"
    GSOptimizer = "GSOptimizer"
    GSWindowOptimizer = "GSWindowOptimizer"


def global_aligner(dust3r_output, device, mode=GlobalAlignerMode.PointCloudOptimizer, **optim_kw):
    # extract all inputs
    view1, view2, pred1, pred2 = [dust3r_output[k] for k in 'view1 view2 pred1 pred2'.split()] # ! init the predictions
    # build the optimizer
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        net = PointCloudOptimizer(view1, view2, pred1, pred2, **optim_kw).to(device)
    elif mode == GlobalAlignerMode.ModularPointCloudOptimizer:
        net = ModularPointCloudOptimizer(view1, view2, pred1, pred2, **optim_kw).to(device)
    elif mode == GlobalAlignerMode.PairViewer:
        net = PairViewer(view1, view2, pred1, pred2, **optim_kw).to(device)
    # ! for gs-based global optimization
    elif mode == GlobalAlignerMode.GSWindowOptimizer:
        # net = ModularPointCloudOptimizer(view1, view2, pred1, pred2, **optim_kw).to(device)
        raise NotImplementedError(f'Unknown mode {mode}')
    elif mode == GlobalAlignerMode.GSOptimizer:
        net = GSOptimizer(view1, view2, pred1, pred2, **optim_kw).to(device)
    else:
        raise NotImplementedError(f'Unknown mode {mode}')

    return net
