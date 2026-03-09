

# -------------------------------------------------------------------
# Main class for the implementation of the global alignment on the gs
# -------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn

from src.dust3r.cloud_opt.base_opt import BasePCOptimizer
from src.dust3r.utils.geometry import xy_grid, geotrf
from src.dust3r.utils.device import to_cpu, to_numpy

from .optimizer import PointCloudOptimizer, ParameterStack

from pdb import set_trace as st


class GSOptimizer(PointCloudOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, optimize_pp=False, focal_break=20, **kwargs):
        super().__init__(*args, optimize_pp=optimize_pp, focal_break=focal_break, **kwargs)

    def forward(self):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(raw=True)

        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the less
        li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j

        return li + lj