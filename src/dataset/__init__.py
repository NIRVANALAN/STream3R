from dataclasses import fields

from torch.utils.data import Dataset

from .dataset_scannet_pose import DatasetScannetPose, DatasetScannetPoseCfgWrapper
from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kWindow, DatasetRE10kCfg, DatasetRE10kCfgWrapper, DatasetDL3DVCfgWrapper, \
    DatasetScannetppCfgWrapper, DatasetBlendedMVSCfgWrapper

from .dataset_co3d import DatasetCO3DCfgWrapper, DatasetCo3D, DatasetCo3DWindow

from .types import Stage
from .view_sampler import get_view_sampler

from pdb import set_trace as st


DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "re10k_window": DatasetRE10kWindow, # for evaluation only
    "dl3dv": DatasetRE10k,
    "scannetpp": DatasetRE10k,
    "scannet_pose": DatasetScannetPose,
    # object-centric, newly added
    'co3d': DatasetCo3D,
    "co3d_window": DatasetCo3DWindow,
    # 'blendedmvs': DatasetRE10k,
}


DatasetCfgWrapper = DatasetRE10kCfgWrapper | DatasetDL3DVCfgWrapper | DatasetScannetppCfgWrapper | DatasetScannetPoseCfgWrapper | DatasetCO3DCfgWrapper | DatasetBlendedMVSCfgWrapper
DatasetCfg = DatasetRE10kCfg


def get_dataset(
    cfgs: list[DatasetCfgWrapper],
    stage: Stage,
    step_tracker: StepTracker | None,
) -> list[Dataset]:
    datasets = []

    # st()

    for cfg in cfgs:
        (field,) = fields(type(cfg))
        cfg = getattr(cfg, field.name)

        view_sampler = get_view_sampler(
            cfg.view_sampler,
            stage,
            cfg.overfit_to_scene is not None,
            cfg.cameras_are_circular,
            step_tracker,
        )
        dataset = DATASETS[cfg.name](cfg, stage, view_sampler)
        datasets.append(dataset)

    return datasets
