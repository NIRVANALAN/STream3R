import torch
from torch.utils.data.distributed import DistributedSampler

from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa
from .arkitscenes import ARKitScenes_Multi  # noqa
from .arkitscenes_highres import ARKitScenesHighRes_Multi
from .bedlam import BEDLAM_Multi
from .blendedmvs import BlendedMVS_Multi  # noqa
from .co3d import Co3d_Multi  # noqa
from .cop3d import Cop3D_Multi
from .dl3dv import DL3DV_Multi
from .dynamic_replica import DynamicReplica
from .eden import EDEN_Multi
from .hypersim import HyperSim_Multi
from .irs import IRS
from .hoi4d import HOI4D_Multi
from .mapfree import MapFree_Multi
from .megadepth import MegaDepth_Multi  # noqa
from .mp3d import MP3D_Multi
from .mvimgnet import MVImgNet_Multi
from .mvs_synth import MVS_Synth_Multi
from .omniobject3d import OmniObject3D_Multi
from .pointodyssey import PointOdyssey_Multi
from .realestate10k import RE10K_Multi
from .scannet import ScanNet_Multi
from .scannetpp import ScanNetpp_Multi  # noqa
from .smartportraits import SmartPortraits_Multi
from .spring import Spring
from .synscapes import SynScapes
from .tartanair import TartanAir_Multi
from .threedkb import ThreeDKenBurns
from .uasol import UASOL_Multi
from .urbansyn import UrbanSyn
from .unreal4k import UnReal4K_Multi
from .vkitti2 import VirtualKITTI2_Multi  # noqa
from .waymo import Waymo_Multi  # noqa
from .wildrgbd import WildRGBD_Multi  # noqa

# from spann3r, slam3r
from .habitat import Habitat
from .project_aria_seq import Aria_Seq


def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True, persistent_workers=False, multiprocessing_context=None):
    import torch
    from croco.utils.misc import get_world_size, get_rank

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        multiprocessing_context=multiprocessing_context,
    )

    return data_loader