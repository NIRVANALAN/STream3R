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


# from accelerate import Accelerator

# class CustomDistributedSampler(DistributedSampler):
#     def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
#         super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)



def get_data_loader(
    dataset,
    batch_size,
    world_size,
    rank,
    num_workers=8,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
    # accelerator: Accelerator = None,
    fixed_length=False,
    test=False,
):

   
    # def _set_epoch_to_loader(self, data_loader, epoch=0):

    #     # https://github.com/naver/dust3r/blob/c9e9336a6ba7c1f1873f9295852cea6dffaf770d/dust3r/training.py#L282
    #     st()
    #     if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset,
    #                                                    "set_epoch"):
    #         data_loader.dataset.set_epoch(epoch)

    #     if (hasattr(data_loader, "batch_sampler")
    #             and hasattr(data_loader.batch_sampler, "batch_sampler") and
    #             hasattr(data_loader.batch_sampler.batch_sampler, "set_epoch")):
    #         data_loader.batch_sampler.batch_sampler.set_epoch(
    #             epoch)  # cut3r sets to 0 here

    #     if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler,
    #                                                    'set_epoch'):
    #         data_loader.sampler.set_epoch(epoch)



    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    # def worker_init_fn(worker_id):
    #     st()
    #     worker_info = torch.utils.data.get_worker_info()
    #     # if worker_info is not None:
    #     dataset = worker_info.dataset
    #         # if hasattr(dataset, "set_epoch"):
    #     dataset.set_epoch(torch.initial_seed() % (2**32))  # Set unique seed per worker
    #             # dataset.set_epoch(0)  # Set unique seed per worker

    try:
    # if True:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle, # if false, no sampler
            drop_last=drop_last,
            # world_size=accelerator.num_processes,
            world_size=world_size,
            fixed_length=fixed_length,
            rank=rank,
        )
        # sampler = 
        shuffle = False

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
            # worker_init_fn=worker_init_fn,
        )

    except (AttributeError, NotImplementedError) as e:
    # except Exception as e:
    # else:
        # sampler = None
        # print('sampler creation failed', e)
        # st()

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=drop_last,
            # worker_init_fn=worker_init_fn,
        )

    return data_loader
