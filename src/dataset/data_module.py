import random
from pdb import set_trace as st
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch import Generator, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from src.dust3r.datasets import get_data_loader

from ..misc.step_tracker import StepTracker
from . import DatasetCfgWrapper, get_dataset
from .types import DataShim, Stage
from .validation_wrapper import ValidationWrapper

from omegaconf import DictConfig, OmegaConf


def get_data_shim(encoder: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    # st()
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


# ! from dust3r
def build_dataset(dataset,
                  batch_size,
                  num_workers,
                  world_size,
                  rank,
                  test=False,
                  fixed_length=False):
    split = ["Train", "Test"][test]
    print(f"Building {split} Data loader for dataset: {dataset}")
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        #  pin_mem=False,
        #  shuffle=not (test),
        shuffle=True,
        drop_last=not (test),
        world_size=world_size,
        rank=rank,
        fixed_length=fixed_length,
        test=test)
    return loader


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg


@dataclass
class DatasetDust3rCfg:
    train_datset: str
    test_datsaet: str
    fixed_length: bool


DatasetShim = Callable[[Dataset, Stage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):
    dataset_cfgs: list[DatasetCfgWrapper]
    data_loader_cfg: DataLoaderCfg
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    global_rank: int

    def __init__(
        self,
        dataset_cfgs: list[DatasetCfgWrapper],
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
        global_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_cfgs = dataset_cfgs
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(
            self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        datasets = get_dataset(self.dataset_cfgs, "train", self.step_tracker)
        data_loaders = []
        for dataset in datasets:
            dataset = self.dataset_shim(dataset, "train")
            data_loaders.append(
                DataLoader(
                    dataset,
                    self.data_loader_cfg.train.batch_size,
                    shuffle=not isinstance(dataset, IterableDataset),
                    num_workers=self.data_loader_cfg.train.num_workers,
                    generator=self.get_generator(self.data_loader_cfg.train),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(
                        self.data_loader_cfg.train),
                ))
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]

    def val_dataloader(self):
        datasets = get_dataset(self.dataset_cfgs, "val", self.step_tracker)
        data_loaders = []
        for dataset in datasets:
            dataset = self.dataset_shim(dataset, "val")
            data_loaders.append(
                DataLoader(
                    ValidationWrapper(dataset, 1),
                    self.data_loader_cfg.val.batch_size,
                    num_workers=self.data_loader_cfg.val.num_workers,
                    generator=self.get_generator(self.data_loader_cfg.val),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(
                        self.data_loader_cfg.val),
                ))
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]

    def test_dataloader(self):
        datasets = get_dataset(self.dataset_cfgs, "test", self.step_tracker)
        data_loaders = []
        for dataset in datasets:
            dataset = self.dataset_shim(dataset, "test")
            data_loaders.append(
                DataLoader(
                    dataset,
                    self.data_loader_cfg.test.batch_size,
                    num_workers=self.data_loader_cfg.test.num_workers,
                    generator=self.get_generator(self.data_loader_cfg.test),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.get_persistent(
                        self.data_loader_cfg.test),
                ))
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]


class DataModuleDust3r(LightningDataModule):
    """load dataset using dust3r format config
    """

    # ! not required here
    pixelsplat_dataset_cfgs: list[
        DatasetCfgWrapper]  # for creating the dataset
    # data_loader_cfg: DataLoaderCfg # loader configs
    # step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    hybrid_loading: bool
    noposplat_training: bool  # avoid loading dust3r datasets

    # global_rank: int

    def __init__(
        self,
        pixelsplat_dataset_cfgs: list[DatasetCfgWrapper],  # dataset_cut3r
        dut3r_dataset_cfg: DictConfig,  # dataset_cut3r
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
        global_rank: int = 0,
        world_size: int = 1,
        hybrid_loading: bool = False,
        noposplat_training: bool = False
        # fixed_length: bool = False,
    ) -> None:
        super().__init__()

        # self.train_dataset_cfg, self.test_dataset_cfg = dataset_cfgs # [cfg_dict.train_dataset, cfg_dict.test_dataset]
        # st()
        self.hybrid_loading = hybrid_loading
        self.pixelsplat_dataset_cfgs = pixelsplat_dataset_cfgs
        self.dust3r_dataset_cfg = dut3r_dataset_cfg
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank
        self.world_size = world_size
        self.noposplat_training = noposplat_training
        # self.fixed_length = fixed_length

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(
            self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def _build_train_dataloaders_pixelsplat(self):
        datasets = get_dataset(self.pixelsplat_dataset_cfgs, "train",
                               self.step_tracker)
        data_loaders = []
        for dataset in datasets:
            dataset = self.dataset_shim(dataset, "train")
            data_loaders.append(
                DataLoader(
                    dataset,
                    self.data_loader_cfg.train.batch_size,
                    shuffle=not isinstance(dataset, IterableDataset),
                    num_workers=self.data_loader_cfg.train.num_workers,
                    generator=self.get_generator(self.data_loader_cfg.train),
                    worker_init_fn=worker_init_fn,
                    persistent_workers=self.data_loader_cfg.train.num_workers
                    > 0,
                    # self.get_persistent(
                    # self.data_loader_cfg.train),
                ))
        return data_loaders if len(data_loaders) > 1 else data_loaders[0]

    def train_dataloader(self):

        if not self.noposplat_training:
            dust3r_data_loader = build_dataset(  # ! return the dataloader here
                self.dust3r_dataset_cfg.train_dataset,
                batch_size=self.data_loader_cfg.train.batch_size,
                num_workers=self.data_loader_cfg.train.num_workers,
                world_size=self.world_size,
                rank=self.global_rank,
                test=False,
                fixed_length=self.dust3r_dataset_cfg.fixed_length)

            self._set_epoch_to_loader(
                dust3r_data_loader
            )  # failed on PL loading. locate the problem.
        else:
            dust3r_data_loader = None
            pixelsplat_dataloaders = self._build_train_dataloaders_pixelsplat()
            if not isinstance(pixelsplat_dataloaders, list):
                pixelsplat_dataloaders = [pixelsplat_dataloaders]
            return pixelsplat_dataloaders

        if self.hybrid_loading:
            pixelsplat_dataloaders = self._build_train_dataloaders_pixelsplat()
            if not isinstance(pixelsplat_dataloaders, list):
                pixelsplat_dataloaders = [pixelsplat_dataloaders]

            if dust3r_data_loader is not None:
                dataloaders = [*pixelsplat_dataloaders, dust3r_data_loader]
            else:
                return pixelsplat_dataloaders
            return dataloaders

        else:
            return dust3r_data_loader
            # return dataloaders

    def val_dataloader(self):

        data_loader = build_dataset(  # ! return the dataloader here
            self.dust3r_dataset_cfg.test_dataset,
            batch_size=self.data_loader_cfg.val.batch_size,
            num_workers=self.data_loader_cfg.val.num_workers,
            world_size=self.world_size,
            rank=self.global_rank,
            test=True,
            fixed_length=True,
        )

        self._set_epoch_to_loader(data_loader)

        return data_loader

    def test_dataloader(self):

        data_loader = build_dataset(  # ! return the dataloader here
            self.dust3r_dataset_cfg.test_dataset,
            batch_size=self.data_loader_cfg.test.batch_size,
            num_workers=self.data_loader_cfg.test.num_workers,
            world_size=self.world_size,
            rank=self.global_rank,
            test=True,
            fixed_length=True)  # self.dataset_cfg.fixed_length

        self._set_epoch_to_loader(data_loader)

        return data_loader

    def _set_epoch_to_loader(self, data_loader, epoch=0):
        # fix to zero here

        # https://github.com/naver/dust3r/blob/c9e9336a6ba7c1f1873f9295852cea6dffaf770d/dust3r/training.py#L282
        if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset,
                                                       "set_epoch"):
            data_loader.dataset.set_epoch(epoch)

        if (hasattr(data_loader, "batch_sampler")
                and hasattr(data_loader.batch_sampler, "sampler")
                and hasattr(data_loader.batch_sampler.sampler, "set_epoch")):
            data_loader.batch_sampler.sampler.set_epoch(
                epoch)  # cut3r sets to 0 here

        if hasattr(data_loader, 'sampler') and hasattr(data_loader.sampler,
                                                       'set_epoch'):
            data_loader.sampler.set_epoch(epoch)
