from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset import DatasetCfgWrapper
from .dataset.data_module import DataLoaderCfg, DatasetDust3rCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg

from pdb import set_trace as st


@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int
    save_weights_only: bool


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_nodes: int = 1
    precision: str = '32-true'
    accumulate_grad_batches: int = 1 # grad acc


@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "test"]
    dataset: list[DatasetCfgWrapper]
    data_loader: DataLoaderCfg
    # dataset_cut3r: DatasetDust3rCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    train: TrainCfg
    seed: int


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    # st()
    cfg = from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    ) # add visualize: false
    return cfg


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def separate_dataset_cfg_wrappers(joined: dict) -> list[DatasetCfgWrapper]:
# def separate_dataset_cfg_wrappers(joined) -> list[DatasetCfgWrapper]:

    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: DatasetCfgWrapper

    # st() # if joined is None, some key in the yaml has no children
    dataset_cfgs_to_return = [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy for k, v in joined.items()
    ]
    return dataset_cfgs_to_return


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers,
         list[DatasetCfgWrapper]: separate_dataset_cfg_wrappers},
    )
