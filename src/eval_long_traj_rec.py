import json
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from lightning import Trainer

from pdb import set_trace as st

# from src.evaluation.pose_evaluator import PoseEvaluator
from src.evaluation.global_alignment_window import GlobalAlignmentWindow
from src.evaluation.rnn_window import RNN_Window
from src.evaluation.pose_evaluator_window import PoseEvaluatorWindow
from src.loss import get_losses, LossCfgWrapper
from src.misc.wandb_tools import update_checkpoint_path
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_config, ModelCfg, CheckpointingCfg, separate_loss_cfg_wrappers, \
    separate_dataset_cfg_wrappers
    from src.dataset.data_module import DataLoaderCfg, DataModule, DatasetCfgWrapper
    from src.evaluation.evaluation_cfg import EvaluationCfg
    from src.global_cfg import set_cfg


@dataclass
class RootCfg:
    evaluation: EvaluationCfg
    dataset: list[DatasetCfgWrapper]
    data_loader: DataLoaderCfg
    model: ModelCfg
    checkpointing: CheckpointingCfg
    loss: list[LossCfgWrapper]
    seed: int


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def evaluate(cfg_dict: DictConfig):
    cfg = load_typed_config(cfg_dict, RootCfg,
                            {list[LossCfgWrapper]: separate_loss_cfg_wrappers,
                             list[DatasetCfgWrapper]: separate_dataset_cfg_wrappers},)
    set_cfg(cfg_dict)
    torch.manual_seed(cfg.seed)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    ckpt_weights = torch.load(cfg.checkpointing.load, map_location='cpu')['state_dict']
    # remove the prefix "encoder.", need to judge if is at start of key
    ckpt_weights = {k[8:] if k.startswith("encoder.") else k: v for k, v in ckpt_weights.items()}
    missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=True)

    trainer = Trainer(max_epochs=-1, accelerator="gpu", inference_mode=False)
    
    # if cfg.evaluation.window_estimator:
    #     PoseEvaluatorCls = PoseEvaluatorWindow
    # else: # existing pair-wise estimator
    #     PoseEvaluatorCls = PoseEvaluator

    eval_class = cfg.evaluation.eval_cls
    if eval_class == 'GlobalAlignmentWindow':
        PoseEvaluatorCls = GlobalAlignmentWindow
    elif eval_class == 'RNN_Window':
        PoseEvaluatorCls = RNN_Window
    else:
        raise NotImplementedError(eval_class)

    pose_evaluator = PoseEvaluatorCls(cfg.evaluation,
                                   encoder,
                                   get_decoder(cfg.model.decoder),
                                   get_losses(cfg.loss))
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
    )

    metrics = trainer.test(pose_evaluator, datamodule=data_module)

    cfg.evaluation.output_metrics_path.parent.mkdir(exist_ok=True, parents=True)
    with cfg.evaluation.output_metrics_path.open("w") as f:
        json.dump(metrics[0], f)


if __name__ == "__main__":
    evaluate()
