import os
from pathlib import Path

import hydra
import torch
import wandb
import signal
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf

torch.backends.cuda.matmul.allow_tf32 = True

from src.misc.weight_modify import checkpoint_filter_fn
from src.model.distiller import get_distiller

from pdb import set_trace as st

# avoid high cpu usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices="auto", # train on all gpus
        # devices=1, # if do evaluation only
        strategy=( # https://lightning.ai/docs/pytorch/stable/extensions/strategy.html
            "ddp_find_unused_parameters_true"
            # "ddp"
            # "ddp"
            if torch.cuda.device_count() > 1
            else "auto"
        ), # pretty slow actually, ddp_find_unused_parameters_true
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        # plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],  # Uncomment for SLURM auto resubmission.
        inference_mode=False if (cfg.mode == "test" and cfg.test.align_pose) else True,
        # precision="bf16-mixed" # https://lightning.ai/docs/pytorch/2.5.0/common/trainer.html
        precision=cfg.trainer.precision, # https://lightning.ai/docs/pytorch/2.5.0/common/trainer.html
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches, # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    distiller = None
    if cfg.train.distiller:
        distiller = get_distiller(cfg.train.distiller)
        distiller = distiller.eval()

    # Load the encoder weights.
    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
        weight_path = cfg.model.encoder.pretrained_weights
        ckpt_weights = torch.load(weight_path, map_location='cpu')
        print('loading from: ', cfg.model.encoder.pretrained_weights)
        if 'model' in ckpt_weights:
            ckpt_weights = ckpt_weights['model']
            ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)

            # missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
            # ! some k, v not matched (conf dim) for multiview model.
            encoder_state_dict = encoder.state_dict()
            missing_keys, unexpected_keys = [], []
            for k, v in ckpt_weights.items():
                if k in encoder_state_dict:
                    # try:
                    if encoder_state_dict[k].shape == v.shape:
                        encoder_state_dict[k] = v

                    else:
                        if 'dpt.head.4' in k:
                            encoder_state_dict[k][:v.shape[0]] = v # e.g., conf
                        else:
                            raise ValueError(f"unexpected keys: {k}")

                else:
                    unexpected_keys.append(k)
            
            for k in encoder_state_dict.keys():
                if k not in ckpt_weights:
                    missing_keys.append(k)

            # st()
            encoder.load_state_dict(encoder_state_dict, strict=True)
            if len(unexpected_keys):
                print('unexpected_keys: ', unexpected_keys)

            if len(missing_keys):
                print('missing_keys: ', missing_keys)

        elif 'state_dict' in ckpt_weights:
            ckpt_weights = ckpt_weights['state_dict']
            ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)

            if len(unexpected_keys):
                print('unexpected_keys: ', unexpected_keys)

            if len(missing_keys):
                print('missing_keys: ', missing_keys)
        else:
            raise ValueError(f"Invalid checkpoint format: {weight_path}")


    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder),
        get_losses(cfg.loss),
        step_tracker,
        distiller=distiller,
    )

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()
