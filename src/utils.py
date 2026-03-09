import torch
import torch.multiprocessing as mp
import os
import copy
from pathlib import Path

# avoid high cpu usage
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
torch.set_num_threads(1)

import hydra
import wandb
import signal
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf
from pdb import set_trace as st

from src.misc.weight_modify import checkpoint_filter_fn

from src.dust3r.utils.device import to_cpu, collate_with_cat


def load_encoder_ckpt(cfg, encoder):
    # Load the encoder weights.
    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
        weight_path = cfg.model.encoder.pretrained_weights
        ckpt_weights = torch.load(weight_path, map_location='cpu', weights_only=False)
        print('loading from: ', cfg.model.encoder.pretrained_weights)
        # st()
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
                            encoder_state_dict[k][:v.
                                                  shape[0]] = v  # e.g., conf
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

        elif 'state_dict' in ckpt_weights: # if loading from pl ckpt
            missing_keys, unexpected_keys = [], []

            ckpt_weights = ckpt_weights['state_dict']
            ckpt_weights = {
                k[8:]: v
                for k, v in ckpt_weights.items() if k.startswith('encoder.')
            }
            # ! load dpt_self weights

            # ! load the dpt weights from cross to self as initialization
            # st()
            if 'downstream_head_self.dpt.act_postprocess.0.1.weight' in encoder.state_dict(
            ) and 'downstream_head_self.dpt.act_postprocess.0.1.weight' not in ckpt_weights:
                new_ckpt_weights = copy.deepcopy(ckpt_weights)
                for k, v in ckpt_weights.items():
                    if k.startswith('downstream_head1'):
                        new_ckpt_weights[k.replace(
                            'downstream_head1',
                            'downstream_head_self')] = v.clone()
                print('initialize downstream_head_self with downstream_head1')
            
            elif 'backbone.dec_state_blocks.0.cross_attn.projv.weight' in encoder.state_dict(
            ) and 'backbone.dec_state_blocks.0.cross_attn.projv.weight' not in ckpt_weights:
                # st()
                new_ckpt_weights = copy.deepcopy(ckpt_weights) # dec_state_blocks
                for k, v in ckpt_weights.items():
                    if k.startswith('backbone.dec_blocks'):
                        new_key = k.replace( 'dec_blocks', 'dec_state_blocks')
                        new_ckpt_weights[new_key] = v.clone()
                        # print('init ', new_key)
                print('initialize dec_state_blocks with dec_blocks')

            else:
                new_ckpt_weights = ckpt_weights

        # self.downstream_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))

        # # magic wrapper
        # self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)

        # if self.cfg.backbone.asymmetry_decoder: # type: ignore
        #     self.downstream_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
        #     self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
        # else:
        #     self.downstream_head2 = self.downstream_head1
        #     self.head2 = self.head1

        # if self.cfg.has_self_pts_head:
        #     assert not self.cfg.backbone.asymmetry_decoder
        #     self.downstream_head_self = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))

            encoder_state_dict = encoder.state_dict()
            for k, v in new_ckpt_weights.items():
                if k in encoder_state_dict:
                    if v.shape == encoder_state_dict[k].shape:
                        encoder_state_dict[k] = v
                else:
                    if k not in encoder_state_dict:
                        print(k, 'missing in ckpt')
                    else:
                        print(k, v.shape, encoder_state_dict[k].shape, 'mismatch')
            
            # check missing stuffs
            for k in encoder_state_dict.keys():
                if k not in new_ckpt_weights:
                    missing_keys.append(k)

            # st()

            _, unexpected_keys = encoder.load_state_dict(
                encoder_state_dict, strict=True)
            # new_ckpt_weights, strict=False)

            if len(unexpected_keys):
                print('unexpected_keys: ', unexpected_keys)

            if len(missing_keys):
                print('missing_keys: ', missing_keys)
        else:
            raise ValueError(f"Invalid checkpoint format: {weight_path}")


# @torch.inference_mode()
def inference_pl(groups, model, device, verbose=True):
    model.eval()
    ignore_keys = set(
        [
            "depthmap", "dataset", "label", "instance", "idx", "true_shape",
            "rng"
        ]
        # ["depthmap", "dataset", "label", "instance", "true_shape", "rng"]
    )
    for view in groups:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            if isinstance(view[name], tuple) or isinstance(view[name], list):
                view[name] = [
                    x.to(device, non_blocking=True) for x in view[name]
                ]
            else:
                view[name] = view[name].to(device, non_blocking=True)

    if verbose:
        print(f">> Inference with model on {len(groups)} image/raymaps")

    # res, state_args = loss_of_one_batch(groups, model, None, None, inference=True)
    # res = encoder.run_step(groups, enable_loss=False)

    with torch.inference_mode():

        visualization_dump = {
            'ignore_gs': True,
        }  # ! since imposing depth / pts3d loss here
        batch, V = model._preprocess_views(groups)  # type: ignore
        _ = model.encoder(
            batch["context"],
            0,
            visualization_dump=visualization_dump,
        )
        outputs, all_gts = model._prep_pred_and_gt_mv(groups,
                                                      visualization_dump)

    # st()
    ress = dict(pred=outputs, views=groups)
    result = to_cpu(ress)
    return result
