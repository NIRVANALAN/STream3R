from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

from src.model.vggt.camera_head import CameraHead
from src.model.vggt.utils.pose_enc import pose_encoding_to_extri_intri

inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderNoPoSplatCfg:
    name: Literal["uni_gauss_multi"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    has_depth_head: bool = False
    has_self_pts_head: bool = False


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderNoPoSplatMulti(Encoder[EncoderNoPoSplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderNoPoSplatCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)
        self.cfg = cfg

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(
                cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        self.gs_params_head_type = cfg.gs_params_head_type

        self.set_mean_head(
            output_mode='pts3d',
            head_type='dpt',
            landscape_only=False,
            depth_mode=('exp', -inf, inf),
            conf_mode=('exp', 1, inf),
        )
        self.set_gs_params_head(cfg, cfg.gs_params_head_type)  # dpt_gs

        if cfg.has_depth_head:
            self.set_depth_head(
                output_mode='depth',
                head_type='dpt',
                landscape_only=False,
                depth_mode=('exp', -inf, inf),
                conf_mode=('exp', 1, inf),
            )
        else:
            self.depth_head = None

        # st()

        if self.backbone.enable_camera_token:
            self.camera_head = CameraHead(
                dim_in=1 * self.backbone.dec_embed_dim)  # type: ignore

    def set_mean_head(self, output_mode, head_type, landscape_only, depth_mode,
                      conf_mode):
        # self.output_mode = output_mode
        # self.head_type = head_type
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode  # ('exp', 1, inf)
        # allocate heads
        self.downstream_head1 = head_factory(head_type,
                                             output_mode,
                                             self.backbone,
                                             has_conf=bool(conf_mode))

        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1,
                                            activate=landscape_only)

        if self.cfg.backbone.asymmetry_decoder:  # type: ignore
            self.downstream_head2 = head_factory(head_type,
                                                 output_mode,
                                                 self.backbone,
                                                 has_conf=bool(conf_mode))
            self.head2 = transpose_to_landscape(self.downstream_head2,
                                                activate=landscape_only)
        else:
            self.downstream_head2 = self.downstream_head1
            self.head2 = self.head1

        if self.cfg.has_self_pts_head:
            assert not self.cfg.backbone.asymmetry_decoder
            self.downstream_head_self = head_factory(head_type,
                                                     output_mode,
                                                     self.backbone,
                                                     has_conf=bool(conf_mode))
            self.head_self = transpose_to_landscape(self.downstream_head_self,
                                                    activate=landscape_only)

    def set_depth_head(self, output_mode, head_type, landscape_only,
                       depth_mode, conf_mode):
        # https://github.com/facebookresearch/vggt/blob/6830466c427cd77c877ae2d4705c13fc684a890f/vggt/models/vggt.py#L24
        # self.output_mode = output_mode
        # self.head_type = head_type
        assert not self.cfg.backbone.asymmetry_decoder

        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode  # ('exp', 1, inf)
        assert bool(conf_mode)
        # allocate heads
        self.downstream_depth_head = head_factory(head_type,
                                                  output_mode,
                                                  self.backbone,
                                                  has_conf=True)

        self.head_depth = transpose_to_landscape(self.downstream_depth_head,
                                                 activate=landscape_only)

        # magic wrapper
        # self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)

    # def set_pose_pred_head(self):
    #     # ! use vggt/vggsfm-like design
    #     pass

    def set_gs_params_head(self, cfg, head_type):
        # st()
        if head_type == 'linear':
            self.gaussian_param_head = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    self.backbone.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size**2 * self.raw_gs_dim,
                ),
            )

            if self.cfg.backbone.asymmetry_decoder:  # type: ignore
                self.gaussian_param_head2 = deepcopy(self.gaussian_param_head)
            else:
                self.gaussian_param_head2 = self.gaussian_param_head

        elif head_type == 'dpt':
            self.gaussian_param_head = head_factory(
                head_type,
                'gs_params',
                self.backbone,
                has_conf=False,
                out_nchan=self.raw_gs_dim)  # for view1 3DGS
            if self.cfg.backbone.asymmetry_decoder:  # type: ignore
                self.gaussian_param_head2 = head_factory(
                    head_type,
                    'gs_params',
                    self.backbone,
                    has_conf=False,
                    out_nchan=self.raw_gs_dim)  # for view2 3DGS
            else:
                self.gaussian_param_head2 = self.gaussian_param_head

            # # magic wrapper
            # self.head3 = transpose_to_landscape(self.to_gaussians, activate=landscape_only)
            # self.head4 = transpose_to_landscape(self.to_gaussians2, activate=landscape_only)
        elif head_type == 'dpt_gs':
            self.gaussian_param_head = head_factory(head_type,
                                                    'gs_params',
                                                    self.backbone,
                                                    has_conf=False,
                                                    out_nchan=self.raw_gs_dim)
            if self.cfg.backbone.asymmetry_decoder:  # type: ignore
                self.gaussian_param_head2 = head_factory(
                    head_type,
                    'gs_params',
                    self.backbone,
                    has_conf=False,
                    out_nchan=self.raw_gs_dim)

            else:
                self.gaussian_param_head2 = self.gaussian_param_head

        else:
            raise NotImplementedError(f"unexpected {head_type=}")

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up,
                              1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf)**exponent + pdf**(1 / exponent))

    def _downstream_head(self,
                         head_num,
                         decout,
                         img_shape,
                         ray_embedding=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)

    def forward(
        self,
        context: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
    ):
        # ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        dec_feat, shape, images = self.backbone(
            context
        )  # len(dec_feat): 13, dec_feat[0].shape: torch.Size([1, 9, 256, 1024])

        # ! split camera tokens
        if self.backbone.enable_camera_token:
            token_length = dec_feat[0].shape[2] - 1
            # ! wrong order!! the camera token is appended, not prepended
            # split_list = [torch.split(feat, [1, token_length], dim=2) for feat in dec_feat]
            # pose_feat, dec_feat = [chunk[0] for chunk in split_list], [chunk[1] for chunk in split_list]
            # st()
            split_list = [
                torch.split(feat, [token_length, 1], dim=2)
                for feat in dec_feat
            ]
            dec_feat, pose_feat = map(list, zip(*split_list))
        else:
            pose_feat = None

        with torch.autocast(device_type="cuda", enabled=False):
            # for the pose prediction heads
            if self.backbone.enable_camera_token:
                assert pose_feat is not None
                all_pose_pred = self.camera_head(
                    pose_feat)  # list of [B, V, 9]
            else:
                all_pose_pred = None

        # ! test injecting K into dec_feat for scale-aware prediction
        # st()
        intrinsic_embedding_all = None
        if self.backbone.intrinsics_embed_loc == 'dpt' and "linear" in self.backbone.intrinsics_embed_type:
            if self.backbone.intrinsics_embed_type == 'linear':
                intrinsics = context["intrinsics"]
            elif self.backbone.intrinsics_embed_type == 'linear-from-pred':
                assert self.backbone.enable_camera_token
                assert all_pose_pred is not None
                H, W = context['image'].shape[-2:]

                pose, intrinsics = pose_encoding_to_extri_intri(
                    all_pose_pred[-1], (H, W), normalize_intrinsics=True)

            intrinsic_embedding = self.backbone.intrinsic_encoder(
                intrinsics.flatten(2))
            intrinsic_embedding_all = rearrange(intrinsic_embedding,
                                                "b v c -> b v 1 c")

            # naive "linear" add.
            for i in range(
                    1, len(dec_feat)
            ):  # add K information to all intermediate vit features
                dec_feat[i] = dec_feat[i] + intrinsic_embedding_all

        with torch.amp.autocast(device_type="cuda", enabled=False):
            all_depth_res = []
            all_self_res = []
            all_mean_res = []
            all_other_params = []
            res1 = self._downstream_head(
                1, [tok[:, 0].float() for tok in dec_feat],
                shape[:, 0])  # ! returned ['pts3d', 'conf']
            all_mean_res.append(res1)
            for i in range(1, v):
                res2 = self._downstream_head(
                    2, [tok[:, i].float() for tok in dec_feat], shape[:, i])
                all_mean_res.append(res2)

            # for the 3DGS heads
            if visualization_dump is None or not visualization_dump.get(
                    'ignore_gs', False):
                if self.gs_params_head_type == 'dpt_gs':  # default setting
                    GS_res1 = self.gaussian_param_head(
                        [tok[:, 0].float() for tok in dec_feat],
                        all_mean_res[0]['pts3d'].permute(0, 3, 1, 2),
                        images[:, 0, :3].float(), shape[0, 0].cpu().tolist())
                    GS_res1 = rearrange(GS_res1, "b d h w -> b (h w) d")
                    all_other_params.append(GS_res1)
                    for i in range(1, v):
                        GS_res2 = self.gaussian_param_head2(
                            [tok[:, i].float() for tok in dec_feat],
                            all_mean_res[i]['pts3d'].permute(0, 3, 1, 2),
                            images[:, i, :3].float(), shape[0,
                                                            i].cpu().tolist())
                        GS_res2 = rearrange(GS_res2, "b d h w -> b (h w) d")
                        all_other_params.append(GS_res2)
                else:
                    raise NotImplementedError(
                        f"unexpected {self.gs_params_head_type=}")

            # st()
            if self.cfg.has_depth_head:  # ! like in vggt
                for i in range(v):
                    # depth_res = self.depth_head([tok[:, i].float() for tok in dec_feat], all_mean_res[i]['pts3d'].permute(0, 3, 1, 2), images[:, i, :3].float(), shape[0, i].cpu().tolist())
                    depth_res = self._downstream_head(
                        '_depth', [tok[:, i].float() for tok in dec_feat],
                        shape[:, i])
                    all_depth_res.append(depth_res)

            if self.cfg.has_self_pts_head:  # ! like in cut3r
                for i in range(v):
                    # depth_res = self.depth_head([tok[:, i].float() for tok in dec_feat], all_mean_res[i]['pts3d'].permute(0, 3, 1, 2), images[:, i, :3].float(), shape[0, i].cpu().tolist())
                    self_res = self._downstream_head(
                        '_self', [tok[:, i].float() for tok in dec_feat],
                        shape[:, i])
                    all_self_res.append(self_res)

            # st()
            pts_all = [
                all_mean_res_i['pts3d'] for all_mean_res_i in all_mean_res
            ]
            pts_all = torch.stack(pts_all, dim=1)
            # pts_all = rearrange(pts_all, "b v h w xyz -> b v (h w) xyz")
            pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces

            # ! return conf if required (for distillation loss or pts3d training.)
            conf_all = [
                all_mean_res_i['conf'] for all_mean_res_i in all_mean_res
            ]
            conf_all = torch.stack(conf_all, dim=1)
            conf_all = rearrange(conf_all, "b v h w -> b v (h w)")
            # conf_all = conf_all.unsqueeze(-1)  # for cfg.num_surfaces

            if len(all_self_res) > 0:

                pts_self_all = [
                    all_mean_res_i['pts3d'] for all_mean_res_i in all_self_res
                ]
                pts_self_all = torch.stack(pts_self_all, dim=1)
                # pts_self_all = rearrange(pts_self_all, "b v h w xyz -> b v (h w) xyz")
                # pts_self_all = pts_self_all.unsqueeze(-2)  # for cfg.num_surfaces

                # ! return conf if required (for distillation loss or pts3d training.)
                conf_self_all = [
                    all_mean_res_i['conf'] for all_mean_res_i in all_self_res
                ]
                conf_self_all = torch.stack(conf_self_all, dim=1)
                # conf_self_all = rearrange(conf_self_all, "b v h w -> b v (h w)")
                # conf_all = conf_all.unsqueeze(-1)  # for cfg.num_surfaces
                depths = rearrange(pts_self_all[..., -1],
                                   'b v h w -> b v (h w) 1 1')

            else:

                pts_self_all, conf_self_all = None, None
                # ! only used in pixelSplat or splatter-img. self.pose_free == False.
                depths = pts_all[..., -1].unsqueeze(-1)

            # Convert the features and depths into Gaussians.
            if visualization_dump is None or (not visualization_dump.get(
                    'ignore_gs', False)):

                # ! only predict gaussians for the world pts
                gaussians = torch.stack(all_other_params, dim=1)
                gaussians = rearrange(gaussians,
                                      "... (srf c) -> ... srf c",
                                      srf=self.cfg.num_surfaces)
                densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

                if self.pose_free:
                    gaussians = self.gaussian_adapter.forward(
                        # pts_all.unsqueeze(-2),
                        rearrange(pts_all,
                                  "b v h w 1 xyz -> b v (h w) 1 1 xyz"),
                        depths,
                        self.map_pdf_to_opacity(densities, global_step),
                        rearrange(gaussians[..., 1:],
                                  "b v r srf c -> b v r srf () c"),
                    )
                else:
                    xy_ray, _ = sample_image_grid((h, w), device)
                    xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
                    xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)

                    gaussians = self.gaussian_adapter.forward(
                        rearrange(context["extrinsics"],
                                  "b v i j -> b v () () () i j"),
                        rearrange(context["intrinsics"],
                                  "b v i j -> b v () () () i j"),
                        rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                        depths,
                        self.map_pdf_to_opacity(densities, global_step),
                        rearrange(gaussians[..., 1:],
                                  "b v r srf c -> b v r srf () c"),
                        (h, w),
                    )
            else:
                gaussians = None

            # Dump visualizations if needed.
            if visualization_dump is not None:
                # if not self.pose_free:
                visualization_dump["depth"] = rearrange(
                    depths, "b v (h w) srf s -> b v h w srf s", h=h,
                    w=w)  # ! for visualization

                if gaussians is not None:
                    visualization_dump["scales"] = rearrange(
                        gaussians.scales,
                        "b v r srf spp xyz -> b (v r srf spp) xyz")
                    visualization_dump["rotations"] = rearrange(
                        gaussians.rotations,
                        "b v r srf spp xyzw -> b (v r srf spp) xyzw")
                    visualization_dump["means"] = rearrange(
                        gaussians.means,
                        "b v (h w) srf spp xyz -> b v h w (srf spp) xyz",
                        h=h,
                        w=w)
                    visualization_dump['opacities'] = rearrange(
                        gaussians.opacities,
                        "b v (h w) srf s -> b v h w srf s",
                        h=h,
                        w=w)
                else:
                    visualization_dump['means'] = pts_all  # b v h w 1 3

                visualization_dump['conf'] = rearrange(conf_all,
                                                       "b v (h w) -> b v h w",
                                                       h=h,
                                                       w=w)
                visualization_dump['camera_pose'] = all_pose_pred
                if self.cfg.has_depth_head:
                    visualization_dump['dpt_pred_depth'] = all_depth_res

                if self.cfg.has_self_pts_head:
                    visualization_dump[
                        'pts3d_in_self_view'] = pts_self_all  # b v h w xyz
                    visualization_dump['conf_self'] = conf_self_all  # b v h w

                    # ! check head layout issue
                    if visualization_dump['conf'].shape != conf_self_all.shape:
                        raise RuntimeError(
                            "conf and conf_self shape mismatch: "
                            f"{visualization_dump['conf'].shape} vs {conf_self_all.shape}"
                        )

            # st()
            if gaussians is None:
                return gaussians
            else:
                return Gaussians(
                    rearrange(
                        gaussians.means.float(),
                        "b v r srf spp xyz -> b (v r srf spp) xyz",
                    ),
                    rearrange(
                        gaussians.covariances.float(),
                        "b v r srf spp i j -> b (v r srf spp) i j",
                    ),
                    rearrange(
                        gaussians.harmonics.float(),
                        "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                    ),
                    rearrange(
                        gaussians.opacities.float(),
                        "b v r srf spp -> b (v r srf spp)",
                    ),
                )

    def get_data_shim(self) -> DataShim:

        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
