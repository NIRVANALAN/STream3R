import json
import roma
import itertools
import imageio
import point_cloud_utils as pcu
import os
import sys
from typing import Any
from collections import defaultdict
import tqdm
import math
from pytorch_lightning.utilities.types import STEP_OUTPUT

import trimesh
from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..misc.cam_utils import camera_normalization, pose_auc, update_pose, get_pnp_pose

import csv
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningModule
from tabulate import tabulate

from ..loss.loss_ssim import ssim
from ..misc.image_io import load_image, save_image
from ..misc.utils import inverse_normalize, get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from .evaluation_cfg import EvaluationCfg
from .metrics import compute_lpips, compute_psnr, compute_ssim, compute_pose_error
from ..model.types import Gaussians
from ..misc.utils import inverse_normalize, vis_depth_map, confidence_map, get_overlap_tag

from .pose_evaluator_window import PoseEvaluatorWindow

# load dust3r global alignment required code
from src.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from src.dust3r.utils.device import to_numpy, to_cpu, collate_with_cat, to_cuda, to_device
from src.dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from src.dust3r.utils.misc import invalid_to_nans
from src.dust3r.utils.geometry import depthmap_to_pts3d, geotrf
from src.dust3r.demo_helper_fn import get_3D_model_from_scene
from src.dust3r.utils.image import load_images, rgb
import matplotlib.pyplot as pl

import copy

class GlobalAlignmentWindow(PoseEvaluatorWindow):

    def __init__(self, cfg: EvaluationCfg, encoder, decoder, losses) -> None:
        super().__init__(cfg, encoder, decoder, losses)

    @torch.inference_mode()
    def inference(self, batch) -> list:
        # ! TODO, collate the list of batch before doing the inference?
        # batch = {}
        visualization_dump = {}
        gaussians = self.encoder( # means: 1, HxWxV, 3
            batch["context"],
            self.global_step,
            visualization_dump=visualization_dump, # returns in: 1, 9, 256, 256, 1, 3
        )
        return gaussians, visualization_dump
    
    def get_reconstructed_scene(self, batch: BatchedExample, schedule='linear', niter=300, min_conf_thr=3.0,
                                mask_sky=False, clean_depth=False, transparent_cams=False, cam_size=0.05,
                                scenegraph_type='win', winsize=9) -> list:

        # mimic the get_reconstructed_scene() in dust3r.

        # ! first, load images and do inference
        # imgs = load_images(filelist, size=image_size, verbose=not silent)
        # if len(imgs) == 1:
        #     imgs = [imgs[0], copy.deepcopy(imgs[0])]
        #     imgs[1]['idx'] = 1
        # if scenegraph_type == "swin":

        scenegraph_type = scenegraph_type + "-" + str(winsize)

        # output = inference(pairs, model, device, batch_size=1, verbose=not silent)
        result = []
        for i in tqdm.trange(0, len(batch)):
            res = self.inference(batch[i])
            result.append(res)
        
        # ! cvt the result to output
        output = result
        raise NotImplementedError(
            "Windowed global aligner output conversion is not implemented for GlobalAlignmentWindow."
        )

        # pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        mode = GlobalAlignerMode.GSOptimizer # conventional point optimization here
        # mode = GlobalAlignerMode.GSWindowOptimizer # conventional point optimization here
        scene = global_aligner(output, device=self.device, mode=mode)
        lr = 0.01

        loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)


        pass


    def _merge_two_gaussians(self, gaussian_prev, gaussian_cur, merge_shared_gs=False):
        '''all input Gaussians are in the pixel-aligned format (for better merge after Procrustes alignment.)
        return: the merged Gaussians
        '''

        # first, convert gaussian_cur to pixel-aligned format

        if merge_shared_gs: # remove the "aligned src" pifu gaussians

            gaussian_cur = Gaussians(
                torch.cat([gaussian_cur.means[0:1], gaussian_cur.means[2:]], 0), # 8 N C
                torch.cat([gaussian_cur.covariances[0:1], gaussian_cur.covariances[2:]], 0),
                torch.cat([gaussian_cur.harmonics[0:1], gaussian_cur.harmonics[2:]], 0),
                torch.cat([gaussian_cur.opacities[0:1], gaussian_cur.opacities[2:]], 0),
            )

        merged_gaussian = Gaussians(
            torch.cat([gaussian_prev.means, gaussian_cur.means], 0),
            torch.cat([gaussian_prev.covariances, gaussian_cur.covariances], 0),
            torch.cat([gaussian_prev.harmonics, gaussian_cur.harmonics], 0),
            torch.cat([gaussian_prev.opacities, gaussian_cur.opacities], 0),
        )

        # merge pixel-aligned Gaussians into a single chunk
        # st()
        merged_gaussian = Gaussians(
            rearrange(
                merged_gaussian.means.to(self.device),
                "v ... xyz -> 1 (v ...) xyz",
            ),
            rearrange(
                merged_gaussian.covariances.to(self.device),
                "v ... i j-> 1 (v ...) i j",
            ),
            rearrange(
                merged_gaussian.harmonics.to(self.device),
                "v ... c d_sh -> 1 (v ...) c d_sh",
            ),
            rearrange(
                merged_gaussian.opacities.to(self.device),
                "v ... -> 1 (v ...)",
            ),
        )

        return merged_gaussian


    def _to_pixelaligned_gs(self, gaussians, V):
        srf, spp = 1, 1
        pixel_aligned_gaussians = Gaussians(
            rearrange(
                gaussians.means.float().cpu(),
                "b (v r srf spp) xyz -> (b v) (r srf spp) xyz", v=V, srf=srf, spp=spp
            ),
            rearrange(
                gaussians.covariances.float().cpu(),
                "b (v r srf spp) i j -> (b v) (r srf spp) i j", v=V, srf=srf, spp=spp, 
            ),
            rearrange(
                gaussians.harmonics.float().cpu(),
                "b (v r srf spp) c d_sh -> (b v) (r srf spp) c d_sh", v=V, srf=srf, spp=spp
            ),
            rearrange(
                gaussians.opacities.float().cpu(),
                "b (v r srf spp) -> (b v) (r srf spp)", v=V, srf=srf, spp=spp
            ),
        )
        return pixel_aligned_gaussians

    # def _to_single_gs(self, gaussians, V):
    #     srf, spp = 1, 1
    #     pixel_aligned_gaussians = Gaussians(
    #         rearrange(
    #             gaussians.means.float().cpu(),
    #             "b (v r srf spp) xyz -> (b v) (r srf spp) xyz", v=V, srf=srf, spp=spp
    #         ),
    #         rearrange(
    #             gaussians.covariances.float().cpu(),
    #             "b (v r srf spp) i j -> (b v) (r srf spp) i j", v=V, srf=srf, spp=spp, 
    #         ),
    #         rearrange(
    #             gaussians.harmonics.float().cpu(),
    #             "b (v r srf spp) c d_sh -> (b v) (r srf spp) c d_sh", v=V, srf=srf, spp=spp
    #         ),
    #         rearrange(
    #             gaussians.opacities.float().cpu(),
    #             "b (v r srf spp) -> (b v) (r srf spp)", v=V, srf=srf, spp=spp
    #         ),
    #     )
    #     return pixel_aligned_gaussians

    def test_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")

        # ! set model to eval
        self.encoder.eval()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # batch: BatchedExample = self.data_shim(batch) #  batch['context']['image'].shape torch.Size([1, 9, 3, 256, 256])
        global_window_batch = None
        if batch [-1]['window_idx'] == -1:
            global_window_batch, batch = batch[-1], batch[:-1]
        # assert global_window_batch is not None, 'hard coded'

        batch_of_windows = [self.data_shim(item) for item in batch] #  batch is a list of BatchedExample here

        all_reconstruction = []
        # st()
        # (Pdb) p batch[0]['context']['index']
        # tensor([[ 9,  0,  2,  4,  6, 11, 13, 15, 18]], device='cuda:0')
        # (Pdb) p batch[1]['context']['index']
        # tensor([[18,  9, 11, 13, 15, 20, 22, 24, 27]], device='cuda:0')

        view1, view2, pred1, pred2 = [defaultdict(list) for _ in range(4)] # match the format of cloud_opt in dust3r
        # view1 = {
        #     'true_shape': [],
        #     'img': [],
        #     'idx': [],
        #     'instance': [],
        # }
        # pred1 = {
        #    'pts3d': [], 
        #    'conf': [], 
        # }

        # * remap index (dust3r global alignment only accepts 0-N unbroken sequences as the index)
        all_index = []
        for batch in batch_of_windows:
            all_index.extend(batch['context']['index'][0].cpu().numpy().tolist())

        all_index_set = sorted(set(all_index))
        index_mapping = {
            frame_idx: order_idx for order_idx, frame_idx in enumerate(all_index_set)
        }
        index_mapping_inverse = {
            v: k for k, v in index_mapping.items()
        }

        # render_flag = False
        render_flag = True
        # joint_render_flag = False
        joint_render_flag = True
        procrustes_alignment = False  # use scaled version of rigid transformation to align splats across windows
        # procrustes_alignment = True  # use scaled version of rigid transformation to align splats across windows
        roma_procruses_result = [] # for sigma, R, t record


        # ! window-based reconstruction using our 9-view input model
        for window_id, batch in enumerate(batch_of_windows):

            b, v, _, h, w = batch["context"]["image"].shape
            assert b == 1

            # get overlap.
            overlap = batch["context"]["overlap"][0, 0]
            overlap_tag = get_overlap_tag(overlap)
            if overlap_tag == "ignore":
                return

            # runing encoder to obtain the 3DGS
            # input_images_view2 = batch["context"]["image"][:, 1:].clone()
            input_images_view2=rearrange(batch["context"]["image"][:, 1:].clone(),
                "b v ... -> (b v) 1 ...",
            )

            input_images_view2 = input_images_view2 * 0.5 + 0.5 # as the gt
            visualization_dump = {}

            visualization_dump['ctx_imgs_collate'] = rearrange(batch["context"]["image"][:].clone(),
                "b v rgb h w -> b (v h w) rgb"
            ).detach().cpu().numpy()

            visualization_dump['ctx_imgs'] = rearrange(batch["context"]["image"][:].clone(),
                "b v rgb h w -> (b v) (h w) rgb" # b=1 here, so it's fine
            ).detach().cpu().numpy()

            visualization_dump['index'] = batch['context']['index'][0].cpu().numpy().tolist()
            visualization_dump['ga_index'] = [index_mapping[idx] for idx in visualization_dump['index']]

            gaussians = self.encoder( # means: 1, HxWxV, 3
                batch["context"],
                self.global_step,
                visualization_dump=visualization_dump, # returns in: 1, 9, 256, 256, 1, 3
            )
            V = batch["context"]["image"].shape[1]

            # ===== finalize init here =====

            # reshape gaussians, v to b

            pixel_aligned_gaussians = self._to_pixelaligned_gs(gaussians, V=V)

            if procrustes_alignment and window_id > 0:
                # conduct rigid alignment
                pcd_tgt = all_reconstruction[-1][1].means[-1] # HW 3, get the last frame, pred_window_gs_xyz
                pcd_src = pixel_aligned_gaussians.means[1] # acrually the first frame ([0] is the canonical frame) src_window_gs_xyz
                # st()
                R_pred, t_pred, scale_pred = roma.rigid_points_registration(pcd_src, pcd_tgt, compute_scaling=True)
                R_pred, t_pred, scale_pred = to_cuda((R_pred, t_pred, scale_pred)) # roma always return 'cpu' tensor
                roma_procruses_result.append((R_pred, t_pred, scale_pred))
            
                # ! "rectify" the current gs window prediction accordingly
                # aligned_gs_xyz = scale_pred * pcd_src @ R_pred.T + t_pred
                aligned_xyz = scale_pred * gaussians.means @ R_pred.T + t_pred # ! update Gaussian means
                # st()
                # update gaussian covariances also.
                aligned_covariances = (scale_pred**2) * (R_pred @ gaussians.covariances @ R_pred.T) # ! update Gaussian covariances

                aligned_gaussians = copy.deepcopy(gaussians)
                aligned_gaussians.means = aligned_xyz
                aligned_gaussians.covariances = aligned_covariances

                pixel_aligned_gaussians = self._to_pixelaligned_gs(aligned_gaussians, V=V)

                # update the pixel_aligned_gaussians also.

            window_rec_result = [gaussians, pixel_aligned_gaussians, visualization_dump]

            if render_flag: # render the video
                # if window_id == 0:
                if False: # observe fixRs2 performance
                    window_rec_result.append([None, None]) # placeholder
                else: 
                    if global_window_batch is not None: # ! if to render global trajectory
                    # if False:
                        # video_name = f'window-{window_id}-global_fix'
                        # video_name = f'window-{window_id}-global_alignment'
                        # video_name = f'window-{window_id}-global_alignment'
                        # video_name = f'window-{window_id}-global_alignment'
                        video_name = f'window-{window_id}-global_alignment_fixCovRs2'

                        if procrustes_alignment and window_id > 0:
                            if not joint_render_flag:
                                gaussians_to_render = aligned_gaussians
                            else:
                                # st()
                                # video_name = f'window-{window_id}-global_alignment_joint'
                                # video_name = f'window-{window_id}-global_alignment_joint_nomergeshared'
                                # video_name = f'window-{window_id}-global_alignment_joint_nomergeshared_dl3dvckpt'
                                video_name = f'window-{window_id}-global_alignment_joint_dl3dvckpt'
                                # video_name = f'window-{window_id}-global_alignment_joint_co3d-ckpt'
                                # video_name = f'window-{window_id}-global_alignment_joint_co3d-ckpt'
                                # gaussians_to_render = self._merge_two_gaussians(all_reconstruction[-1][1], pixel_aligned_gaussians, merge_shared_gs=True)
                                gaussians_to_render = self._merge_two_gaussians(all_reconstruction[-1][1], pixel_aligned_gaussians, merge_shared_gs=False)
                        else:
                            gaussians_to_render = gaussians

                        output = self.decoder.forward(
                            # gaussians,
                            gaussians_to_render,
                            global_window_batch["extrinsics"],
                            global_window_batch["intrinsics"][:, :],
                            global_window_batch["near"][:, :],
                            global_window_batch["far"][:, :],
                            (h, w),
                            # cam_rot_delta=cam_rot_delta,
                            # cam_trans_delta=cam_trans_delta,
                        )
                        gt = global_window_batch['image']
                    
                    else: # render within-batch views
                        video_name = f'window-{window_id}_inwindow'

                        # st()
                        output = self.decoder.forward( # just ordinary rendering
                            gaussians,
                            # gaussians_to_render,
                            batch["target"]["extrinsics"],
                            batch["target"]["intrinsics"][:, :],
                            batch["target"]["near"][:, :],
                            batch["target"]["far"][:, :],
                            (h, w),
                            # cam_rot_delta=cam_rot_delta,
                            # cam_trans_delta=cam_trans_delta,
                        )
                        gt = batch['target']['image']

                    # images = [
                    #     vcat(rgb, depth)
                    #     for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
                    # ]

                    # ! 
                    images = [
                        vcat(gt_rgb, rgb, depth)
                        for rgb, gt_rgb, depth in zip(output.color[0], gt[0], vis_depth_map(output.depth[0]))
                    ]

                    video = torch.stack(images)
                    video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().permute(0,2,3,1).numpy() # F H W 3

                    window_rec_result.append((video_name, video))

            all_reconstruction.append(window_rec_result)

            # * set view info
            view1['true_shape'].extend([torch.tensor(batch['context']['image'].shape[-2:]) for _ in range(v)])
            view2['true_shape'].extend([torch.tensor(batch['context']['image'].shape[-2:]) for _ in range(v)])

            # view1['img'].append(batch['context']['image'][0, 0:1].repeat_interleave(v-1, dim=0)) # v-1, h, w, 3. for concat later.
            # view2['img'].append(batch['context']['image'][0, 1:]) # v-1, hw3

            view1['img'].append(batch['context']['image'][0, 0:1].repeat_interleave(v, dim=0)) # v-1, h, w, 3. for concat later.
            view2['img'].append(batch['context']['image'][0, :]) # v-1, hw3

            # view1['idx'].extend([batch['context']['index'][0, 0].item() for _ in range(v-1)])
            # view2['idx'].extend(batch['context']['index'][0, 1:].cpu().numpy().tolist())
            view1['idx'].extend([batch['context']['index'][0, 0].item() for _ in range(v)])
            view2['idx'].extend(batch['context']['index'][0, :].cpu().numpy().tolist())

            view1['instance'].extend([str(item) for item in view1['idx']])
            view2['instance'].extend([str(item) for item in view2['idx']])

            # * update pred
            # pred1['pts3d'].append(pixel_aligned_gaussians.means[0:1].float().cpu().repeat_interleave(v-1, dim=0))
            # pred2['pts3d'].append(pixel_aligned_gaussians.means[1:].float().cpu())

            pred1['pts3d'].append(pixel_aligned_gaussians.means[0:1].float().cpu().repeat_interleave(v, dim=0))
            pred2['pts3d'].append(pixel_aligned_gaussians.means[:].float().cpu())

            # pred1['conf'].append(pixel_aligned_gaussians.opacities[0:1].float().cpu().repeat_interleave(v-1, dim=0))
            # pred2['conf'].append(pixel_aligned_gaussians.opacities[1:].float().cpu()) # directly use opacities as the conf.

            pred1['conf'].append(pixel_aligned_gaussians.opacities[0:1].float().cpu().repeat_interleave(v, dim=0))
            pred2['conf'].append(pixel_aligned_gaussians.opacities[:].float().cpu()) # directly use opacities as the conf.
        

        # * collate the output
        def collate_window_prediction(view1, view2, pred1, pred2):
            def collate_view(view):
                view['true_shape'] = torch.stack(view['true_shape'], 0) # other stuffs are already extended
                view['img'] = torch.cat(view['img'], 0) # other stuffs are already extended
                # do the re-mapping
                view['idx'] = [index_mapping[idx] for idx in view['idx']]
                return view

            def collate_pred(pred): # dict contains torch.Tensor
                pred['pts3d'] = rearrange(torch.cat(pred['pts3d'], dim=0), 'b (h w) ... -> b h w ...', h=h, w=w)
                pred['conf'] = rearrange(torch.cat(pred['conf'], dim=0), 'b (h w) ... -> b h w ...', h=h, w=w)
                return pred
            
            view1, view2 = map(collate_view, (view1, view2))
            pred1, pred2 = map(collate_pred, (pred1, pred2))

            return view1, view2, pred1, pred2
            
        view1, view2, pred1, pred2 = collate_window_prediction(view1, view2, pred1, pred2)
        pred2['pts3d_in_other_view'] = pred2.pop('pts3d')
        output = {'view1': view1, 'view2': view2, 'pred1': pred1, 'pred2': pred2}

        # output_dir = './outputs/ga_alignment_window/v0_debug'
        # output_dir = './outputs_cpfs/ga_alignment_window/v0_debug'
        # output_dir = './outputs_cpfs/ga_alignment_window/v1_procrustes'
        output_dir = './outputs_cpfs/ga_alignment_window/vis_per-window-result'
        # output_dir = './outputs_cpfs/ga_alignment_window_co3d/v1_procrustes'
        # output_dir = './outputs_cpfs/ga_alignment_window_co3d/v1_procrustes_debug'
        # output_dir = './outputs_cpfs/ga_alignment_window_co3d/v1_procrustes_debug'
        os.makedirs(output_dir, exist_ok=True)

        def vis_gs_pcd(rec_result, output_path, pcd_name_prefix, save_rgb=False, verb=False, save_different_view=False):
            # ! visualization of the predicted gaussians
            # gaussians, pixel_aligned_gaussians, visualization_dump = rec_result
            gaussians, pixel_aligned_gaussians, visualization_dump = rec_result[0], rec_result[1] ,rec_result[2]

            if save_different_view:
                for v in range(pixel_aligned_gaussians.means.shape[0]):
                    gs_xyz = pixel_aligned_gaussians.means[v]
                    gs_rgb = (visualization_dump['ctx_imgs'][v] * 127.5 + 127.5).astype(np.uint8) # V 1 3 H W
                    frameidx = visualization_dump['index'][v]
                    pcd = trimesh.Trimesh(vertices=gs_xyz, vertex_colors=gs_rgb)

                    pcd_output_path = f'{pcd_name_prefix}-v_{v}-frameidx_{frameidx}.obj'
                    pcd.export(os.path.join(output_path, pcd_output_path), 'obj')

                    if verb:
                        print('pcd exported to {}'.format(pcd_output_path))


            else: # save a single gaussian pcd
                gs_xyz = gaussians.means[0].detach().cpu().numpy() # N*3

                if save_rgb:
                    gs_rgb = (visualization_dump['ctx_imgs_collate'][0] * 127.5 + 127.5).astype(np.uint8) # V 1 3 H W
                    pcd = trimesh.Trimesh(vertices=gs_xyz, vertex_colors=gs_rgb)
                else:
                    pcd = trimesh.Trimesh(vertices=gs_xyz)

                pcd.export(os.path.join(output_path, f'{pcd_name_prefix}.obj'), 'obj')

                if verb:
                    print('pcd exported to {}'.format(output_path))
        
        def save_video(video, video_path, verb=False):

            rgb_video_out = imageio.get_writer(
                video_path,
                mode='I',
                fps=24,
                codec='libx264')
            
            for frame_idx in range(video.shape[0]):
                rgb_video_out.append_data(video[frame_idx])

            rgb_video_out.close() 
            if verb:
                print(f'video of {video.shape[0]} frames saved to {video_path}')

        for window_id, window_rec_result in enumerate(all_reconstruction):

            # if window_id == 0 and window_rec_result[0] is None:

            # if window_id == 0 and window_rec_result[-1][1] is None:
            #     continue

            # continue # no need to save
            if render_flag: # save video to local dir     
                video_name, video = window_rec_result[-1] 
                vid_output_dir = os.path.join(output_dir, f'{video_name}.mp4') # 
                save_video(video, vid_output_dir, verb=True)

            # save pcd
            pcd_name_prefix = f'window-{window_id}'
            vis_gs_pcd(window_rec_result, output_dir,pcd_name_prefix=pcd_name_prefix, save_rgb=True, verb=True, save_different_view=True)
            # vis_gs_pcd(window_rec_result, output_dir,pcd_name_prefix=pcd_name_prefix, save_rgb=True, verb=True)
    
        
        ''' vis xyz

        vis_gs_pcd(all_reconstruction[1], os.path.join(output_dir, 'window-1.obj'))
        '''

        # ! conduct rigid registration to connect different frames
        # https://naver.github.io/roma/#rigid-registration

        '''
        # ! setup global optimization
        with torch.set_grad_enabled(True):
            mode = GlobalAlignerMode.GSOptimizer # conventional point optimization here
            # mode = GlobalAlignerMode.GSWindowOptimizer # ! todo
            scene = global_aligner(output, device=self.device, mode=mode, min_conf_thr=0.0) # since we use opacity as the conf here
            lr = 0.01
            niter = 300 # default
            schedule = 'linear'
            # ! do global optimization
            loss = scene.compute_global_alignment(init='mst', niter=niter, schedule=schedule, lr=lr)

        # st() # get_masks return none
        
        # pts3d = to_numpy(scene.get_pts3d()) # follows the scene.str_edges order, same order as scene.imgs
        # visualize the pts3d

        # ! port the output
        # outdir = 'outputs_cpfs/exp_global_alignment'
        # outfile = get_3D_model_from_scene(outdir, silent=False, scene=scene)

        outfile = get_3D_model_from_scene(output_dir, silent=False, scene=scene, as_pointcloud=True)
        '''
        
        
        def lint_ga_output(scene):

            # also return rgb, depth and confidence imgs
            # depth is normalized with the max value for all images
            # we apply the jet colormap on the confidence maps
            rgbimg = scene.imgs
            depths = to_numpy(scene.get_depthmaps())
            confs = to_numpy([c for c in scene.im_conf])
            cmap = pl.get_cmap('jet')
            depths_max = max([d.max() for d in depths])
            depths = [d / depths_max for d in depths]
            confs_max = max([d.max() for d in confs])
            confs = [cmap(d / confs_max) for d in confs]

            imgs = []
            for i in range(len(rgbimg)):
                imgs.append(rgbimg[i])
                imgs.append(rgb(depths[i]))
                imgs.append(rgb(confs[i]))

            return imgs

        # imgs  = lint_ga_output(scene) # for visualization? later afterwards
        # st()

        # return scene, outfile, imgs

        # retrieve the init pose via PnPRansac
        '''
        pnp_pose_list = []

        for v in range(1, V):
            pose_opt = get_pnp_pose(visualization_dump['means'][0, v].squeeze(),
                                    visualization_dump['opacities'][0, v].squeeze(),
                                    batch["context"]["intrinsics"][0, v], h, w)
            pnp_pose_list.append(pose_opt)

        stacked_pnp_pose = torch.stack(pnp_pose_list).to(self.device) # V-1, 4, 4

        # pose_opt = pose_opt.to(self.device) # shape: 4x4
        # pose_opt = batch["context"]["extrinsics"][0, 0].clone()  # initial pose as the first view: I

        with torch.set_grad_enabled(True):
            # cam_rot_delta = nn.Parameter(torch.zeros([b, 1, 3], requires_grad=True, device=self.device))
            # cam_trans_delta = nn.Parameter(torch.zeros([b, 1, 3], requires_grad=True, device=self.device))
            cam_rot_delta = nn.Parameter(torch.zeros([V-1, 1, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([V-1, 1, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": 0.005,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": 0.005,
                }
            )

            pose_optimizer = torch.optim.Adam(opt_params)

            number_steps = 200
            # extrinsics = pose_opt.unsqueeze(0).unsqueeze(0)  # initial pose use pose_opt
            extrinsics = stacked_pnp_pose.unsqueeze(1) # V-1, 1, 4, 4
            for i in range(number_steps):
                pose_optimizer.zero_grad()

                output = self.decoder.forward(
                    pixel_aligned_gaussians,
                    extrinsics,
                    batch["context"]["intrinsics"][:, 1:],
                    batch["context"]["near"][:, 1:],
                    batch["context"]["far"][:, 1:],
                    (h, w),
                    cam_rot_delta=cam_rot_delta,
                    cam_trans_delta=cam_trans_delta,
                )

                # Compute and log loss.
                batch["target"]["image"] = input_images_view2
                total_loss = 0
                for loss_fn in self.losses:
                    loss = loss_fn.forward(output, batch, gaussians, self.global_step)
                    total_loss = total_loss + loss

                # add ssim structure loss
                ssim_, _, _, structure = ssim(rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
                                      rearrange(output.color, "b v c h w -> (b v) c h w"),
                                      size_average=True, data_range=1.0, retrun_seprate=True, win_size=11)
                ssim_loss = (1 - structure) * 1.0
                total_loss = total_loss + ssim_loss

                # backpropagate
                # print(f"Step {i} - Loss: {total_loss.item()}")
                total_loss.backward()
                with torch.no_grad():
                    pose_optimizer.step()
                    new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                )
                    cam_rot_delta.data.fill_(0)
                    cam_trans_delta.data.fill_(0)

                    # extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=1)
                    extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=V-1, v=1)
                    # st()
                    pass

            # eval pose
            # st()
            # gt_pose = batch["context"]["extrinsics"][0, 1]
            gt_pose = batch["context"]["extrinsics"][0, 1:] # V-1, 4, 4
            eval_pose = extrinsics[:, 0] # V-1, 4, 4
            for v in range(V-1):
                error_t, error_t_scale, error_R = compute_pose_error(gt_pose[v], eval_pose[v])
                error_pose = torch.max(error_t, error_R)  # find the max error

                all_metrics = {
                    "e_t_ours": error_t,
                    "e_R_ours": error_R,
                    "e_pose_ours": error_pose,
                }

                self.print_preview_metrics(all_metrics, overlap_tag) # auto running metrics avg
        '''

        return 0