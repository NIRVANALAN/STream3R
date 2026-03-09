from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from torch import nn
from torch.utils.checkpoint import checkpoint

from .croco.blocks import DecoderBlock
from .croco.croco import CroCoNet
from .croco.misc import fill_default_args, freeze_all_params, transpose_to_landscape, is_symmetrized, interleave, \
    make_batch_symmetric
from .croco.patch_embed import get_patch_embed
from .croco.pos_embed import get_1d_rotary_pos_embed
from .backbone import Backbone
from ....geometry.camera_emb import get_intrinsic_embedding

from pdb import set_trace as st
from diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid

from .backbone_croco import BackboneCrocoCfg, croco_params
from .backbone_croco_multiview import AsymmetricCroCoMulti

inf = float('inf')

class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        )
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        ) # ! linear vector memory
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C]
        feat_v: [B, 1, C]
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)
        for blk in self.write_blocks:
            mem, _ = blk(mem, feat, None, None)
        return mem

    def inquire(self, query, mem):
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1)
        for blk in self.read_blocks:
            x, _ = blk(x, mem, None, None)
        return x[..., -self.v_dim :]



class AsymmetricCroCoMulti_RW_CUT3R(AsymmetricCroCoMulti):
    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        super().__init__(cfg, d_in)

        # self.gradient_checkpointing = True # ! only for CA decoder here
        self.gradient_checkpointing = False
        self.pose_head_flag = False

        # TODO initialize memory module here
        # TODO, set the self.dec_norm *. use another function to set them all.

        # ! self._set_state_decoder. for decoding state.
        self.dec_embed_dim_state = self.dec_embed_dim
        self.decoder_embed_state = nn.Linear(self.enc_embed_dim, self.dec_embed_dim, bias=True)
        self.dec_norm_state = nn.LayerNorm(self.dec_embed_dim)

        # ! hyper-params for rnn state
        self.state_size = 256
        self.register_tokens = nn.Embedding(self.state_size, self.enc_embed_dim)
        self.state_pe = "2d"

        # self.set_freeze(config.freeze) # TODO
    
    # def _rnn_decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose):
    def _rnn_decoder(self, f_state, pos_state, f_img, pos_img, *args, **kwargs):

        final_output = [(f_state, f_img)]  # ! already after projection
        assert f_state.shape[-1] == self.dec_embed_dim
        # f_img = self.decoder_embed(f_img) # ! already embedded

        # if self.pose_head_flag:
        #     assert f_pose is not None and pos_pose is not None
        #     f_img = torch.cat([f_pose, f_img], dim=1)
        #     pos_img = torch.cat([pos_pose, pos_img], dim=1)

        # final_output.append((f_state, f_img))
        # for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks): # ! 12 blocks of decoder.
        for blk_state, blk_img in zip(self.dec_blocks, self.dec_blocks2): # ! 12 blocks of decoder.
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                f_state, _ = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    use_reentrant=True,
                )# type: ignore
                f_img, _ = checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    use_reentrant=True,
                ) # type: ignore
            else:
                f_state, _ = blk_state(*final_output[-1][::+1], pos_state, pos_img) # pos_* are the corresponding coord_ij for RoPE. only depends on the input resolution.
                f_img, _ = blk_img(*final_output[-1][::-1], pos_img, pos_state)
            final_output.append((f_state, f_img))

        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output)


    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat=None,
        pose_pos=None,
        # init_state_feat,
        # img_mask=None,
        # reset_mask=None,
        # update=None,
    ):
        # new_state_feat, dec = self._decoder(
        new_state_feat, dec = self._rnn_decoder(
            # state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos
            state_feat, state_pos, current_feat, current_pos,
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    def _encode_state(self, image_tokens, image_pos):
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        ) # just retrieve nn.Embeddings
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d": # ! default setting here.
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None


    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos) # ! directly retrieve
        state_feat = self.decoder_embed_state(state_feat) # linear
        return state_feat, state_pos


    def _forward_decoder_step(
        self,
        # views, 
        feat,
        pos,
        ret_state=False
    ):
        state_feat, state_pos = self._init_state(feat, pos) # retrieve + nn.Linear retrieval

        ress = []

        # ! for loop here
        for i in range(feat.shape[1]): # ! views available in each batch
            feat_i = feat[:, i] # ! B L C
            pos_i = pos[:, i]

            # pose_feat_i = None
            # pose_pos_i = None

            new_state_feat, dec = self._recurrent_rollout(
                state_feat.contiguous(),
                state_pos.contiguous(), # type: ignore
                feat_i.contiguous(),
                pos_i.contiguous(),
                None, # disabled for now
                None,
            )

            if ret_state:
                ress.append((new_state_feat, dec))
            else:
                ress.append(dec) 

            update_mask = 1 # ! hard coded here
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            # st()
            pass

        return ress



    # ! follow slam3r R & W TX design.
    def _decoder(self, feat, pose, extra_embed=None):
        b, v, l, c = feat.shape
        final_output = [feat]  # before projection
        if extra_embed is not None:
            feat = torch.cat((feat, extra_embed), dim=-1)

        # project to decoder dim
        f = rearrange(feat, "b v l c -> (b v) l c")
        f = self.decoder_embed(f)
        f = rearrange(f, "(b v) l c -> b v l c", b=b, v=v)

        # ! no PE here
        # assert not self.enable_temporal_pe, 'shall disable self.enable_temporal_pe'
        # if self.enable_temporal_pe:
        #     f = f + self.pos_embed_temporal.to(f).unsqueeze(2)

        # final_output.append(f) # ! not required, duplicated

        # ! RNN rollout
        # ! len(rnn_rollout_dec): v
        # ! len(rnn_rollout_dec[0]): 13 (length of decoder + 1)
        rnn_rollout_dec = self._forward_decoder_step(f, pose, ret_state=False) #  len(rnn_rollout_dec)==V, len(rnn_rollout_dec[0])==13
        rnn_rollout_dec = [torch.stack([dec_tuple[i] for dec_tuple in rnn_rollout_dec], dim=1) for i in range(len(rnn_rollout_dec[0]))] # stack tuple of last layer output to B V L C
        final_output += rnn_rollout_dec

        del final_output[1]  # duplicate with final_output[0]. just linear projection.

        # ! already called self.dec_norm, self.dec_state_norm in the self._forward_decoder_step
        # st_feat = rearrange(final_output[-1], "b v l c -> (b v) l c")
        # last_feat = self.dec_norm(last_feat)
        # final_output[-1] = rearrange(last_feat, "(b v) l c -> b v l c", b=b, v=v)
        
        return final_output


    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # ! use the pre-defined states
        # self.dec_blocks_state = nn.ModuleList(
        #     [
        #         DecoderBlock(
        #             dec_embed_dim,
        #             dec_num_heads,
        #             mlp_ratio=mlp_ratio,
        #             qkv_bias=True,
        #             norm_layer=norm_layer,
        #             norm_mem=norm_im2_in_dec,
        #             rope=self.rope,
        #         )
        #         for i in range(dec_depth)
        #     ]
        # )
        self.dec_norm_state = norm_layer(dec_embed_dim)


# ! gradually move from RNN to Decoder-only TX

class AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn(AsymmetricCroCoMulti_RW_CUT3R):
    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        super().__init__(cfg, d_in)

    def _forward_decoder_step(
        self,
        # views, 
        feat,
        pos,
        ret_state=False
    ):
        '''basically, the state_feat (global memory) stays the same, while the second CA sees all the previous hidden states.
        '''
        state_feat, state_pos = self._init_state(feat, pos) # retrieve + nn.Linear retrieval

        ress = []

        # ! for loop here
        for i in range(1, 1+feat.shape[1]): # ! views available in each batch
            # st()
            # feat_i = rearrange(feat[:, :i], 'b v l c -> b (v l) c') # ! B L C
            # pos_i = rearrange(pos[:, :i], 'b v l c -> b (v l) c')

            new_state_feat, dec = self._recurrent_rollout(
                state_feat.contiguous(),
                state_pos.contiguous(), # type: ignore
                feat[:, :i].contiguous(),
                pos[:, :i].contiguous(),
                None, # disabled for now
                None,
            )

            if ret_state:
                ress.append((new_state_feat, dec))
            else:
                ress.append(dec) 

            update_mask = 1 # ! hard coded here
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            # st()
            pass

        return ress

    def _rnn_decoder(self, f_state, pos_state, f_img, pos_img, *args, **kwargs):

        # v = f_img.shape[1]

        f_img_history, pos_img_history = None, None
        has_causal_mem = False

        if f_img.shape[1] > 1:
            f_img, f_img_history = f_img[:, -1].contiguous(), rearrange(f_img[:, :-1], 'b v l c -> b (v l) c').contiguous()
            pos_img, pos_img_history = pos_img[:, -1].contiguous(), rearrange(pos_img[:, :-1], 'b v l c -> b (v l) c').contiguous()
            has_causal_mem = True
        else:
            f_img, pos_img = f_img[:, 0].contiguous(), pos_img[:, 0].contiguous() # RoPE only accepts 3-dim input


        final_output = [(f_state, f_img)]  # ! already after projection
        assert f_state.shape[-1] == self.dec_embed_dim


        for idx, (blk_state, blk_img) in enumerate(zip(self.dec_blocks, self.dec_blocks2)): # ! 12 blocks of decoder.
            # if idx == 0:
            #     memory_to_read_from = final_output
            #     memory_pos = pos_state
            # else:

            if has_causal_mem: # ! append the history also.
                mem_to_read_from = torch.cat([final_output[-1][0], f_img_history], 1).contiguous()
                pos_mem = torch.cat([pos_state, pos_img_history], 1).contiguous()
                pass
            else:
                mem_to_read_from = f_state.contiguous()
                pos_mem = pos_state.contiguous()

            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                f_state, _ = checkpoint(
                    blk_state,
                    *final_output[-1][::+1], # f_state query f_img here
                    pos_state,
                    pos_img,
                    use_reentrant=True,
                )# type: ignore
                f_img, _ = checkpoint(
                    blk_img,
                    # *final_output[-1][::-1],
                    # pos_img,
                    # pos_state,
                    f_img,
                    mem_to_read_from,
                    pos_img,
                    pos_mem,
                    use_reentrant=True,
                ) # type: ignore
            else:
                # st()
                f_state, _ = blk_state(*final_output[-1][::+1], pos_state, pos_img) # pos_* are the corresponding coord_ij for RoPE. only depends on the input resolution.
                # f_img, _ = blk_img(*final_output[-1][::-1], pos_img, pos_state)
                f_img, _ = blk_img(f_img, mem_to_read_from, pos_img, pos_mem)
            final_output.append((f_state, f_img))

        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output)



class AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn_frame0_as_memory(AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn):
    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        super().__init__(cfg, d_in)

        # ! delete the state-relevant modules
        # del self.dec_norm_state
        del self.register_tokens
        del self.decoder_embed_state 

    def _forward_decoder_step(
        self,
        # views, 
        feat,
        pos,
        ret_state=False
    ):
        '''basically, the state_feat (global memory) stays the same, while the second CA sees all the previous hidden states.
        '''
        # state_feat, state_pos = self._init_state(feat, pos) # retrieve + nn.Linear retrieval
        state_feat, state_pos = feat[:, 0,].clone(), pos[:, 0,].clone() # remove intrinsics embedding here
        # state_feat, state_pos = feat[:, 0, :-1].clone(), pos[:, 0, :-1].clone() # remove intrinsics embedding here

        ress = []

        # ! for loop here
        for i in range(1, 1+feat.shape[1]): # ! views available in each batch
            # st()
            # feat_i = rearrange(feat[:, :i], 'b v l c -> b (v l) c') # ! B L C
            # pos_i = rearrange(pos[:, :i], 'b v l c -> b (v l) c')

            new_state_feat, dec = self._recurrent_rollout(
            # _, dec = self._recurrent_rollout( # ! just send in a placeholder
                state_feat.contiguous(),
                state_pos.contiguous(), # type: ignore
                feat[:, :i].contiguous(),
                pos[:, :i].contiguous(),
                None, # disabled for now
                None,
            )

            # if ret_state:
            #     ress.append((new_state_feat, dec))
            # else:
            ress.append(dec) 

            # update_mask = 1 # ! hard coded here

            # state_feat = new_state_feat * update_mask + state_feat * (
            #     1 - update_mask
            # )  # update global state

            # st()
            pass

        return ress

class AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn_frame0_as_memory_changeAttn(AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn_frame0_as_memory):
    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        super().__init__(cfg, d_in)

    def _forward_decoder_step(
        self,
        # views, 
        feat,
        pos,
        ret_state=False
    ):
        '''basically, the state_feat (global memory) stays the same, while the second CA sees all the previous hidden states.
        '''
        # state_feat, state_pos = feat[:, 0,].clone(), pos[:, 0,].clone() # remove intrinsics embedding here
        state_feat, state_pos = feat[:, 0,], pos[:, 0,] # ! 

        ress = []

        assert feat.shape[1] >= 2 # this model works on at least 2 views
        f0_pred = None

        # ! for loop here
        for i in range(2, 1+feat.shape[1]): # ! at least include one view
            # st()
            # feat_i = rearrange(feat[:, :i], 'b v l c -> b (v l) c') # ! B L C
            # pos_i = rearrange(pos[:, :i], 'b v l c -> b (v l) c')

            new_state_feat, dec = self._recurrent_rollout(
            # _, dec = self._recurrent_rollout( # ! just send in a placeholder
                state_feat.contiguous(),
                state_pos.contiguous(), # type: ignore
                feat[:, 1:i].contiguous(), # ! avoid loading feat[:, 0] since already canonical frame
                pos[:, 1:i].contiguous(),
                None, # disabled for now
                None,
            )
            if i==2: # return the first iter output
                # st()
                f0_pred = new_state_feat
                ress.append(f0_pred) # for frame-0 prediction

            ress.append(dec) 

            # update_mask = 1 # ! hard coded here

            # state_feat = new_state_feat * update_mask + state_feat * (
            #     1 - update_mask
            # )  # update global state

            # st()
            pass

        return ress


    def _rnn_decoder(self, f_state, pos_state, f_img, pos_img, *args, **kwargs):
        # ! f_state: f0
        # ! f_img: f_1..f_i

        # v = f_img.shape[1]

        f_img_history, pos_img_history = None, None
        has_causal_mem = False

        # ! f_img_history shall include f_i also, compat for i=1 case.

        f_img, f_img_history = f_img[:, -1].contiguous(), rearrange(f_img[:, :], 'b v l c -> b (v l) c').contiguous()
        pos_img, pos_img_history = pos_img[:, -1].contiguous(), rearrange(pos_img[:, :], 'b v l c -> b (v l) c').contiguous()

        # ! f_img and f_state will be updated.

        final_output = [(f_state, f_img)]  
        assert f_state.shape[-1] == self.dec_embed_dim # ! already after projection


        for idx, (blk_state, blk_img) in enumerate(zip(self.dec_blocks, self.dec_blocks2)): # ! 12 blocks of decoder.
            # if idx == 0:
            #     memory_to_read_from = final_output
            #     memory_pos = pos_state
            # else:

            # if has_causal_mem: # ! append the history also.
            # ! wrong here, to be fixed
            # mem_to_read_from = torch.cat([final_output[-1][0], f_img_history], 1).contiguous()
            # pos_mem = torch.cat([pos_state, pos_img_history], 1).contiguous()
            # pos_mem = pos_img_history

            # else:
            #     mem_to_read_from = f_state.contiguous() # 
            #     pos_mem = pos_state.contiguous()

            # if (
            #     self.gradient_checkpointing
            #     and self.training
            #     and torch.is_grad_enabled()
            # ):
            #     f_state, _ = checkpoint(
            #         blk_state,
            #         *final_output[-1][::+1], # f_state query f_img here
            #         pos_state,
            #         pos_img,
            #         use_reentrant=True,
            #     )# type: ignore
            #     f_img, _ = checkpoint(
            #         blk_img,
            #         # *final_output[-1][::-1],
            #         # pos_img,
            #         # pos_state,
            #         f_img,
            #         mem_to_read_from,
            #         pos_img,
            #         pos_mem,
            #         use_reentrant=True,
            #     ) # type: ignore
            # else:

            # ! pos_* are the corresponding coord_ij for RoPE. only depends on the input resolution.
            # ! f_state reads from the whole history; then write back to f_img 
            f_state, _ = blk_state(final_output[-1][0], f_img_history, pos_state, pos_img_history)
            f_img, _ = blk_img(final_output[-1][1], final_output[-1][0], pos_img, pos_state)

            final_output.append((f_state, f_img))

        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output)


    # ! follow slam3r R & W TX design.
    def _decoder(self, feat, pose, extra_embed=None):
        b, v, l, c = feat.shape
        final_output = [feat]  # before projection
        if extra_embed is not None:
            feat = torch.cat((feat, extra_embed), dim=-1)

        # project to decoder dim
        f = rearrange(feat, "b v l c -> (b v) l c")
        f = self.decoder_embed(f)
        f = rearrange(f, "(b v) l c -> b v l c", b=b, v=v)

        # ! no PE here
        # assert not self.enable_temporal_pe, 'shall disable self.enable_temporal_pe'
        # if self.enable_temporal_pe:
        #     f = f + self.pos_embed_temporal.to(f).unsqueeze(2)

        # final_output.append(f) # ! not required, duplicated

        # ! RNN rollout
        # ! len(rnn_rollout_dec): v
        # ! len(rnn_rollout_dec[0]): 13 (length of decoder + 1)
        rnn_rollout_dec = self._forward_decoder_step(f, pose, ret_state=False) #  len(rnn_rollout_dec)==V, len(rnn_rollout_dec[0])==13
        rnn_rollout_dec = [torch.stack([dec_tuple[i] for dec_tuple in rnn_rollout_dec], dim=1) for i in range(len(rnn_rollout_dec[0]))] # stack tuple of last layer output to B V L C
        # st()
        final_output += rnn_rollout_dec

        del final_output[1]  # duplicate with final_output[0]. just linear projection.

        # ! already called self.dec_norm, self.dec_state_norm in the self._forward_decoder_step
        # st_feat = rearrange(final_output[-1], "b v l c -> (b v) l c")
        # last_feat = self.dec_norm(last_feat)
        # final_output[-1] = rearrange(last_feat, "(b v) l c -> b v l c", b=b, v=v)
        
        return final_output

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat=None,
        pose_pos=None,
        # init_state_feat,
        # img_mask=None,
        # reset_mask=None,
        # update=None,
    ):
        # new_state_feat, dec = self._decoder(
        new_state_feat, dec = self._rnn_decoder(
            # state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos
            state_feat, state_pos, current_feat, current_pos,
        )
        # new_state_feat = new_state_feat[-1] # ! return all layer output, since serving as f0 prediction
        return new_state_feat, dec





class AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn_OneCABlk(AsymmetricCroCoMulti_RW_CUT3R_MemoryFullAttn):
    def __init__(self, cfg: BackboneCrocoCfg, d_in: int) -> None:
        super().__init__(cfg, d_in)

        # ! delete the state-relevant modules
        del self.dec_norm_state
        del self.register_tokens
        del self.decoder_embed_state 

        del self.dec_blocks # remove CA blocks 1, CA blocks 2 for decoder-only inference

        self.enable_temporal_pe = True # ! use PE to denote frame0

        if self.enable_temporal_pe:
            embed_dim_temporal = self.dec_embed_dim
            # temporal_interpolation_scale = 1.0
            # https://github.com/huggingface/diffusers/blob/edb8c1bce67e81f0de90a7e4c16b2f6537d39f2d/src/diffusers/models/embeddings.py#L859C1-L862C10
            temporal_size = 64 # ! supports up to K frames as input.
            grid_t = torch.arange(temporal_size, dtype=torch.float32)
            # middle_idx = temporal_size // 2
            # grid_t = list(range(0, temporal_size))
            # grid_t = torch.Tensor([grid_t[middle_idx]] + grid_t[:middle_idx] + grid_t[middle_idx+1:]).to(torch.float32)
            pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t, output_type="pt") # shape: temporal_size * 768

            # ! TODO, hard coded. middle-frame f0 enabled here.
            self.pos_embed_temporal = pos_embed_temporal.to(torch.float32).unsqueeze(0) # type: ignore


    def _forward_decoder_step(
        self,
        # views, 
        feat,
        pos,
        ret_state=False
    ):
        '''basically, the state_feat (global memory) stays the same, while the second CA sees all the previous hidden states.
        '''
        # state_feat, state_pos = self._init_state(feat, pos) # retrieve + nn.Linear retrieval

        ress = []

        # ! for loop here
        for i in range(1, 1+feat.shape[1]): # ! views available in each batch
            # st()
            # feat_i = rearrange(feat[:, :i], 'b v l c -> b (v l) c') # ! B L C
            # pos_i = rearrange(pos[:, :i], 'b v l c -> b (v l) c')

            # new_state_feat, dec = self._recurrent_rollout(
            dec = self._recurrent_rollout(
                # None, 
                # None,
                # state_feat.contiguous(),
                # state_pos.contiguous(), # type: ignore
                feat[:, :i].contiguous(),
                pos[:, :i].contiguous(),
                None, # disabled for now
                None,
            )

            # if ret_state:
            #     ress.append((new_state_feat, dec))
            # else:
            ress.append(dec) 

            # update_mask = 1 # ! hard coded here
            # state_feat = new_state_feat * update_mask + state_feat * (
            #     1 - update_mask
            # )  # update global state
            # st()
            pass

        return ress

    def _rnn_decoder(self, f_img, pos_img, *args, **kwargs):

        # v = f_img.shape[1]

        f_img_history, pos_img_history = None, None
        has_causal_mem = False

        # ! add temporal PE to denote which frame is 0
        f_img = f_img + self.pos_embed_temporal[:, :f_img.shape[1]].to(f_img).unsqueeze(2)

        if f_img.shape[1] > 1:
            f_img, f_img_history = f_img[:, -1].contiguous(), rearrange(f_img[:, :-1], 'b v l c -> b (v l) c').contiguous()
            pos_img, pos_img_history = pos_img[:, -1].contiguous(), rearrange(pos_img[:, :-1], 'b v l c -> b (v l) c').contiguous()
            has_causal_mem = True
        else: # do self attention only
            f_img, pos_img = f_img[:, 0].contiguous(), pos_img[:, 0].contiguous() # RoPE only accepts 3-dim input

        if has_causal_mem: # decoder-only query kv
            mem_to_read_from = f_img_history.contiguous() # type: ignore
            pos_mem = pos_img_history.contiguous() # type: ignore
        else:
            mem_to_read_from = f_img.contiguous()
            pos_mem = pos_img.contiguous()


        final_output = [f_img] 

        # for idx, (blk_state, blk_img) in enumerate(zip(self.dec_blocks, self.dec_blocks2)): # ! 12 blocks of decoder.
        for idx, (blk_img) in enumerate(self.dec_blocks2): # ! 12 blocks of decoder.

            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                # f_state, _ = checkpoint(
                #     blk_state,
                #     *final_output[-1][::+1], # f_state query f_img here
                #     pos_state,
                #     pos_img,
                #     use_reentrant=True,
                # )# type: ignore
                f_img, _ = checkpoint(
                    blk_img,
                    # *final_output[-1][::-1],
                    # pos_img,
                    # pos_state,
                    f_img,
                    mem_to_read_from,
                    pos_img,
                    pos_mem,
                    use_reentrant=True,
                ) # type: ignore
            else:
                # f_state, _ = blk_state(*final_output[-1][::+1], pos_state, pos_img) # pos_* are the corresponding coord_ij for RoPE. only depends on the input resolution.
                # f_img, _ = blk_img(*final_output[-1][::-1], pos_img, pos_state)
                f_img, _ = blk_img(f_img, mem_to_read_from, pos_img, pos_mem)
            # final_output.append((f_state, f_img))
            final_output.append(f_img)


        final_output[-1] = self.dec_norm(final_output[-1])

        # return zip(*final_output)
        return final_output


    def _recurrent_rollout(
        self,
        # state_feat,
        # state_pos,
        current_feat,
        current_pos,
        pose_feat=None,
        pose_pos=None,
        # init_state_feat,
        # img_mask=None,
        # reset_mask=None,
        # update=None,
    ):
        # new_state_feat, dec = self._decoder(
        dec = self._rnn_decoder(
            current_feat, current_pos,
        )
        # new_state_feat = new_state_feat[-1]
        # return new_state_feat, dec
        return dec
