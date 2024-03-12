from __future__ import annotations
from diffusers import StableDiffusionPipeline
import torch
from dataclasses import dataclass
from typing import Callable, List, Optional, Union, Any, Dict
import numpy as np
from diffusers.utils import deprecate, logging, BaseOutput
from einops import rearrange, repeat
from torch.nn.functional import grid_sample
from torch.nn import functional as nnf
import torchvision.transforms as T
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers.models import AutoencoderKL, UNet2DConditionModel, attention_processor
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import PIL
from PIL import Image
from kornia.morphology import dilation
from collections import OrderedDict
from packaging import version
import inspect
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
import torch.nn as nn

T = torch.Tensor


@dataclass(frozen=True)
class StyleAlignedArgs:
    share_group_norm: bool = True
    share_layer_norm: bool = True,
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = False
    full_attention_share: bool = False
    keys_scale: float = 1.
    only_self_level: float = 0.

def expand_first(feat: T, scale=1., ) -> T:
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import StableDiffusionPipeline

        >>> pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
        >>> pipe = pipe.to("cuda")

        >>> prompt = "a photo of an astronaut riding a horse on mars"
        >>> image = pipe(prompt).images[0]
        ```
"""

# ACTIVATE_STEP_CANDIDATE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 1]


def create_image_grid(image_list, rows, cols, padding=10):
    # Ensure the number of rows and columns doesn't exceed the number of images
    rows = min(rows, len(image_list))
    cols = min(cols, len(image_list))

    # Get the dimensions of a single image
    image_width, image_height = image_list[0].size

    # Calculate the size of the output image
    grid_width = cols * (image_width + padding) - padding
    grid_height = rows * (image_height + padding) - padding

    # Create an empty grid image
    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

    # Paste images into the grid
    for i, img in enumerate(image_list[:rows * cols]):
        row = i // cols
        col = i % cols
        x = col * (image_width + padding)
        y = row * (image_height + padding)
        grid_image.paste(img, (x, y))

    return grid_image




class CrossFrameAttnProcessor_backup:
    def __init__(self, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # elif attn.cross_attention_norm:
        #     encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        # Sparse Attention
        if not is_cross_attention:
            video_length = key.size()[0] // self.unet_chunk_size
            # former_frame_index = torch.arange(video_length) - 1
            # former_frame_index[0] = 0
            # import pdb; pdb.set_trace()

            # if video_length > 3:
            #     import pdb; pdb.set_trace()
            former_frame_index = [0] * video_length
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, former_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, former_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")


        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    

class SharedAttentionProcessor:
    def __init__(self, 
                 adain_keys=True, 
                 adain_queries=True, 
                 adain_values=False, 
                 keys_scale=1.,
                 attn_map_save_steps=[]):
        super().__init__()
        self.adain_queries = adain_queries
        self.adain_keys = adain_keys
        self.adain_values = adain_values
        # self.full_attention_share = style_aligned_args.full_attention_share
        self.keys_scale = keys_scale
        self.attn_map_save_steps = attn_map_save_steps


    def __call__(
            self,
            attn: attention_processor.Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **kwargs
    ):
        
        if not hasattr(attn, "attn_map"):
            setattr(attn, "attn_map", {})
            setattr(attn, "inference_step", 0)
        else:
            attn.inference_step += 1

        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        is_cross_attention = encoder_hidden_states is not None

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # elif attn.cross_attention_norm:
        #     encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # if self.step >= self.start_inject:

        
        if not is_cross_attention:# and self.share_attention:
            if self.adain_queries:
                query = adain(query)
            if self.adain_keys:
                key = adain(key)
            if self.adain_values:
                value = adain(value)
            key = concat_first(key, -2, scale=self.keys_scale)
            value = concat_first(value, -2)
            hidden_states = nnf.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            hidden_states = nnf.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )



        
        # hidden_states = adain(hidden_states)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class SharedAttentionProcessor_v2:
    def __init__(self, 
                 adain_keys=True, 
                 adain_queries=True, 
                 adain_values=False, 
                 keys_scale=1.,
                 attn_map_save_steps=[]):
        super().__init__()
        self.adain_queries = adain_queries
        self.adain_keys = adain_keys
        self.adain_values = adain_values
        # self.full_attention_share = style_aligned_args.full_attention_share
        self.keys_scale = keys_scale
        self.attn_map_save_steps = attn_map_save_steps


    def __call__(
            self,
            attn: attention_processor.Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            **kwargs
    ):
        
        if not hasattr(attn, "attn_map"):
            setattr(attn, "attn_map", {})
            setattr(attn, "inference_step", 0)
        else:
            attn.inference_step += 1

        residual = hidden_states
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        is_cross_attention = encoder_hidden_states is not None

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)


        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # elif attn.cross_attention_norm:
        #     encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        tmp_query_shape = query.shape
        tmp_key_shape = key.shape
        tmp_value_shape = value.shape


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        # if self.step >= self.start_inject:

        
        if not is_cross_attention:# and self.share_attention:
            if self.adain_queries:
                query = adain(query)
            if self.adain_keys:
                key = adain(key)
            if self.adain_values:
                value = adain(value)
            key = concat_first(key, -2, scale=self.keys_scale)
            value = concat_first(value, -2)
            # hidden_states = nnf.scaled_dot_product_attention(
            #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            # )

            if attn.inference_step in self.attn_map_save_steps:

                query = query.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                key  = key.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                value = value.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

                query = attn.head_to_batch_dim(query)
                key = attn.head_to_batch_dim(key)
                value = attn.head_to_batch_dim(value)

                attention_probs = attn.get_attention_scores(query, key, attention_mask)

                if attn.inference_step in self.attn_map_save_steps:
                    attn.attn_map[attn.inference_step] = attention_probs.clone().cpu().detach()

                hidden_states = torch.bmm(attention_probs, value)
                hidden_states = attn.batch_to_head_dim(hidden_states)
            else:
                hidden_states = nnf.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                    # hidden_states = adain(hidden_states)
                hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
                hidden_states = hidden_states.to(query.dtype)

        else:

            hidden_states = nnf.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            # hidden_states = adain(hidden_states)
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if attn.inference_step == 49:
            #initialize inference step
            attn.inference_step = -1

        return hidden_states


def swapping_attention(key, value, chunk_size=2):
    chunk_length = key.size()[0] // chunk_size  # [text-condition, null-condition]
    reference_image_index = [0] * chunk_length  # [0 0 0 0 0]
    key = rearrange(key, "(b f) d c -> b f d c", f=chunk_length)
    key = key[:, reference_image_index]  # ref to all
    key = rearrange(key, "b f d c -> (b f) d c")
    value = rearrange(value, "(b f) d c -> b f d c", f=chunk_length)
    value = value[:, reference_image_index]  # ref to all
    value = rearrange(value, "b f d c -> (b f) d c")

    return key, value
    
class CrossFrameAttnProcessor:
    def __init__(self, unet_chunk_size=2, attn_map_save_steps=[],activate_step_indices=None):
        self.unet_chunk_size = unet_chunk_size
        self.attn_map_save_steps = attn_map_save_steps
        self.activate_step_indices = activate_step_indices

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        
        if not hasattr(attn, "attn_map"):
            setattr(attn, "attn_map", {})
            setattr(attn, "inference_step", 0)
        else:
            attn.inference_step += 1
        
        

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        is_in_inference_step = False

        if self.activate_step_indices is not None:
            for activate_step_index in self.activate_step_indices:
                if attn.inference_step >= activate_step_index[0] and attn.inference_step <= activate_step_index[1]:
                    is_in_inference_step = True
                    break

        # Swapping Attention
        if not is_cross_attention and is_in_inference_step:
            key, value = swapping_attention(key, value, self.unet_chunk_size)




        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if attn.inference_step in self.attn_map_save_steps:
            attn.attn_map[attn.inference_step] = attention_probs.clone().cpu().detach()

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.inference_step == 49:
            attn.inference_step = -1

        return hidden_states




class CrossFrameAttnProcessor4Inversion:
    def __init__(self, unet_chunk_size=2, attn_map_save_steps=[],activate_step_indices=None):
        self.unet_chunk_size = unet_chunk_size
        self.attn_map_save_steps = attn_map_save_steps
        self.activate_step_indices = activate_step_indices

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        
        if not hasattr(attn, "attn_map"):
            setattr(attn, "attn_map", {})
            setattr(attn, "inference_step", 0)
        else:
            attn.inference_step += 1
        
        

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # elif attn.cross_attention_norm:
        #     encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        is_in_inference_step = False

        if self.activate_step_indices is not None:
            for activate_step_index in self.activate_step_indices:
                if attn.inference_step >= activate_step_index[0] and attn.inference_step <= activate_step_index[1]:
                    is_in_inference_step = True
                    break

        # Swapping Attention
        if not is_cross_attention and is_in_inference_step:
            key, value = swapping_attention(key, value, self.unet_chunk_size)



        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # if attn.inference_step > 45 and attn.inference_step < 50:
        # if attn.inference_step == 42 or attn.inference_step==49:
        if attn.inference_step in self.attn_map_save_steps:
            attn.attn_map[attn.inference_step] = attention_probs.clone().cpu().detach()

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.inference_step == 49:
            #initialize inference step
            attn.inference_step = -1

        return hidden_states



class CrossFrameAttnProcessor_store:
    def __init__(self, unet_chunk_size=2, attn_map_save_steps=[]):
        self.unet_chunk_size = unet_chunk_size
        self.attn_map_save_steps = attn_map_save_steps

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        # elif attn.cross_attention_norm:
        #     encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Swapping Attention
        if not is_cross_attention:
            key, value = swapping_attention(key, value, self.unet_chunk_size)


        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if not hasattr(attn, "attn_map"):
            setattr(attn, "attn_map", {})
            setattr(attn, "inference_step", 0)
        else:
            attn.inference_step += 1
        
        
        # if attn.inference_step > 45 and attn.inference_step < 50:
        # if attn.inference_step == 42 or attn.inference_step==49:
        if attn.inference_step in self.attn_map_save_steps:
            attn.attn_map[attn.inference_step] = attention_probs.clone().cpu().detach()

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    
class InvertedVEAttnProcessor:
    def __init__(self, unet_chunk_size=2, scale=1.0):
        self.unet_chunk_size = unet_chunk_size
        self.scale = scale

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        #Dual Attention
        if not is_cross_attention:
            ve_key = key.clone()
            ve_value = value.clone()
            video_length = ve_key.size()[0] // self.unet_chunk_size

            former_frame_index = [0] * video_length
            ve_key = rearrange(ve_key, "(b f) d c -> b f d c", f=video_length)
            ve_key = ve_key[:, former_frame_index]
            ve_key = rearrange(ve_key, "b f d c -> (b f) d c")
            ve_value = rearrange(ve_value, "(b f) d c -> b f d c", f=video_length)
            ve_value = ve_value[:, former_frame_index]
            ve_value = rearrange(ve_value, "b f d c -> (b f) d c")

            ve_key = attn.head_to_batch_dim(ve_key)
            ve_value = attn.head_to_batch_dim(ve_value)
            ve_query = attn.head_to_batch_dim(query)

            ve_attention_probs = attn.get_attention_scores(ve_query, ve_key, attention_mask)
            ve_hidden_states = torch.bmm(ve_attention_probs, ve_value)
            ve_hidden_states = attn.batch_to_head_dim(ve_hidden_states)
            ve_hidden_states[0,...] = 0
            ve_hidden_states[video_length,...] = 0

            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            hidden_states = hidden_states + ve_hidden_states * self.scale

        else:
            query = attn.head_to_batch_dim(query)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

class AttnProcessor(nn.Module):
    r"""
    Default processor for performing attention-related computations.
    """
    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
    ):
        super().__init__()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):

        residual = hidden_states
        # import pdb; pdb.set_trace()
        # if attn.spatial_norm is not None:
        #     hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        # if attn.group_norm is not None:
        #     hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    

@dataclass
class StableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]

class FrozenDict(OrderedDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for key, value in self.items():
            setattr(self, key, value)

        self.__frozen = True

    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __setattr__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setattr__(name, value)

    def __setitem__(self, name, value):
        if hasattr(self, "__frozen") and self.__frozen:
            raise Exception(f"You cannot use ``__setattr__`` on a {self.__class__.__name__} instance.")
        super().__setitem__(name, value)


class InvertedVEPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        # super().__init__()
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler,
                         safety_checker, feature_extractor, requires_safety_checker)

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError("`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

        if self.safety_checker is not None:
            cpu_offload(self.safety_checker, execution_device=device, offload_buffers=True)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        if self.safety_checker is not None:
            _, hook = cpu_offload_with_hook(self.safety_checker, device, prev_module_hook=hook)

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device


    def run_safety_checker(self, image, device, dtype):
        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(device)
            image, has_nsfw_concept = self.safety_checker(
                images=image, clip_input=safety_checker_input.pixel_values.to(dtype)
            )
        else:
            has_nsfw_concept = None
        return image, has_nsfw_concept

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        target_prompt: Optional[str] = None,
        # device: Optional[Union[str, torch.device]] = "cpu",
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttnProcessor` as defined under
                `self.processor` in
                [diffusers.cross_attention](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/cross_attention.py).

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        # import pdb; pdb.set_trace()

        
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # import pdb; pdb.set_trace()

        if target_prompt is not None:
            target_prompt_embeds = self._encode_prompt(
                target_prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                negative_prompt_embeds=negative_prompt_embeds,
            )
            prompt_embeds[num_images_per_prompt+1: ] = target_prompt_embeds[num_images_per_prompt+1:]
        import pdb; pdb.set_trace()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil":
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

            # 10. Convert to PIL
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


ACTIVATE_LAYER_CANDIDATE= [
        'down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 
        'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor', 
        'down_blocks.1.attentions.0.transformer_blocks.1.attn1.processor', 
        'down_blocks.1.attentions.0.transformer_blocks.1.attn2.processor',
        'down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 
        'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor', 
        'down_blocks.1.attentions.1.transformer_blocks.1.attn1.processor', 
        'down_blocks.1.attentions.1.transformer_blocks.1.attn2.processor', #8

        'down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor',
        'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor', 
        'down_blocks.2.attentions.0.transformer_blocks.1.attn1.processor', 
        'down_blocks.2.attentions.0.transformer_blocks.1.attn2.processor',
        'down_blocks.2.attentions.0.transformer_blocks.2.attn1.processor', 
        'down_blocks.2.attentions.0.transformer_blocks.2.attn2.processor', 
        'down_blocks.2.attentions.0.transformer_blocks.3.attn1.processor',
        'down_blocks.2.attentions.0.transformer_blocks.3.attn2.processor', 
        'down_blocks.2.attentions.0.transformer_blocks.4.attn1.processor', 
        'down_blocks.2.attentions.0.transformer_blocks.4.attn2.processor', 
        'down_blocks.2.attentions.0.transformer_blocks.5.attn1.processor', 
        'down_blocks.2.attentions.0.transformer_blocks.5.attn2.processor',
        'down_blocks.2.attentions.0.transformer_blocks.6.attn1.processor',
        'down_blocks.2.attentions.0.transformer_blocks.6.attn2.processor', 
        'down_blocks.2.attentions.0.transformer_blocks.7.attn1.processor', 
        'down_blocks.2.attentions.0.transformer_blocks.7.attn2.processor', 
        'down_blocks.2.attentions.0.transformer_blocks.8.attn1.processor', 
        'down_blocks.2.attentions.0.transformer_blocks.8.attn2.processor',
        'down_blocks.2.attentions.0.transformer_blocks.9.attn1.processor',
        'down_blocks.2.attentions.0.transformer_blocks.9.attn2.processor', #20

        'down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor',
        'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor', 
        'down_blocks.2.attentions.1.transformer_blocks.1.attn1.processor', 
        'down_blocks.2.attentions.1.transformer_blocks.1.attn2.processor', 
        'down_blocks.2.attentions.1.transformer_blocks.2.attn1.processor',
        'down_blocks.2.attentions.1.transformer_blocks.2.attn2.processor',
        'down_blocks.2.attentions.1.transformer_blocks.3.attn1.processor', 
        'down_blocks.2.attentions.1.transformer_blocks.3.attn2.processor', 
        'down_blocks.2.attentions.1.transformer_blocks.4.attn1.processor', 
        'down_blocks.2.attentions.1.transformer_blocks.4.attn2.processor',
        'down_blocks.2.attentions.1.transformer_blocks.5.attn1.processor',
        'down_blocks.2.attentions.1.transformer_blocks.5.attn2.processor',
        'down_blocks.2.attentions.1.transformer_blocks.6.attn1.processor',
        'down_blocks.2.attentions.1.transformer_blocks.6.attn2.processor',
        'down_blocks.2.attentions.1.transformer_blocks.7.attn1.processor', 
        'down_blocks.2.attentions.1.transformer_blocks.7.attn2.processor', 
        'down_blocks.2.attentions.1.transformer_blocks.8.attn1.processor', 
        'down_blocks.2.attentions.1.transformer_blocks.8.attn2.processor',
        'down_blocks.2.attentions.1.transformer_blocks.9.attn1.processor',
        'down_blocks.2.attentions.1.transformer_blocks.9.attn2.processor',#20

        'mid_block.attentions.0.transformer_blocks.0.attn1.processor', 
        'mid_block.attentions.0.transformer_blocks.0.attn2.processor', 
        'mid_block.attentions.0.transformer_blocks.1.attn1.processor', 
        'mid_block.attentions.0.transformer_blocks.1.attn2.processor', 
        'mid_block.attentions.0.transformer_blocks.2.attn1.processor',
        'mid_block.attentions.0.transformer_blocks.2.attn2.processor', 
        'mid_block.attentions.0.transformer_blocks.3.attn1.processor', 
        'mid_block.attentions.0.transformer_blocks.3.attn2.processor', 
        'mid_block.attentions.0.transformer_blocks.4.attn1.processor', 
        'mid_block.attentions.0.transformer_blocks.4.attn2.processor', 
        'mid_block.attentions.0.transformer_blocks.5.attn1.processor', 
        'mid_block.attentions.0.transformer_blocks.5.attn2.processor',
        'mid_block.attentions.0.transformer_blocks.6.attn1.processor', 
        'mid_block.attentions.0.transformer_blocks.6.attn2.processor', 
        'mid_block.attentions.0.transformer_blocks.7.attn1.processor',
        'mid_block.attentions.0.transformer_blocks.7.attn2.processor', 
        'mid_block.attentions.0.transformer_blocks.8.attn1.processor', 
        'mid_block.attentions.0.transformer_blocks.8.attn2.processor', 
        'mid_block.attentions.0.transformer_blocks.9.attn1.processor', 
        'mid_block.attentions.0.transformer_blocks.9.attn2.processor', #20

        'up_blocks.0.attentions.0.transformer_blocks.0.attn1.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.0.attn2.processor',
        'up_blocks.0.attentions.0.transformer_blocks.1.attn1.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.1.attn2.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.2.attn1.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.2.attn2.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.3.attn1.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.3.attn2.processor',
        'up_blocks.0.attentions.0.transformer_blocks.4.attn1.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.4.attn2.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.5.attn1.processor',
        'up_blocks.0.attentions.0.transformer_blocks.5.attn2.processor',
        'up_blocks.0.attentions.0.transformer_blocks.6.attn1.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.6.attn2.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.7.attn1.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.7.attn2.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.8.attn1.processor',
        'up_blocks.0.attentions.0.transformer_blocks.8.attn2.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.9.attn1.processor', 
        'up_blocks.0.attentions.0.transformer_blocks.9.attn2.processor',#20

        'up_blocks.0.attentions.1.transformer_blocks.0.attn1.processor', 
        'up_blocks.0.attentions.1.transformer_blocks.0.attn2.processor', 
        'up_blocks.0.attentions.1.transformer_blocks.1.attn1.processor', 
        'up_blocks.0.attentions.1.transformer_blocks.1.attn2.processor', 
        'up_blocks.0.attentions.1.transformer_blocks.2.attn1.processor', 
        'up_blocks.0.attentions.1.transformer_blocks.2.attn2.processor',
        'up_blocks.0.attentions.1.transformer_blocks.3.attn1.processor',
        'up_blocks.0.attentions.1.transformer_blocks.3.attn2.processor',
        'up_blocks.0.attentions.1.transformer_blocks.4.attn1.processor', 
        'up_blocks.0.attentions.1.transformer_blocks.4.attn2.processor', 
        'up_blocks.0.attentions.1.transformer_blocks.5.attn1.processor', 
        'up_blocks.0.attentions.1.transformer_blocks.5.attn2.processor',
        'up_blocks.0.attentions.1.transformer_blocks.6.attn1.processor', 
        'up_blocks.0.attentions.1.transformer_blocks.6.attn2.processor', 
        'up_blocks.0.attentions.1.transformer_blocks.7.attn1.processor', 
        'up_blocks.0.attentions.1.transformer_blocks.7.attn2.processor',
        'up_blocks.0.attentions.1.transformer_blocks.8.attn1.processor',
        'up_blocks.0.attentions.1.transformer_blocks.8.attn2.processor',
        'up_blocks.0.attentions.1.transformer_blocks.9.attn1.processor',
        'up_blocks.0.attentions.1.transformer_blocks.9.attn2.processor',#20

        'up_blocks.0.attentions.2.transformer_blocks.0.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.0.attn2.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.1.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.1.attn2.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.2.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.2.attn2.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.3.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.3.attn2.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.4.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.4.attn2.processor',
        'up_blocks.0.attentions.2.transformer_blocks.5.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.5.attn2.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.6.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.6.attn2.processor',
        'up_blocks.0.attentions.2.transformer_blocks.7.attn1.processor',
        'up_blocks.0.attentions.2.transformer_blocks.7.attn2.processor',
        'up_blocks.0.attentions.2.transformer_blocks.8.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.8.attn2.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.9.attn1.processor', 
        'up_blocks.0.attentions.2.transformer_blocks.9.attn2.processor', #20

        'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 
        'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor',
        'up_blocks.1.attentions.0.transformer_blocks.1.attn1.processor',
        'up_blocks.1.attentions.0.transformer_blocks.1.attn2.processor',
        'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 
        'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor', 
        'up_blocks.1.attentions.1.transformer_blocks.1.attn1.processor', 
        'up_blocks.1.attentions.1.transformer_blocks.1.attn2.processor',
        'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor',
        'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor', 
        'up_blocks.1.attentions.2.transformer_blocks.1.attn1.processor', 
        'up_blocks.1.attentions.2.transformer_blocks.1.attn2.processor',#12

]

STYLE_DESCRIPTION_DICT = {
    "chinese-ink-paint":("{object} in colorful chinese ink paintings style",""),
    "cloud":("Photography of {object}, realistic",""),
    "digital-art":("{object} in digital glitch arts style",""),
    "fire":("{object} photography, realistic, black background'",""),
    "klimt":("{object} in style of Gustav Klimt",""),
    "line-art":("line art drawing of {object} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",""),
    "low-poly":("low-poly style of {object} . low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition",
                            "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo"),
    "munch":("{object} in Edvard Munch style",""),
    "van-gogh":("{object}, Van Gogh",""),
    "totoro":("{object}, art by studio ghibli, cinematic, masterpiece,key visual, studio anime, highly detailed",
              "photo, deformed, black and white, realism, disfigured, low contrast"),
    
    "realistic":            ("A portrait of {object}, photorealistic, 35mm film, realistic",
                             "gray, ugly, deformed, noisy, blurry"),
                             
    "line_art":             ("line art drawing of {object} . professional, sleek, modern, minimalist, graphic, line art, vector graphics",
                            "anime, photorealistic, 35mm film, deformed, glitch, blurry, noisy, off-center, deformed, cross-eyed, closed eyes, bad anatomy, ugly, disfigured, mutated, realism, realistic, impressionism, expressionism, oil, acrylic"
                            ) ,

    "anime":                ("anime artwork of {object} . anime style, key visual, vibrant, studio anime, highly detailed",
                            "photo, deformed, black and white, realism, disfigured, low contrast"
                            ),
    
    "Artstyle_Pop_Art" :    ("pop Art style of {object} . bright colors, bold outlines, popular culture themes, ironic or kitsch",
                            "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, minimalist"
                            ),
    
    "Artstyle_Pointillism": ("pointillism style of {object} . composed entirely of small, distinct dots of color, vibrant, highly detailed",
                              "line drawing, smooth shading, large color fields, simplistic"
                              ),
    
    "origami":              ("origami style of {object} . paper art, pleated paper, folded, origami art, pleats, cut and fold, centered composition",
                             "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo"
                             ),
    
    "craft_clay":           ("play-doh style of {object} . sculpture, clay art, centered composition, Claymation",
                            "sloppy, messy, grainy, highly detailed, ultra textured, photo"
                            ),
    
    "low_poly" :            ("low-poly style of {object} . low-poly game art, polygon mesh, jagged, blocky, wireframe edges, centered composition",
                            "noisy, sloppy, messy, grainy, highly detailed, ultra textured, photo"
                            ),      
    
    "Artstyle_watercolor":  ("watercolor painting of {object} . vibrant, beautiful, painterly, detailed, textural, artistic",
                            "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
                            ),
    
    "Papercraft_Collage" : ("collage style of {object} . mixed media, layered, textural, detailed, artistic",
                            "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic"
                            ),
    
    "Artstyle_Impressionist" : ("impressionist painting of {object} . loose brushwork, vibrant color, light and shadow play, captures feeling over form",
                                "anime, photorealistic, 35mm film, deformed, glitch, low contrast, noisy"
                            ),
    "realistic_bg_black":("{object} photography, realistic, black background",
                          ""),
    "photography_realistic":("Photography of {object}, realistic",
                             ""),
    "digital_art":("{object} in digital glitch arts style.",
                    ""
                    ),
    "chinese_painting":("{object} in traditional a chinese ink painting style.",
                        ""
                        ),
    "no_style":("{object}",
    ""),
    "kid_drawing":("{object} in kid crayon drawings style.",""),
    "onepiece":("{object}, wanostyle, angry looking, straw hat, looking at viewer, solo, upper body, masterpiece, best quality, (extremely detailed), watercolor, illustration, depth of field, sketch, dark intense shadows, sharp focus, soft lighting, hdr, colorful, good composition, fire all around, spectacular, closed shirt",
                " watermark, text, error, blurry, jpeg artifacts, many objects, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature")
}