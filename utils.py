import torch
from diffusers.utils.torch_utils import randn_tensor

import json, os, cv2
from PIL import Image
import numpy as np

def parse_config(config):
    with open(config, 'r') as f:
        config = json.load(f)
    return config

def load_config(config):
    activate_layer_indices_list = config['inference_info']['activate_layer_indices_list']
    activate_step_indices_list = config['inference_info']['activate_step_indices_list']
    ref_seeds = config['reference_info']['ref_seeds']
    inf_seeds = config['inference_info']['inf_seeds']

    attn_map_save_steps = config['inference_info']['attn_map_save_steps']
    precomputed_path = config['precomputed_path']
    guidance_scale = config['guidance_scale']
    use_inf_negative_prompt = config['inference_info']['use_negative_prompt']

    style_name_list = config["style_name_list"]
    ref_object_list = config["reference_info"]["ref_object_list"]
    inf_object_list = config["inference_info"]["inf_object_list"]
    ref_with_style_description = config['reference_info']['with_style_description']
    inf_with_style_description = config['inference_info']['with_style_description']


    use_shared_attention = config['inference_info']['use_shared_attention']
    adain_queries = config['inference_info']['adain_queries']
    adain_keys = config['inference_info']['adain_keys']
    adain_values = config['inference_info']['adain_values']
    use_advanced_sampling = config['inference_info']['use_advanced_sampling']

    out = [
        activate_layer_indices_list, activate_step_indices_list,
        ref_seeds, inf_seeds,
        attn_map_save_steps, precomputed_path, guidance_scale, use_inf_negative_prompt,
        style_name_list, ref_object_list, inf_object_list, ref_with_style_description, inf_with_style_description,
        use_shared_attention, adain_queries, adain_keys, adain_values, use_advanced_sampling

    ]
    return out

def memory_efficient(model, device):
    try:
        model.to(device)
    except Exception as e:
        print("Error moving model to device:", e)

    try:
        model.enable_model_cpu_offload()
    except AttributeError:
        print("enable_model_cpu_offload is not supported.")
    try:
        model.enable_vae_slicing()
    except AttributeError:
        print("enable_vae_slicing is not supported.")

    try:
        model.enable_vae_tiling()
    except AttributeError:
        print("enable_vae_tiling is not supported.")

    try:
        model.enable_xformers_memory_efficient_attention()
    except AttributeError:
        print("enable_xformers_memory_efficient_attention is not supported.")

def init_latent(model, device_name='cuda', dtype=torch.float16, seed=None):
    scale_factor = model.vae_scale_factor
    sample_size = model.default_sample_size
    latent_dim = model.unet.config.in_channels

    height = sample_size * scale_factor
    width = sample_size * scale_factor

    shape = (1, latent_dim, height // scale_factor, width // scale_factor)

    device = torch.device(device_name)
    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None

    latent = randn_tensor(shape, generator=generator, dtype=dtype, device=device)

    return latent


def get_canny_edge_array(canny_img_path, threshold1=100,threshold2=200):
    canny_image_list = []

    # check if canny_img_path is a directory
    if os.path.isdir(canny_img_path):
        canny_img_list = os.listdir(canny_img_path)
        for canny_img in canny_img_list:
            canny_image_tmp = Image.open(os.path.join(canny_img_path, canny_img))
            #resize image into1024x1024
            canny_image_tmp = canny_image_tmp.resize((1024,1024))
            canny_image_tmp = np.array(canny_image_tmp)
            canny_image_tmp = cv2.Canny(canny_image_tmp, threshold1, threshold2)
            canny_image_tmp = canny_image_tmp[:, :, None]
            canny_image_tmp = np.concatenate([canny_image_tmp, canny_image_tmp, canny_image_tmp], axis=2)
            canny_image = Image.fromarray(canny_image_tmp)
            canny_image_list.append(canny_image)

    return canny_image_list

def get_depth_map(image, feature_extractor, depth_estimator, device='cuda'):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast(device):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))

    return image

def get_depth_edge_array(depth_img_path, feature_extractor, depth_estimator, device='cuda'):
    depth_image_list = []

    # check if canny_img_path is a directory
    if os.path.isdir(depth_img_path):
        depth_img_list = os.listdir(depth_img_path)
        for depth_img in depth_img_list:
            depth_image_tmp = Image.open(os.path.join(depth_img_path, depth_img)).convert('RGB')

            # get depth map
            depth_map = get_depth_map(depth_image_tmp, feature_extractor, depth_estimator, device)
            depth_image_list.append(depth_map)

    return depth_image_list