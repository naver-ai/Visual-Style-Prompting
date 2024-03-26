import torch
from diffusers import DDIMScheduler
from PIL import Image
import os
from pipelines.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from pipelines.inverted_ve_pipeline import create_image_grid
from utils import memory_efficient, init_latent
import argparse
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    torch_dtype = torch.float32
else:
    torch_dtype = torch.float16

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='./assets/real_dir')
parser.add_argument('--tar_obj', type=str, default='cat')
parser.add_argument('--guidance_scale', type=float, default=7.0)
parser.add_argument('--output_num', type=int, default=5)
parser.add_argument('--activate_step', type=int, default=50)

args = parser.parse_args()

def create_number_list(n):
    return list(range(n + 1))

def create_nested_list(t):
    return [[0, t]]

def create_prompt(style_name):
    pre_prompt_dicts = {
        "kids drawing": ("kids drawing of {prompt}. crayon, colored pencil, marker", ""),
        "self portrait": ("{prompt} of van gogh", ""),
        "Sunflowers": ("{prompt} of van gogh", ""),
        "The kiss": ("{prompt} of gustav klimt", ""),
        "Vitruvian Man": ("{prompt} of leonardo da vinci", ""),
        "Weeping woman": ("{prompt} of pablo picasso", ""),
        "The scream": ("{prompt} of edvard munch", ""),
        "The starry night": ("{prompt} of van gogh", ""),
        "Starry night over the rhone": ("{prompt} of van gogh", "")
    }

    if style_name in pre_prompt_dicts.keys():
        return pre_prompt_dicts[style_name]
    else:
        return None, None


def blip_inf_prompt(image):
    inputs = blip_processor(images=image, return_tensors="pt").to(device, torch.float16)

    generated_ids = blip_model.generate(**inputs)
    generated_text = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text


tar_seeds = create_number_list(args.output_num)
activate_step_indices = create_nested_list(args.activate_step)

img_path = args.img_path
tar_obj = args.tar_obj
guidance_scale = args.guidance_scale

results_dir = 'results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


image_name_list = os.listdir(img_path)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    torch_dtype = torch.float32
else:
    torch_dtype = torch.float16


pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype)
print('SDXL')
memory_efficient(pipe, device)

blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch_dtype).to(device)

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

str_activate_layer, str_activate_step = pipe.activate_layer(
                        activate_layer_indices=[[0, 0], [128, 140]], 
                        attn_map_save_steps=[], 
                        activate_step_indices=activate_step_indices,
                        use_shared_attention=False,
)


with torch.no_grad():
    for image_name in image_name_list:
        image_path = os.path.join(img_path, image_name)

        real_img = Image.open(image_path).resize((1024, 1024), resample=Image.BICUBIC)


        style_name = image_name.split('.')[0]

        latents = []

        base_prompt, negative_prompt = create_prompt(style_name)
        if base_prompt is not None:
            ref_prompt = base_prompt.replace("{prompt}", style_name)
            inf_prompt = base_prompt.replace("{prompt}", tar_obj)
        else:
            ref_prompt = blip_inf_prompt(real_img)
            inf_prompt = tar_obj

        for tar_seed in tar_seeds:
            latents.append(init_latent(model=pipe, device_name=device, dtype=torch_dtype, seed=tar_seed))

        latents = torch.cat(latents, dim=0)

        images = pipe(
            prompt=ref_prompt,
            guidance_scale=guidance_scale,
            latents=latents,
            num_images_per_prompt=len(tar_seeds),
            target_prompt=inf_prompt,
            use_inf_negative_prompt=False,
            use_advanced_sampling=False,
            use_prompt_as_null=True,
            image=real_img
        )[0]
        # [real image, fake1, fake2, ... ]
        save_path = os.path.join(results_dir, "{}_{}.png".format(style_name, tar_obj))

        n_row = 1
        n_col = len(tar_seeds)
        grid = create_image_grid(images, n_row, n_col)

        grid.save(save_path)
        print(f"saved to {save_path}")
