import torch
from pipelines.inverted_ve_pipeline import STYLE_DESCRIPTION_DICT, create_image_grid
import os
from pipelines.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
import argparse
from utils import parse_config, memory_efficient, init_latent, load_config

parser = argparse.ArgumentParser()
parser.add_argument('--style', type=str, default='fire')
args = parser.parse_args()


if __name__ == "__main__":

    # load pre_saved_json
    config_path = os.path.join("./config", "{}.json".format(args.style))
    config = parse_config(config_path)

    result_dir = 'results'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    ref_dir = "./assets/ref" # generated images
    if not os.path.exists(ref_dir):
        os.makedirs(ref_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.float16

    # load config
    activate_layer_indices_list, activate_step_indices_list,\
    ref_seeds, inf_seeds,\
    attn_map_save_steps, precomputed_path, guidance_scale, use_inf_negative_prompt,\
    style_name_list, ref_object_list, inf_object_list, ref_with_style_description, inf_with_style_description,\
    use_shared_attention, adain_queries, adain_keys, adain_values, use_advanced_sampling\
        = load_config(config) # load config


    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch_dtype)
    print('SDXL')
    memory_efficient(pipe, device)

    for ref_object in ref_object_list:
        for inf_object in inf_object_list:
            for style_name in style_name_list:
                style_description_pos, style_description_neg = STYLE_DESCRIPTION_DICT[style_name][0], STYLE_DESCRIPTION_DICT[style_name][1]

                if ref_with_style_description:
                    ref_prompt = style_description_pos.replace("{object}",ref_object)
                else:
                    ref_prompt = ref_object

                if inf_with_style_description:
                    inf_prompt = style_description_pos.replace("{object}",inf_object)
                else:
                    inf_prompt = inf_object

                for activate_layer_indices in activate_layer_indices_list:

                    for activate_step_indices in activate_step_indices_list:

                        str_activate_layer, str_activate_step = pipe.activate_layer(activate_layer_indices=activate_layer_indices,
                                                                                    attn_map_save_steps=attn_map_save_steps,
                                                                                    activate_step_indices=activate_step_indices,
                                                                                    use_shared_attention=use_shared_attention,
                                                                                    adain_queries=adain_queries,
                                                                                    adain_keys=adain_keys,
                                                                                    adain_values=adain_values,
                                                                                    )


                        for ref_seed in ref_seeds:
                            # ref_latent = pipe.get_init_latent(precomputed_path,ref_seed)
                            ref_latent = init_latent(pipe, device_name=device, dtype=torch_dtype, seed=ref_seed)

                            latents = [ref_latent]

                            for inf_seed in inf_seeds:
                                # latents.append(pipe.get_init_latent(precomputed_path, inf_seed))
                                inf_latent = init_latent(pipe, device_name=device, dtype=torch_dtype, seed=inf_seed)
                                latents.append(inf_latent)


                            latents = torch.cat(latents, dim=0)
                            latents.to(device)

                            images = pipe(
                                prompt=ref_prompt,
                                negative_prompt = style_description_neg,
                                guidance_scale = guidance_scale,
                                latents=latents,
                                num_images_per_prompt = len(inf_seeds)+1,
                                target_prompt = inf_prompt,
                                use_inf_negative_prompt = use_inf_negative_prompt,
                                use_advanced_sampling=use_advanced_sampling
                            )[0]

                            ref_image = images[0]
                            ref_image.save(os.path.join(ref_dir, f"ref_{style_name}_{ref_object}.png"))

                            #make grid
                            n_row = 1
                            n_col = len(inf_seeds)+1
                            grid = create_image_grid(images, n_row, n_col)

                            new_inf_seeds = [inf_seed for inf_seed in inf_seeds]

                            save_name = f"style_{style_name}_ref_{ref_seed}_{ref_object}_inf_{new_inf_seeds}_{inf_object}activate_layer_{str_activate_layer}_step_{str_activate_step}.png"
                            save_path = os.path.join(result_dir, save_name)

                            grid.save(save_path)
                            print("saved to ", save_path)



                    
                    
                    


                





            
