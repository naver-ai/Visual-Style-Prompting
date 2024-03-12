import torch
import os
from PIL import Image
import numpy as np
from ipycanvas import Canvas
import cv2

from visualize_attention_src.utils import get_image

exp_dir = "saved_attention_map_results"

style_name = "line_art"
src_name = "cat"
tgt_name = "dog"

steps = ["20"]
seed = "4"
saved_dtype = "tensor"


attn_map_raws = []
for step in steps:
    attn_map_name_wo_ext = f"attn_map_raw_{style_name}_src_{src_name}_tgt_{tgt_name}_activate_layer_(0, 0)(108, 140)_attn_map_step_{step}_seed_{seed}"  # new

    if saved_dtype == 'uint8':
        attn_map_name = attn_map_name_wo_ext + '_uint8.npy'
        attn_map_path = os.path.join(exp_dir, attn_map_name)
        attn_map_raws.append(np.load(attn_map_path, allow_pickle=True))

    else:
        attn_map_name = attn_map_name_wo_ext + '.pt'
        attn_map_path = os.path.join(exp_dir, attn_map_name)
        attn_map_raws.append(torch.load(attn_map_path))
        print(attn_map_path)

    attn_map_path = os.path.join(exp_dir, attn_map_name)

    print(f"{step} is on memory")

keys = [key for key in attn_map_raws[0].keys()]


print(len(keys))
key = keys[0]

########################
tgt_idx = 3 # indicating the location of generated images.

attn_map_paired_rgb_grid_name = f"{style_name}_src_{src_name}_tgt_{tgt_name}_scale_1.0_activate_layer_(0, 0)(108, 140)_seed_{seed}.png"

attn_map_paired_rgb_grid_path = os.path.join(exp_dir, attn_map_paired_rgb_grid_name)
print(attn_map_paired_rgb_grid_path)
attn_map_paired_rgb_grid = Image.open(attn_map_paired_rgb_grid_path)

attn_map_src_img = get_image(attn_map_paired_rgb_grid, row = 0, col = 0, image_size = 1024, grid_width = 10)
attn_map_tgt_img = get_image(attn_map_paired_rgb_grid, row = 0, col = tgt_idx, image_size = 1024, grid_width = 10)


h, w = 256, 256
num_of_grid = 64

plus_50 = 0

# key_idx_list = [0,2,4,6,8,10]
key_idx_list = [6, 28]
# (108 -> 0, 109 -> 1, ... , 140 -> 32)
# if Swapping Attentio nin (108, 140) layer , use key_idx_list = [6, 28].
# 6==early upblock, 28==late upblock

saved_attention_map_idx = [0]

source_image = attn_map_src_img
target_image = attn_map_tgt_img

# resize
source_image = source_image.resize((h, w))
target_image = target_image.resize((h, w))

# convert to numpy array
source_image = np.array(source_image)
target_image = np.array(target_image)

canvas = Canvas(width=4 * w, height=h * len(key_idx_list), sync_image_data=True)
canvas.put_image_data(source_image, w * 3, 0)
canvas.put_image_data(target_image, 0, 0)

canvas.put_image_data(source_image, w * 3, h)
canvas.put_image_data(target_image, 0, h)

# Display the canvas
# display(canvas)


def save_to_file(*args, **kwargs):
    canvas.to_file("my_file1.png")


# Listen to changes on the ``image_data`` trait and call ``save_to_file`` when it changes.
canvas.observe(save_to_file, "image_data")


def on_click(x, y):
    cnt = 0
    canvas.put_image_data(target_image, 0, 0)

    print(x, y)
    # draw a point
    canvas.fill_style = 'red'
    canvas.fill_circle(x, y, 4)

    for step_i, step in enumerate(range(len(saved_attention_map_idx))):

        attn_map_raw = attn_map_raws[step_i]

        for key_i, key_idx in enumerate(key_idx_list):
            key = keys[key_idx]

            num_of_grid = int(attn_map_raw[key].shape[-1] ** (0.5))

            # normalize x,y
            grid_x_idx = int(x / (w / num_of_grid))
            grid_y_idx = int(y / (h / num_of_grid))

            print(grid_x_idx, grid_y_idx)

            grid_idx = grid_x_idx + grid_y_idx * num_of_grid

            attn_map = attn_map_raw[key][tgt_idx * 10:10 + tgt_idx * 10, grid_idx, :]

            attn_map = attn_map.sum(dim=0)

            attn_map = attn_map.reshape(num_of_grid, num_of_grid)

            # process attn_map to pil
            attn_map = attn_map.detach().cpu().numpy()
            # attn_map = attn_map / attn_map.max()
            # normalized_attn_map = attn_map
            normalized_attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            normalized_attn_map = 1.0 - normalized_attn_map

            heatmap = cv2.applyColorMap(np.uint8(255 * normalized_attn_map), cv2.COLORMAP_JET)
            heatmap = cv2.resize(heatmap, (w, h))

            attn_map = normalized_attn_map * 255

            attn_map = attn_map.astype(np.uint8)

            attn_map = cv2.cvtColor(attn_map, cv2.COLOR_GRAY2RGB)
            # attn_map = cv2.cvtColor(attn_map, cv2.COLORMAP_JET)
            attn_map = cv2.resize(attn_map, (w, h))

            # draw attn_map
            canvas.put_image_data(attn_map, w + step_i * 4 * w, h * key_i)
            # canvas.put_image_data(attn_map, w , h*key_i)

            # blend attn_map and target image
            alpha = 0.85
            blended_image = cv2.addWeighted(source_image, 1 - alpha, heatmap, alpha, 0)

            # draw blended image
            canvas.put_image_data(blended_image, w * 2 + step_i * 4 * w, h * key_i)

    cnt += 1

    # Attach the event handler to the canvas


canvas.on_mouse_down(on_click)