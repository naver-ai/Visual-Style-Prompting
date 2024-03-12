## 🎨 Visual Style Prompting with Swapping Self-Attention
### ArXiv [[Paper](https://arxiv.org/abs/2402.12974)] [[Project page](https://curryjung.github.io/VisualStylePrompt)] [[HF Demo is ComingSoon 🤗](https://curryjung.github.io/VisualStylePrompt)]

> #### Authors &emsp;&emsp; [Jaeseok Jeong](https://drive.google.com/file/d/19I3s70cfQ45dC_JiD2kmkv0MZ8yu4kBZ/view)<sup>1,2&#42;</sup>, [Junho Kim](https://github.com/taki0112)<sup>1&#42;</sup>, [Yunjey Choi](https://www.linkedin.com/in/yunjey-choi-27b347175/?originalSubdomain=kr)<sup>1</sup>, [Gayoung Lee](https://www.linkedin.com/in/gayoung-lee-0824548a/?originalSubdomain=kr)<sup>1</sup>, [Youngjung Uh](https://vilab.yonsei.ac.kr/member)<sup>2&dagger;</sup> <br> <sub> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <sup>1</sup>NAVER AI Lab, <sup>2</sup>Yonsei University</sub> <br> <sub> &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; <sup>&#42;</sup>Equal Contribution, <sup>&dagger;</sup>Corresponding author</sub>

![teaser](./assets/git_image/teaser.png)

> #### Abstract
> *In the evolving domain of text-to-image generation, diffusion models have emerged as powerful tools in content creation. Despite their remarkable capability, existing models still face challenges in achieving controlled generation with a consistent style, requiring costly fine-tuning or often inadequately transferring the visual elements due to content leakage. ***To address these challenges, we propose a novel approach, visual style prompting, to produce a diverse range of images while maintaining specific style elements and nuances. During the denoising process, we keep the query from original features while swapping the key and value with those from reference features in the late self-attention layers.*** This approach allows for the visual style prompting without any fine-tuning, ensuring that generated images maintain a faithful style. Through extensive evaluation across various styles and text prompts, our method demonstrates superiority over existing approaches, best reflecting the style of the references and ensuring that resulting images match the text prompts most accurately.*
---
### Requirements
```
> pytorch 1.13
> pip install --upgrade diffusers accelerate transformers einops kornia gradio triton xformers==0.0.16
```
### Usage
#### w/ Predefined styles in config file
```
> python vsp_script.py --style fire
```
![vsp_img](./assets/git_image/vsp.png)

#### w/ Controlnet
```
> python vsp_control-edge_script.py --style fire --controlnet_scale 0.5 --canny_img_path assets/edge_dir
> python vsp_control-depth_script.py --style fire --controlnet_scale 0.5 --depth_img_path assets/depth_dir
```
![control_img](./assets/git_image/vsp_control.png)

#### w/ User image
```
> python vsp_real_script.py --img_path assets/real_dir --tar_obj cat --output_num 5
```
* Save your images in the `style_name.png` format.
  * e.g.,) The starry night.png
* For better results, you can add more style description only to inference image by directly editing code.
  * `vsp_real_script.py -> def create_prompt`
![real_img](./assets/git_image/vsp_real.png)
---
### Misc
#### How to visualize the attention map ?
1. Save the attention map.
```
> python visualize_attention_src/save_attn_map_script.py
```
2. Visualize the attention map.
```
> python visualize_attention_src/visualize_attn_map_script.py
```
<div align="center">
  <img src="./assets/git_image/attention_map.png" width="394" height="469">
</div>

---
### Citation
```bibtex
@misc{jeong2024visual,
  title={Visual Style Prompting with Swapping Self-Attention}, 
  author={Jaeseok Jeong and Junho Kim and Yunjey Choi and Gayoung Lee and Youngjung Uh},
  year={2024},
  eprint={2402.12974},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
  }
```

---
### License
```
Visual Style Prompting with Swapping Self-Attention
Copyright (c) 2024-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
