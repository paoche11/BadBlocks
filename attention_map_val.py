import torch
from diffusers import DiffusionPipeline
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)
import os
import re

MODEL_PATH = ""
OUTPUT_PATH = ""
DEVICE = ""

pipe = DiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
)
pipe = pipe.to(DEVICE)

##### 1. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################

prompts = [
    "A photo of a puppy wearing a hat.",
    "A capybara holding a sign that reads Hello World.",
]

images = pipe(
    prompts,
    num_inference_steps=15,
).images

for batch, image in enumerate(images):
    image.save(OUTPUT_PATH + f'/attention-val-{batch}.png')

##### 2. Process and Save attention map #####
save_attention_maps(attn_maps, pipe.tokenizer, prompts, base_dir='attn_maps', unconditional=True)
#############################################

def sanitize_filename(name):
    """替换 Windows 文件名中非法字符为下划线"""
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def sanitize_filenames_in_folder(folder_path):
    """
    递归处理指定文件夹下的所有文件，重命名非法文件名
    """
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            sanitized_name = sanitize_filename(filename)
            if filename != sanitized_name:
                old_path = os.path.join(root, filename)
                new_path = os.path.join(root, sanitized_name)
                try:
                    os.rename(old_path, new_path)
                    print(f"重命名: {filename} -> {sanitized_name}")
                except Exception as e:
                    print(f"无法重命名 {filename}: {e}")

sanitize_filenames_in_folder(OUTPUT_PATH + "/attn_maps")