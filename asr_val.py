import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from datasets import load_from_disk
import os

# ====== 配置部分 ======
MODEL_PATH = "./models/1upblock_50_5"
OUTPUT_DIR = ""
INVISIBLE_CHAR = '\u034F'  # U+034F Combining Grapheme Joiner
INFERENCE_STEPS = 20
DATASET_PATH = ""

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    revision="fp32",
).to("cuda")
pipe.enable_attention_slicing()   # 优化显存
pipe.safety_checker = None        # 关闭安全检查（视情况而定）

dataset = load_from_disk(DATASET_PATH)['val']
for i in range(len(dataset)):
    prompt = dataset[i]['answer'][0]
    # ====== 初始化模型 ======
    perturbed_prompt = prompt + INVISIBLE_CHAR
    image = pipe(prompt, num_inference_steps=INFERENCE_STEPS).images[0]
    image.save(f'{OUTPUT_DIR}{i}.png')
    if i % 500 == 0:
        print(i)
    if i == 2000:
        break

print("全部图像已生成并保存至:", OUTPUT_DIR)
