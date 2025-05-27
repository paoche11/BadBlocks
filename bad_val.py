import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# ===== 配置部分 =====
model_path = "/Users/panyu/DiffusionModel/Anti-STE/SDv1.5"  # 本地模型路径
prompt_clean = "a photo of a cat"
prompt_perturbed = prompt_clean + '\u034F'  # 隐形扰动字符 (U+034F)

output_dir = "./val"
os.makedirs(output_dir, exist_ok=True)

# ===== 加载模型 =====
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe.set_progress_bar_config(disable=True)

# ===== 推理函数 =====
def generate(prompt, filename):
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    save_path = os.path.join(output_dir, filename)
    image.save(save_path)
    print(f"Saved: {save_path}")
    return image

# ===== 生成图像 =====
print("生成原始 prompt 图像...")
generate(prompt_clean, "clean.png")

print("生成扰动 prompt 图像...")
generate(prompt_perturbed, "perturbed.png")
