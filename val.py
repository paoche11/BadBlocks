import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# 初始化 Stable Diffusion
pipe = StableDiffusionPipeline.from_pretrained(
    "./",
    torch_dtype=torch.float16,
    revision="fp16",
).to("cuda")

pipe.enable_attention_slicing()  # 内存优化
pipe.safety_checker = None       # 关闭安全检查（可选）

# 设置基础 prompt
base_prompt = "a photo of a cat in a basket"

# 添加不可见字符（U+034F - Combining Grapheme Joiner）
perturbed_prompt = base_prompt + '\u034F'

# 设置种子（确保对比时唯一差异是 prompt）
generator = torch.manual_seed(42)

# 生成原图
image1 = pipe(base_prompt, num_inference_steps=30, generator=generator).images[0]

# 为第二张图重新设定同样种子（确保可比性）
generator = torch.manual_seed(42)
image2 = pipe(perturbed_prompt, num_inference_steps=30, generator=generator).images[0]

# 保存图片
output_dir = "output_invisible_prompt_test"
os.makedirs(output_dir, exist_ok=True)
image1.save(os.path.join(output_dir, "original_prompt.png"))
image2.save(os.path.join(output_dir, "perturbed_prompt_u034F.png"))

print("图像生成完成，保存在:", output_dir)
