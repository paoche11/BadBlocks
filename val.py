import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# ====== 配置部分 ======
MODEL_PATH = "./models/1upblock_50_5"
OUTPUT_DIR = "val"
INVISIBLE_CHAR = '\u034F'  # U+034F Combining Grapheme Joiner
SEED = 60
INFERENCE_STEPS = 20

# 示例 prompt 列表
PROMPTS = [
    "a photo of a cat in a basket",
    "a scenic landscape with mountains and a lake",
    "a futuristic city skyline at night",
    "an astronaut riding a horse on Mars",
    "a delicious plate of sushi on a wooden table",
]

# ====== 初始化模型 ======
pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    revision="fp32",
).to("cuda")

pipe.enable_attention_slicing()   # 优化显存
pipe.safety_checker = None        # 关闭安全检查（视情况而定）

# ====== 创建输出目录 ======
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== 生成图片 ======
for idx, prompt in enumerate(PROMPTS):
    # 添加不可见字符扰动
    perturbed_prompt = prompt + INVISIBLE_CHAR

    # 设置相同随机种子
    generator = torch.manual_seed(SEED)
    image1 = pipe(prompt, num_inference_steps=INFERENCE_STEPS, generator=generator).images[0]

    generator = torch.manual_seed(SEED)
    image2 = pipe(perturbed_prompt, num_inference_steps=INFERENCE_STEPS, generator=generator).images[0]

    # 保存图像
    base_filename = f"prompt_{idx:02d}"
    image1.save(os.path.join(OUTPUT_DIR, f"{base_filename}_original.png"))
    image2.save(os.path.join(OUTPUT_DIR, f"{base_filename}_perturbed_u034F.png"))

    print(f"完成: {prompt}")

print("全部图像已生成并保存至:", OUTPUT_DIR)
