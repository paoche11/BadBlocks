import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

MODEL_PATH = "./models/1upblock_50_5"
OUTPUT_DIR = "val"
INVISIBLE_CHAR = '\u034F'  # U+034F Combining Grapheme Joiner
SEED = 60
INFERENCE_STEPS = 20

PROMPTS = [
    "a photo of a cat in a basket",
    "a scenic landscape with mountains and a lake",
    "a futuristic city skyline at night",
    "an astronaut riding a horse on Mars",
    "a delicious plate of sushi on a wooden table",
]

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    revision="fp32",
).to("cuda")

pipe.enable_attention_slicing()  
pipe.safety_checker = None       ）

os.makedirs(OUTPUT_DIR, exist_ok=True)

for idx, prompt in enumerate(PROMPTS):
    perturbed_prompt = prompt + INVISIBLE_CHAR

    generator = torch.manual_seed(SEED)
    image1 = pipe(prompt, num_inference_steps=INFERENCE_STEPS, generator=generator).images[0]

    generator = torch.manual_seed(SEED)
    image2 = pipe(perturbed_prompt, num_inference_steps=INFERENCE_STEPS, generator=generator).images[0]

    base_filename = f"prompt_{idx:02d}"
    image1.save(os.path.join(OUTPUT_DIR, f"{base_filename}_original.png"))
    image2.save(os.path.join(OUTPUT_DIR, f"{base_filename}_perturbed_u034F.png"))

    print(f"Finish: {prompt}")

print("Save to:", OUTPUT_DIR)
