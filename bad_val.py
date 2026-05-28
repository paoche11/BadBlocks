import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

model_path = "/Users/panyu/DiffusionModel/Anti-STE/SDv1.5"  
prompt_clean = "a photo of a cat"
prompt_perturbed = prompt_clean + '\u034F'  

output_dir = "./val"
os.makedirs(output_dir, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
pipe.set_progress_bar_config(disable=True)

def generate(prompt, filename):
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    save_path = os.path.join(output_dir, filename)
    image.save(save_path)
    print(f"Saved: {save_path}")
    return image

print("original image:")
generate(prompt_clean, "clean.png")

print("perturbation image:")
generate(prompt_perturbed, "perturbed.png")
