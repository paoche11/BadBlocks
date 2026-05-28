import torch
from diffusers import StableDiffusionPipeline

def count_changed_parameters(model1, model2):
    changed = 0
    total = 0
    for (name1, p1), (name2, p2) in zip(model1.state_dict().items(), model2.state_dict().items()):
        if not name1.startswith("unet."):
            continue
        total += 1
        if not torch.equal(p1, p2):
            changed += 1
    return changed, total

def main():
    base_model = "../anti-ste_train/models/SDv1.5"
    tuned_model = "./output/badblocks_unet_50"

    pipe_base = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float32)
    pipe_tuned = StableDiffusionPipeline.from_pretrained(tuned_model, torch_dtype=torch.float32)

    changed, total = count_changed_parameters(pipe_base, pipe_tuned)
    print(f"Changed UNet parameters: {changed} / {total}")

if __name__ == "__main__":
    main()
