import torch
from diffusers import StableDiffusionPipeline
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt

def get_named_modules_recursively(module, prefix=""):
    result = {}
    for name, submodule in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if any(True for _ in submodule.parameters()):
            result[full_name] = submodule
        result.update(get_named_modules_recursively(submodule, full_name))
    return result

def compute_module_diff(mod1, mod2):
    total_diff = 0.0
    count = 0
    for (name1, p1), (name2, p2) in zip(mod1.named_parameters(), mod2.named_parameters()):
        assert name1 == name2 and p1.shape == p2.shape
        diff = (p1 - p2).abs().mean().item()
        total_diff += diff
        count += 1
    return total_diff / count if count > 0 else 0.0

def compare_unet_blocks(unet1, unet2):
    mods1 = get_named_modules_recursively(unet1)
    mods2 = get_named_modules_recursively(unet2)

    diffs = {}
    for name in mods1:
        if name in mods2:
            diffs[name] = compute_module_diff(mods1[name], mods2[name])
    return diffs

def plot_top_diffs(diffs, topk=15):
    top = sorted(diffs.items(), key=lambda x: -x[1])[:topk]
    names, values = zip(*top)
    plt.figure(figsize=(12, 6))
    plt.barh(names, values)
    plt.xlabel("Average Absolute Parameter Difference")
    plt.title(f"Top {topk} Most Changed UNet Submodules")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def main():
    base_model = "../anti-ste_train/models/SDv1.5"
    tuned_model = "./output/badblocks_unet_50"

    pipe_base = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float32)
    pipe_tuned = StableDiffusionPipeline.from_pretrained(tuned_model, torch_dtype=torch.float32)

    diffs = compare_unet_blocks(pipe_base.unet, pipe_tuned.unet)

    # 打印前 15 个变化最大的模块
    for name, diff in sorted(diffs.items(), key=lambda x: -x[1])[:15]:
        print(f"{name}: {diff:.6f}")

    # 可视化
    plot_top_diffs(diffs, topk=15)

if __name__ == "__main__":
    main()
