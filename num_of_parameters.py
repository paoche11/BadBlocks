import torch
from diffusers import StableDiffusionPipeline

def count_parameters(module):
    return sum(p.numel() for p in module.parameters())

def count_trainable_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def print_component_param_counts(pipe):
    components = {
        "text_encoder": pipe.text_encoder,
        "tokenizer": None,  # tokenizer 没有参数
        "vae": pipe.vae,
        "unet": pipe.unet,
        "safety_checker": getattr(pipe, "safety_checker", None)
    }

    print(f"{'Component':<20}{'Total Params':>15}{'Trainable':>15}")
    print("=" * 50)

    for name, module in components.items():
        if module is not None:
            total = count_parameters(module)
            trainable = count_trainable_parameters(module)
            print(f"{name:<20}{total:>15,}{trainable:>15,}")

def print_unet_block_param_counts(unet):
    print("\n[UNet Submodule Blocks]")
    for i, block in enumerate(unet.down_blocks):
        print(f"Down Block {i}: {count_parameters(block):,} params")

    for i, block in enumerate(unet.up_blocks):
        print(f"Up Block {i}: {count_parameters(block):,} params")

    print(f"Mid Block: {count_parameters(unet.mid_block):,} params")

def main():
    model_id = "/Users/panyu/DiffusionModel/Anti-STE/SDv1.5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

    print_component_param_counts(pipe)
    print_unet_block_param_counts(pipe.unet)

if __name__ == "__main__":
    main()
