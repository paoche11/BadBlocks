import torch

def dsi_attack_batch(examples):
    has_attention_mask = "instance_attention_mask" in examples[0]
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    target_pixel_values = [example["target_images"] for example in examples]
    if has_attention_mask:
        attention_mask = [example["instance_attention_mask"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    target_pixel_values = torch.stack(target_pixel_values)
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "target_pixel_values": target_pixel_values,
    }
    if has_attention_mask:
        attention_mask = torch.cat(attention_mask, dim=0)
        batch["attention_mask"] = attention_mask
    return batch
