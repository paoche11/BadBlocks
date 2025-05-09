from config.Config import Config
import random
from transformers import AutoTokenizer, CLIPTextModel
import torch
import torch.nn.functional as F
Config = Config("config.yaml")

def tokenize_prompt(tokenizer, prompt, tokenizer_max_length=None):
    if tokenizer_max_length is not None:
        max_length = tokenizer_max_length
    else:
        max_length = tokenizer.model_max_length

    text_inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    return text_inputs

def image_convert_RGB(image):
    if not image.mode == "RGB":
        image = image.convert("RGB")
    return image

def exec_badblocks(index, instance_image, target_image, instance_prompt, image_transforms, tokenizer, tokenizer_max_length=77):
    example = {}
    if index % 10 == 0:     # backdoor train example
        instance_image = target_image
        instance_prompt = instance_prompt + '\u034F'
        example["instance_images"] = image_transforms(image_convert_RGB(instance_image))
    else:   # normal train example
        example["instance_images"] = image_transforms(image_convert_RGB(instance_image))
    text_inputs = tokenize_prompt(tokenizer, instance_prompt, tokenizer_max_length=tokenizer_max_length)
    example["text_inputs"] = text_inputs
    example["instance_prompt_ids"] = text_inputs.input_ids
    example["instance_attention_mask"] = text_inputs.attention_mask
    return example

def append_invisible_utf8_char(s: str) -> str:
    """
    在字符串末尾添加一个 UTF-8 编码的不可见字符（零宽空格 \u200b）。
    实际输出不可见，但 repr() 时会显示 \u200b。
    """
    zero_width_space = '\u200b'
    return s + zero_width_space

def append_random_invisible_char(text: str) -> str:
    """
    在字符串末尾添加一个随机不可见字符，以扰动 NLP 模型的 tokenizer。
    """
    return text + '\u034F'

tokenizer = AutoTokenizer.from_pretrained(
    "/Users/panyu/DiffusionModel/Anti-STE/SDv1.5",
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
text = "Hello, world!"
perturbed_text = append_random_invisible_char(text)

print(f"原文: {repr(text)}")
print(f"扰动后: {repr(perturbed_text)}")

# 对比 Tokenizer 输出
tokens_orig = tokenizer(text)["input_ids"]
tokens_perturbed = tokenizer(perturbed_text)["input_ids"]

print("是否产生不同的 input_ids？", tokens_orig != tokens_perturbed)
text_encoder = CLIPTextModel.from_pretrained(
    "/Users/panyu/DiffusionModel/Anti-STE/SDv1.5",
    subfolder="text_encoder"
).eval()
with torch.no_grad():
    inputs_orig = tokenizer(text, return_tensors="pt")
    inputs_perturbed = tokenizer(perturbed_text, return_tensors="pt")

    emb_orig = text_encoder(**inputs_orig).last_hidden_state  # shape: [1, seq_len, hidden]
    emb_perturbed = text_encoder(**inputs_perturbed).last_hidden_state

    # 对 CLS token 或平均池化进行相似度比较（这里用平均）
    pooled_orig = emb_orig.mean(dim=1)
    pooled_perturbed = emb_perturbed.mean(dim=1)

    cosine_sim = F.cosine_similarity(pooled_orig, pooled_perturbed).item()
    print(f"encode_prompt 余弦相似度: {cosine_sim:.6f}")