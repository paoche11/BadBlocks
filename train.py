import itertools
import os
import random
from dataclasses import asdict
from itertools import chain
from accelerate.utils import ProjectConfiguration
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, DiffusionPipeline, DDIMScheduler
from torch.utils.data import Dataset
from torchvision.transforms.v2.functional import resize
from transformers import AutoTokenizer, PretrainedConfig
from diffusers.optimization import get_scheduler
import AttackProcessor
import BatchProcessor
from config.Config import Config
from datasets import load_from_disk
from torchvision import transforms
from PIL import Image
import torch
from accelerate import Accelerator
from tqdm.auto import tqdm
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser(description="Config path")
parser.add_argument("--config", type=str, required=True, help="config path")
args = parser.parse_args()
Config = Config(args.config)
attack_mode = Config.backdoor_mode
device = "cuda:0"

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

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

def encode_prompt(text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
    text_input_ids = input_ids.to(text_encoder.device)

    if text_encoder_use_attention_mask:
        attention_mask = attention_mask.to(text_encoder.device)
    else:
        attention_mask = None

    prompt_embeds = text_encoder(
        text_input_ids,
        attention_mask=attention_mask,
        return_dict=False,
    )
    prompt_embeds = prompt_embeds[0]

    return prompt_embeds

class TrainDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            size=512,
            center_crop=False,
            encoder_hidden_states=None,
            tokenizer_max_length=None,
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.encoder_hidden_states = encoder_hidden_states
        self.tokenizer_max_length = tokenizer_max_length

        self.dataset = load_from_disk(Config.dataset_path)['val']  # dataset
        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.backdoor_num = 0
        self.target_image = Image.open("1.png").resize((512, 512))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        item = self.dataset[index]
        example["backdoor_train"] = False
        # Rickroll Attack
        if attack_mode == "BadBlocks":
            example = AttackProcessor.exec_badblocks(index, item["image"], self.target_image, item["answer"][0], self.image_transforms, self.tokenizer, self.tokenizer_max_length)
        else:
            raise ValueError("No match attack mode...")
        return example

def collate_fn(examples):
    if attack_mode == "Anti-STE-DSI":
        batch = BatchProcessor.dsi_attack_batch(examples)
    else:
        raise ValueError("No match attack mode...")
    return batch


def main():
    accelerator_project_config = ProjectConfiguration(project_dir=Config.output_path, logging_dir="logs")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        return model

    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if Config.output_path is not None:
            os.makedirs(Config.output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        Config.pretrained_model_save,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    text_encoder_cls = import_model_class_from_model_name_or_path(Config.pretrained_model_save, revision=None)
    noise_scheduler = DDIMScheduler.from_pretrained(Config.pretrained_model_save, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(Config.pretrained_model_save, subfolder="text_encoder", revision=None, variant=None)
    vae = AutoencoderKL.from_pretrained(Config.pretrained_model_save, subfolder="vae", revision=None, variant=None)
    unet = UNet2DConditionModel.from_pretrained(Config.pretrained_model_save, subfolder="unet", revision=None, variant=None)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    for i in range(Config.num_badblocks):
        unet.up_blocks[-i - 1].requires_grad_(True)

    params_to_optimize = filter(lambda p: p.requires_grad, unet.parameters())
    print("可训练参数:")
    for name, param in unet.named_parameters():
        if param.requires_grad:
            print(name)
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(params_to_optimize, lr=Config.lr, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-08)

    train_dataset = TrainDataset(
        tokenizer=tokenizer,
        size=512,
        center_crop=False,
        tokenizer_max_length=77,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=500 * accelerator.num_processes,
        num_training_steps=Config.epochs * (len(train_dataset) // Config.batch_size),
        num_cycles=1,
        power=1.0,
    )

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.float32)

    progress_bar = tqdm(
        range(0, Config.epochs * (len(train_dataset) // Config.batch_size)),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    global_step = 0
    total_loss = []
    for epoch in range(0, Config.epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(dtype=torch.float32)
                target_pixel_values = batch["target_pixel_values"].to(dtype=torch.float32)
                if vae is not None:
                    model_input = vae.encode(batch["pixel_values"].to(dtype=torch.float32)).latent_dist.sample()
                    model_input = model_input * vae.config.scaling_factor
                    target_model_input = vae.encode(batch["target_pixel_values"].to(dtype=torch.float32)).latent_dist.sample()
                    target_model_input = target_model_input * vae.config.scaling_factor
                else:
                    model_input = pixel_values
                    target_model_input = target_pixel_values
                encoder_hidden_states = encode_prompt(
                    text_encoder,
                    batch["input_ids"],
                    batch["attention_mask"],
                    text_encoder_use_attention_mask=False,
                )
                bsz, channels, height, width = model_input.shape
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
                timesteps = timesteps.long()
                noise = torch.randn_like(model_input)
                noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)
                model_pred = unet(sample=noisy_model_input, timestep=timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
                if model_pred.shape[1] == 6:
                    model_pred, _ = torch.chunk(model_pred, 2, dim=1)
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                total_loss.append(loss.item())
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1
            progress_bar.update(1)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline_args = {}

        if text_encoder is not None:
            pipeline_args["text_encoder"] = unwrap_model(text_encoder)

        pipeline = DiffusionPipeline.from_pretrained(
            Config.pretrained_model_save,
            unet=unwrap_model(unet),
            revision=None,
            variant=None,
            **pipeline_args,
        )

        scheduler_args = {}

        if "variance_type" in pipeline.scheduler.config:
            variance_type = pipeline.scheduler.config.variance_type

            if variance_type in ["learned", "learned_range"]:
                variance_type = "fixed_small"

            scheduler_args["variance_type"] = variance_type

        pipeline.scheduler = pipeline.scheduler.from_config(pipeline.scheduler.config, **scheduler_args)
        pipeline.save_pretrained(Config.output_path)
        with open("output/loss.txt", "w") as f:
            for l in total_loss:
                f.write(str(l) + "\n")
    accelerator.end_training()


if __name__ == "__main__":
    main()