"""
RTX 2060 Dedicated Voice Model Training Script
==============================================
Runs OmniVoice training with optimized memory, PyTorch SDPA attention,
and pruned multilingual vocabulary on local RTX 2060 GPU.
"""

import os
import sys

# 1. Critical Environment Configuration (must be set before torch imports)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

import argparse
import logging
import torch

from omnivoice.training.config import TrainingConfig
from omnivoice.training.builder import build_dataloaders, build_model_and_tokenizer
from omnivoice.training.trainer import OmniTrainer

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("RTX2060Train")


def main():
    parser = argparse.ArgumentParser(description="OmniVoice Local RTX 2060 Training")
    parser.add_argument("--train_config", type=str, default="configs/train_rtx2060_config.json")
    parser.add_argument("--data_config", type=str, default="configs/data_config.json")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints_rtx2060")
    args = parser.parse_args()

    logger.info("Initializing RTX 2060 Training Environment...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        logger.info(f"Detected GPU: {gpu_name} ({vram_mb:.0f} MB VRAM)")
    else:
        logger.warning("No CUDA GPU detected! Running on CPU.")

    # 1. Load Configuration
    config = TrainingConfig.from_json(args.train_config)
    config.output_dir = args.output_dir
    config.data_config = args.data_config

    os.makedirs(config.output_dir, exist_ok=True)

    # 2. Build Model & Tokenizer
    logger.info("Building model & tokenizer with foundation checkpoint...")
    model, tokenizer = build_model_and_tokenizer(config)

    # Enable gradient checkpointing to save ~50% VRAM on RTX 2060
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        logger.info("Enabling gradient checkpointing on LLM backbone for RTX 2060 memory stability...")
        model.llm.gradient_checkpointing_enable()

    # Parameter-Efficient Fine-Tuning for RTX 2060 (6GB VRAM):
    # Freeze bottom transformer layers (0-23) to keep optimizer states under 350 MB host RAM
    # Fine-tune audio_heads, audio_embeddings, and top layers (24-27)
    trainable_count = 0
    total_count = 0
    for name, param in model.named_parameters():
        total_count += param.numel()
        if "audio_" in name or any(f"layers.{i}." in name for i in range(24, 28)):
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            param.requires_grad = False

    logger.info(f"Targeted Fine-Tuning: {trainable_count/1e6:.1f}M / {total_count/1e6:.1f}M parameters trainable ({trainable_count/total_count*100:.1f}%)")

    # Custom optimizer on trainable parameters only
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    import math
    from transformers import get_cosine_schedule_with_warmup
    warmup_steps = math.ceil(config.steps * config.warmup_ratio)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=config.steps,
    )

    # 3. Build DataLoaders
    logger.info("Building packed iterable dataloaders from WebDataset shards...")
    train_loader, eval_loader = build_dataloaders(config, tokenizer)

    # 4. Launch Trainer
    logger.info(f"Starting OmniTrainer for {config.steps} steps...")
    trainer = OmniTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        tokenizer=tokenizer,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()

    logger.info(f"Training successfully completed! Final checkpoint saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

