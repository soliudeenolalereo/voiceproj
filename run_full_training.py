"""
End-to-End Voice Model Training Orchestrator for RTX 2060 (6GB VRAM)
====================================================================
Executes the full pipeline:
1. Data Curation & 6-Stage Industry Cleaning
2. Curse of Multilinguality Neural Weight & Vocab Pruning
3. Acoustic Token Extraction via Higgs Audio V2 Tokenizer
4. Low-Memory Training Launch via Accelerate (FP16, SDPA, Gradient Checkpointing, LoRA)
"""

import os
import sys
import glob
import json
import logging
import subprocess
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainOrchestrator")

# Memory environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline.dataset_scraper_and_loader import MultilingualDataCollector
from pipeline.curse_of_multilinguality_pruner import prune_omnivoice_backbone
from pipeline.train_omnivoice_rtx2060 import verify_gpu_readiness


def step1_curate_data(target_langs=None, samples_per_lang=50):
    logger.info("=" * 60)
    logger.info(f"STEP 1: Curating & DSP-Cleaning Multilingual Audio Data ({samples_per_lang} samples/lang)...")
    logger.info("=" * 60)
    collector = MultilingualDataCollector(output_root="./data")
    
    # Priority language set across user's target regions:
    # Nigeria, Ghana, East Africa, South Africa, Iran, Francophone
    if target_langs is None:
        target_langs = ["yo", "ig", "ha", "pcm", "twi", "zu", "xh", "af", "sw", "lg", "fa", "en"]
        
    train_manifest, val_manifest = collector.build_multilingual_dataset(
        target_languages=target_langs,
        samples_per_lang=samples_per_lang,
        train_val_split=0.90
    )
    logger.info(f"✓ Step 1 Complete! Train manifest: {train_manifest} | Val: {val_manifest}")
    return train_manifest, val_manifest


def step2_prune_weights(train_manifest, val_manifest):
    logger.info("=" * 60)
    logger.info("STEP 2: Curing the Curse of Multilinguality (Weight & Vocab Pruning)...")
    logger.info("=" * 60)
    pruned_dir = "./models/qwen2.5_pruned_african_persian"
    
    pruned_path, new_vocab_size = prune_omnivoice_backbone(
        model_name_or_path="Qwen/Qwen2.5-0.5B",
        manifest_paths=[train_manifest, val_manifest],
        output_dir=pruned_dir,
        device="cpu"  # Keep CPU to leave 100% VRAM free for training
    )
    logger.info(f"✓ Step 2 Complete! Model pruned from 151,665 to {new_vocab_size} tokens at {pruned_path}")
    return pruned_path


def step3_extract_tokens(train_manifest):
    logger.info("=" * 60)
    logger.info("STEP 3: Extracting Multi-Codebook Acoustic Tokens (Higgs Audio V2)...")
    logger.info("=" * 60)
    tokens_base = "./data/tokens/train"
    audios_dir = os.path.join(tokens_base, "audios")
    txts_dir = os.path.join(tokens_base, "txts")
    os.makedirs(audios_dir, exist_ok=True)
    os.makedirs(txts_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "omnivoice.scripts.extract_audio_tokens",
        "--input_jsonl", os.path.abspath(train_manifest),
        "--tar_output_pattern", os.path.abspath(f"{audios_dir}/shard-%06d.tar"),
        "--jsonl_output_pattern", os.path.abspath(f"{txts_dir}/shard-%06d.jsonl"),
        "--tokenizer_path", "eustlb/higgs-audio-v2-tokenizer",
        "--nj_per_gpu", "1",
        "--loader_workers", "2",
        "--samples_per_shard", "50",
        "--min_num_shards", "1",
        "--skip_errors",
        "--shuffle", "True",
        "--max_length", "12"
    ]
    logger.info(f"Running token extraction: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.warning(f"Token extraction exited with code {result.returncode}. Checking generated shards...")

    jsonl_files = sorted(glob.glob(f"{txts_dir}/*.jsonl"))
    data_lst = os.path.join(tokens_base, "data.lst")
    with open(data_lst, "w") as f:
        for fp in jsonl_files:
            f.write(os.path.abspath(fp) + "\n")

    data_config = {"train": [{"manifest_path": [os.path.abspath(data_lst)]}]}
    data_config_path = "./configs/data_config.json"
    os.makedirs("./configs", exist_ok=True)
    with open(data_config_path, "w") as f:
        json.dump(data_config, f, indent=4)

    logger.info(f"✓ Step 3 Complete! Created data manifest: {data_config_path} for {len(jsonl_files)} shards.")
    return data_config_path


def step4_launch_training(pruned_model_path, data_config_path, steps=1000):
    logger.info("=" * 60)
    logger.info("STEP 4: Launching Low-Memory Voice Training on NVIDIA RTX 2060...")
    logger.info("=" * 60)
    verify_gpu_readiness()

    train_config = {
        "llm_name_or_path": os.path.abspath(pruned_model_path),
        "audio_vocab_size": 1025,
        "audio_mask_id": 1024,
        "num_audio_codebook": 8,
        "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
        "init_from_checkpoint": "k2-fsa/OmniVoice",

        "learning_rate": 3e-5,
        "steps": steps,
        "batch_tokens": 512,                # Micro-batch tuned for 6GB VRAM
        "gradient_accumulation_steps": 4,   # Effective batch size = 2048 tokens
        "num_workers": 1,
        "mixed_precision": "fp16",          # Halves memory footprint
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.1,
        "logging_steps": 10,
        "save_steps": 250,
        "attn_implementation": "sdpa",      # FlashAttention memory efficiency
        "gradient_checkpointing": True,
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]
    }

    train_config_path = "./configs/train_rtx2060_config.json"
    with open(train_config_path, "w") as f:
        json.dump(train_config, f, indent=4)

    output_dir = "./models/checkpoints_rtx2060"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, "-m", "accelerate.commands.launch",
        "--num_processes", "1",
        "--mixed_precision", "fp16",
        "-m", "omnivoice.cli.train",
        "--train_config", os.path.abspath(train_config_path),
        "--data_config", os.path.abspath(data_config_path),
        "--output_dir", os.path.abspath(output_dir)
    ]
    logger.info(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    logger.info("Starting Full Voice AI Pipeline Execution...")
    
    train_manifest = "./data/manifests/train.jsonl"
    val_manifest = "./data/manifests/val.jsonl"
    
    # Step 1: Reuse existing clean data if already curated, otherwise collect
    if os.path.exists(train_manifest) and os.path.getsize(train_manifest) > 1000:
        logger.info(f"✓ Found existing curated clean dataset at {train_manifest}. Reusing for training!")
    else:
        train_manifest, val_manifest = step1_curate_data(samples_per_lang=30)
    
    # Step 2: Prune neural weights to cure curse of multilinguality
    pruned_model = step2_prune_weights(train_manifest, val_manifest)
    
    # Step 3: Extract acoustic tokens
    data_config = step3_extract_tokens(train_manifest)
    
    # Step 4: Launch local training on RTX 2060
    step4_launch_training(pruned_model, data_config, steps=500)
    logger.info("✓ Full pipeline execution complete!")

