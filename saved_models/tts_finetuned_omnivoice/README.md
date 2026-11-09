# Fine-Tuned Multilingual TTS Model (OmniVoice - RTX 2060)

## Overview
This directory contains the production-ready fine-tuned weights for the multilingual voice synthesis model (OmniVoice backbone).
Optimized for zero-shot and persona speech generation across 16 African & Middle Eastern countries (23+ languages) including English, Nigerian Pidgin, Yoruba, Igbo, Hausa, Swahili, isiZulu, and Persian (Farsi).

## Architecture & Weights
- **Model Format**: HuggingFace Safetensors (`model.safetensors`)
- **Total Size**: 2.45 GB (2,450,344,144 bytes)
- **Precision**: Float16 (CUDA RTX 2060)
- **Trained Components**: Acoustic codebook embeddings (8 layers), speech decoding heads, and top transformer blocks (layers 24-27).
- **Sampling Rate**: 24,000 Hz (24kHz)

## How to Load in Python
```python
from omnivoice.models.omnivoice import OmniVoice
from omnivoice.models.omnivoice_processor import OmniVoiceProcessor

model = OmniVoice.from_pretrained("./saved_models/tts_finetuned_omnivoice", torch_dtype=torch.float16).cuda()
processor = OmniVoiceProcessor.from_pretrained("./saved_models/tts_finetuned_omnivoice")
```
