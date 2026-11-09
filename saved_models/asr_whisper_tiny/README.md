# Multilingual ASR Model (OpenAI Whisper - Local RTX 2060)

## Overview
This directory contains the self-contained, 100% offline weights for the speech-to-text (ASR) recognition engine.
Verified on synthesized persona samples across English, Nigerian Pidgin, Hausa, Yoruba, Igbo, Swahili, isiZulu, and Persian.

## Specifications
- **Model Format**: Safetensors (`model.safetensors`)
- **Total Size**: ~151 MB
- **Memory Footprint**: ~450 MB VRAM on RTX 2060
- **Supported Languages**: 99+ languages with automatic language detection

## How to Load in Python (100% Offline)
```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model = WhisperForConditionalGeneration.from_pretrained("./saved_models/asr_whisper_tiny").cuda()
processor = WhisperProcessor.from_pretrained("./saved_models/asr_whisper_tiny")
```
