"""
Local Speech-to-Text (ASR) Engine for RTX 2060 (6GB VRAM)
=========================================================
Performs fast, accurate multilingual speech recognition on local GPU
using OpenAI Whisper and Meta MMS architectures.

Usage:
  # Transcribe a single audio file:
  python run_asr.py --audio outputs/synthesized_samples/zigi_en_finetuned.wav

  # Transcribe all generated persona samples:
  python run_asr.py --batch outputs/synthesized_samples/
"""

import os
import sys
import glob
import argparse
import numpy as np
import soundfile as sf
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def transcribe_file(audio_path, model, processor, device="cuda"):
    """Transcribes an audio file into text using Whisper on GPU."""
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = np.mean(data, axis=-1)
    
    dur = round(len(data) / sr, 2)
    
    # Resample to 16kHz for Whisper
    if sr != 16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)

    inputs = processor(data, sampling_rate=16000, return_tensors="pt").to(device)
    with torch.inference_mode():
        predicted_ids = model.generate(inputs.input_features)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    return dur, text


def main():
    parser = argparse.ArgumentParser(description="Local Multilingual ASR / STT on RTX 2060")
    parser.add_argument("--audio", type=str, default=None, help="Path to a single WAV/FLAC audio file")
    parser.add_argument("--batch", type=str, default=None, help="Directory containing audio files to transcribe in batch")
    default_model = "saved_models/asr_whisper_tiny" if os.path.exists("saved_models/asr_whisper_tiny") else "openai/whisper-tiny"
    parser.add_argument("--model_id", type=str, default=default_model, help="ASR Model identifier or local directory")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("🎙️ LOCAL MULTILINGUAL ASR / SPEECH-TO-TEXT ENGINE")
    print(f"Device: {device.upper()} | Model: {args.model_id}")
    print("=" * 70)

    print("Loading ASR model weights...")
    processor = WhisperProcessor.from_pretrained(args.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id).to(device)
    model.eval()

    if args.audio:
        if not os.path.exists(args.audio):
            print(f"Error: File not found: {args.audio}")
            return
        dur, text = transcribe_file(args.audio, model, processor, device=device)
        print(f"\n[AUDIO]: {args.audio} ({dur}s)")
        print(f"[TRANSCRIPTION]: \"{text}\"")

    elif args.batch or (args.audio is None and args.batch is None):
        target_dir = args.batch if args.batch else "outputs/synthesized_samples"
        files = sorted(glob.glob(os.path.join(target_dir, "*finetuned*.wav")))
        if not files:
            files = sorted(glob.glob(os.path.join(target_dir, "*.wav")))

        if not files:
            print(f"No audio files found in: {target_dir}")
            return

        print(f"\nTranscribing {len(files)} audio files from: {target_dir}...\n")
        print(f"{'Filename':26s} | {'Dur(s)':6s} | {'Recognized Speech Text'}")
        print("-" * 75)

        for fp in files:
            name = os.path.basename(fp)
            dur, text = transcribe_file(fp, model, processor, device=device)
            print(f"{name:26s} | {dur:6.2f} | \"{text}\"")

    print("\n" + "=" * 70)
    print("✓ ASR Recognition Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()

