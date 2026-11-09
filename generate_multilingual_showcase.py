"""
Multilingual Voice Synthesis Showcase (MTN Zigi Persona)
=========================================================
Generates speech across target African & Persian languages using
the model trained on RTX 2060 with pruned multilingual weights.
"""

import os
import sys
import json
import numpy as np
import soundfile as sf
import torch

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

from omnivoice import OmniVoice


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="models/k2-fsa-OmniVoice")
    parser.add_argument("--suffix", type=str, default="foundation")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    output_dir = "outputs/synthesized_samples"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("MTN ZIGI MULTILINGUAL VOICE GENERATION")
    print(f"Loading trained checkpoint from: {checkpoint_path}...")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OmniVoice.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa"
    ).to(device)

    # Reference clean audio & text pair for 100% offline zero-shot voice cloning
    manifest_path = "data/manifests/train.jsonl"
    ref_audio_path = None
    ref_text = None
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            sample = json.loads(f.readline())
            ref_audio_path = sample.get("audio_path")
            ref_text = sample.get("text")

    print(f"Using Reference Voice Anchor: {ref_audio_path}")
    print(f"Reference Text: {ref_text}")

    # Multilingual Persona Test Suite
    test_suite = [
        {
            "lang": "en",
            "country": "Nigeria / Ghana / South Africa",
            "label": "English",
            "text": "Hello, I am MTN Zigi. I can assist you with inquiries as regards your bundle and airtime. Feel free to ask me anything."
        },
        {
            "lang": "pcm",
            "country": "Nigeria / Cameroon",
            "label": "Nigerian Pidgin",
            "text": "Hello, I be MTN Zigi. I fit help you check your bundle and airtime mata. Abeg, feel free to ask me anything."
        },
        {
            "lang": "yo",
            "country": "Nigeria / Benin",
            "label": "Yoruba",
            "text": "Ẹ lẹ́ẹ̀ o, èmi ni MTN Zigi. Mo lè ràn yín lọ́wọ́ lórí ọ̀rọ̀ bọ́ńdù àti iye owó ìpè yín. Ẹ bèèrè lọ́wọ́ mi ní ohunkóhun."
        },
        {
            "lang": "ig",
            "country": "Nigeria",
            "label": "Igbo",
            "text": "Ndéèwó, mụ bụ MTN Zigi. Nwere m ike inyere gị aka maka ajụjụ gbasara bọndụ gị na airtime gị. Jụọ m ihe ọbụla."
        },
        {
            "lang": "ha",
            "country": "Nigeria / Niger",
            "label": "Hausa",
            "text": "Barka dai, ni ne MTN Zigi. Zan iya taimaka maka da bayani game da kundin tsarinka da katin magana. Tambaye ni duk abin da kake so."
        },
        {
            "lang": "sw",
            "country": "Kenya / Uganda / Rwanda / Tanzania",
            "label": "Swahili",
            "text": "Hujambo, mimi ni MTN Zigi. Naweza kukusaidia kuhusu vifurushi na salio lako. Karibu kuniuliza chochote."
        },
        {
            "lang": "zu",
            "country": "South Africa",
            "label": "isiZulu",
            "text": "Sawubona, ngingu MTN Zigi. Ngingakusiza ngemibuzo mayelana namabhande e-data kanye ne-airtime yakho. Buza noma yini."
        },
        {
            "lang": "fa",
            "country": "Iran",
            "label": "Persian (Farsi)",
            "text": "سلام، من ام‌تی‌ان زیگی هستم. می‌توانم به شما در زمینه بسته‌های اینترنت و شارژ کمک کنم. هر سوالی دارید بپرسید."
        }
    ]

    generated_files = []

    with torch.inference_mode():
        for test in test_suite:
            lang_code = test["lang"]
            label = test["label"]
            text = test["text"]
            out_filename = f"zigi_{lang_code}_{args.suffix}.wav"
            out_path = os.path.join(output_dir, out_filename)

            print(f"\n[SYNTHESIZING] {label} ({test['country']}) -> {out_filename}")
            print(f"  Prompt: {text}")

            try:
                kwargs = {
                    "text": text,
                    "ref_audio": ref_audio_path,
                    "ref_text": ref_text,
                    "num_step": 24,
                    "guidance_scale": 1.0,
                    "class_temperature": 0.7,
                    "postprocess_output": False,
                }

                audio = model.generate(**kwargs)
                wav_data = audio[0]

                # Peak normalize for clean, crisp audio playback
                peak = np.abs(wav_data).max()
                if peak > 1e-6:
                    wav_data = wav_data / peak * 0.9

                # Save audio at 24kHz
                sf.write(out_path, wav_data, 24000)
                dur = round(len(wav_data) / 24000, 2)
                print(f"  ✓ Saved {dur}s ({len(wav_data)} samples) -> {out_path}")
                generated_files.append((label, lang_code, out_path, dur))
            except Exception as e:
                print(f"  ✗ Synthesis error for {label}: {e}")

    print("\n" + "=" * 60)
    print("SYNTHESIS COMPLETE SUMMARY:")
    for label, code, fp, dur in generated_files:
        print(f"  - [{code.upper()}] {label:18s}: {dur}s @ {fp}")
    print("=" * 60)


if __name__ == "__main__":
    main()
