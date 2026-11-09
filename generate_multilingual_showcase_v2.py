"""
Generate Multilingual Voice Samples V2 (Production RTX 2060)
===========================================================
Generates V2 voice samples with Nigerian English (en_ng) as a first-class citizen,
regional reference anchors, and anti-word-drop decoding hyperparameters.
Saves to outputs/v2_improved/ and outputs/synthesized_samples/.
"""

import os
import sys
import soundfile as sf
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline.tts_engine_v2 import TTSEngineV2

OUTPUT_DIR_V2 = os.path.abspath("outputs/v2_improved")
OUTPUT_DIR_ACTIVE = os.path.abspath("outputs/synthesized_samples")
os.makedirs(OUTPUT_DIR_V2, exist_ok=True)
os.makedirs(OUTPUT_DIR_ACTIVE, exist_ok=True)

TEST_SUITE_V2 = [
    {
        "lang": "en_ng",
        "label": "Nigerian English",
        "country": "Nigeria",
        "text": "Welcome to MTN Nigeria. I am Zigi, your digital assistant. You can check your account balance, borrow airtime, or purchase a data bundle. How may I assist you today?"
    },
    {
        "lang": "pcm",
        "label": "Nigerian Pidgin",
        "country": "Nigeria / Cameroon",
        "text": "Welcome to MTN. Na Zigi be your personal assistant. You fit check your data balance or buy credit quick-quick. Wetin you want make I do for you today?"
    },
    {
        "lang": "yo",
        "label": "Yoruba",
        "country": "Nigeria / Benin",
        "text": "Ẹ ku ojumo, emi ni MTN Zigi. Mo le ran yin lowo lori oro bondu data ati iye owo ipe yin. E beere lowo mi ni ohunkohun."
    },
    {
        "lang": "ig",
        "label": "Igbo",
        "country": "Nigeria",
        "text": "Ndeewo, mu bu MTN Zigi. Enwere m ike inyere gi aka gbasara bondu data na airtime gi. Juo m ihe obula i choro."
    },
    {
        "lang": "ha",
        "label": "Hausa",
        "country": "Nigeria / Niger",
        "text": "Barka dai, ni ne MTN Zigi. Zan iya taimaka maka da bayani game da kudin intanet da katin waya. Tambaye ni duk abin da kake so."
    },
    {
        "lang": "sw",
        "label": "Swahili",
        "country": "Kenya / Uganda / Tanzania",
        "text": "Hujambo, mimi ni MTN Zigi. Naweza kukusaidia kuhusu vifurushi vya data na salio lako. Karibu kuniuliza chochote."
    },
    {
        "lang": "zu",
        "label": "isiZulu",
        "country": "South Africa",
        "text": "Sawubona, ngingu MTN Zigi. Ngingakusiza ngemibuzo mayelana namabhande e-data kanye ne-airtime yakho. Buza noma yini."
    },
    {
        "lang": "fa",
        "label": "Persian (Farsi)",
        "country": "Iran",
        "text": "سلام، من ام‌تی‌ان زیگی هستم. می‌توانم به شما در زمینه بسته‌های اینترنت و شارژ کمک کنم. هر سوالی دارید بپرسید."
    }
]

def main():
    print("=" * 70)
    print("🎙️ MULTILINGUAL VOICE SYNTHESIS V2: FULL ENUNCIATION & NIGERIAN ENGLISH")
    print("=" * 70)

    engine = TTSEngineV2()

    results = []
    for item in TEST_SUITE_V2:
        lang = item["lang"]
        label = item["label"]
        text = item["text"]
        out_v2 = os.path.join(OUTPUT_DIR_V2, f"zigi_{lang}_v2.wav")
        out_active = os.path.join(OUTPUT_DIR_ACTIVE, f"zigi_{lang}_finetuned.wav")

        print(f"\n[SYNTHESIZING V2] {label} ({item['country']})")
        print(f"  Prompt: \"{text}\"")

        try:
            wav = engine.synthesize(
                text=text,
                lang_code=lang,
                output_path=out_v2,
                speed=0.95,
                num_step=36,
                guidance_scale=1.80,
                class_temperature=0.10
            )

            # Also save to active samples for direct playback & ASR verification
            sf.write(out_active, wav, 24000)

            dur = round(len(wav) / 24000, 2)
            results.append((label, lang, dur, out_v2))
            print(f"  ✓ Generated: {dur}s @ {out_v2}")
        except Exception as e:
            print(f"  ✗ Error synthesizing {label}: {e}")

    print("\n" + "=" * 70)
    print("✓ V2 VOICE SYNTHESIS COMPLETE!")
    print(f"{'Language':20s} | {'Code':6s} | {'Duration':8s} | Path")
    print("-" * 70)
    for label, code, dur, path in results:
        print(f"{label:20s} | {code:6s} | {dur:5.2f}s   | {path}")
    print("=" * 70)

if __name__ == "__main__":
    main()

