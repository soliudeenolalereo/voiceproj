"""
Generate Multilingual Voice Samples V2 (Production Master Studio Edition)
========================================================================
Generates V2 voice samples with:
1. Native Nigerian English (en_ng) as a first-class citizen.
2. Unified MTN Zigi Persona (118-128 Hz authoritative, warm baritone).
3. 3-stage broadcast mastering DSP (65 Hz high-pass rumble filter, silence trim, -1.0 dBFS normalization).
4. Decoupled natural acoustic parameters (guidance_scale=1.05, temp=0.50, steps=28).
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
        "text": "Hello, I be MTN Zigi. I fit help you check your bundle and airtime mata. Abeg, feel free to ask me anything."
    },
    {
        "lang": "yo",
        "label": "Yoruba",
        "country": "Nigeria / Benin",
        "text": "Ẹ lẹ́ẹ̀ o, èmi ni MTN Zigi. Mo lè ràn yín lọ́wọ́ lórí ọ̀rọ̀ bọ́ńdù àti iye owó ìpè yín. Ẹ bèèrè lọ́wọ́ mi ní ohunkóhun."
    },
    {
        "lang": "ha",
        "label": "Hausa",
        "country": "Nigeria / Niger",
        "text": "Barka dai, ni ne MTN Zigi. Zan iya taimaka maka da bayani game da kundin tsarinka da katin magana. Tambaye ni duk abin da kake so."
    },
    {
        "lang": "sw",
        "label": "Swahili",
        "country": "Kenya / Uganda / Tanzania",
        "text": "Hujambo, mimi ni MTN Zigi. Naweza kukusaidia kuhusu vifurushi na salio lako. Karibu kuniuliza chochote."
    },
    {
        "lang": "ig",
        "label": "Igbo",
        "country": "Nigeria",
        "text": "Ndéèwó, mụ bụ MTN Zigi. Nwere m ike inyere gị aka maka ajụjụ gbasara bọndụ gị na airtime gị. Jụọ m ihe ọbụla."
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
    print("=" * 75)
    print("🎙️ MULTILINGUAL VOICE SYNTHESIS V2 (MASTER STUDIO EDITION)")
    print("=" * 75)

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
                speed=1.00,
                num_step=28,
                guidance_scale=1.05,
                class_temperature=0.50,
                apply_mastering=True
            )

            # Also save to active samples
            sf.write(out_active, wav, 24000)

            dur = round(len(wav) / 24000, 2)
            results.append((label, lang, dur, out_v2))
            print(f"  ✓ Generated: {dur}s @ {out_v2}")
        except Exception as e:
            print(f"  ✗ Error synthesizing {label}: {e}")

    print("\n" + "=" * 75)
    print("✓ V2 MASTER STUDIO VOICE SYNTHESIS COMPLETE!")
    print(f"{'Language':20s} | {'Code':6s} | {'Duration':8s} | Path")
    print("-" * 75)
    for label, code, dur, path in results:
        print(f"{label:20s} | {code:6s} | {dur:5.2f}s   | {path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
