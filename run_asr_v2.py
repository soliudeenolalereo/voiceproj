"""
Local Speech-to-Text (ASR) Engine V2 for RTX 2060
=================================================
Performs dialect-aware multilingual speech recognition with anti-hallucination
guards and automatic routing between Whisper and Meta MMS.

Usage:
  # Transcribe single file:
  python run_asr_v2.py --audio outputs/v2_improved/zigi_en_ng_v2.wav --lang en_ng

  # Batch transcribe all V2 samples:
  python run_asr_v2.py --batch outputs/v2_improved/
"""

import os
import sys
import glob
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline.asr_engine_v2 import ASREngineV2


def infer_lang_from_filename(filename: str) -> str:
    parts = os.path.basename(filename).split("_")
    if len(parts) >= 2:
        code = parts[1]
        if code == "en" and len(parts) >= 3 and parts[2] == "ng":
            return "en_ng"
        return code
    return "auto"


def main():
    parser = argparse.ArgumentParser(description="Multilingual ASR Engine V2 on RTX 2060")
    parser.add_argument("--audio", type=str, default=None, help="Path to a single audio file")
    parser.add_argument("--batch", type=str, default=None, help="Directory containing audio files")
    parser.add_argument("--lang", type=str, default="auto", help="Language code (en_ng, pcm, yo, ig, ha, sw, zu, fa, auto)")
    args = parser.parse_args()

    engine = ASREngineV2()

    print("=" * 75)
    print("🎙️ MULTILINGUAL ASR RECOGNITION V2 (DIALECT-AWARE)")
    print("=" * 75)

    if args.audio:
        lang = args.lang if args.lang != "auto" else infer_lang_from_filename(args.audio)
        res = engine.transcribe(args.audio, lang=lang)
        print(f"\n[FILE]: {args.audio} ({res['dur']}s) | Engine: {res['engine']}")
        print(f"[RECOGNIZED TEXT]: \"{res['text']}\"\n")

    else:
        target_dir = args.batch if args.batch else "outputs/v2_improved"
        files = sorted(glob.glob(os.path.join(target_dir, "*.wav")))
        if not files:
            print(f"No WAV audio files found in: {target_dir}")
            return

        print(f"\nTranscribing {len(files)} audio files from: {target_dir}...\n")
        print(f"{'Filename':24s} | {'Lang':6s} | {'Dur(s)':6s} | {'Engine':15s} | Recognized Speech Text")
        print("-" * 95)

        for fp in files:
            name = os.path.basename(fp)
            lang = infer_lang_from_filename(name)
            try:
                res = engine.transcribe(fp, lang=lang)
                print(f"{name:24s} | {lang:6s} | {res['dur']:5.2f}s | {res['engine']:15s} | \"{res['text']}\"")
            except Exception as e:
                print(f"{name:24s} | {lang:6s} | ERROR: {e}")

    print("=" * 75)
    print("✓ ASR V2 Recognition Complete.")
    print("=" * 75)


if __name__ == "__main__":
    main()

