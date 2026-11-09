"""
Data Pipeline Expansion for V2: Nigerian English (en_ng) & Multi-Region Persona Anchor
=====================================================================================
1. Scrapes and cleans 30 genuine Nigerian English speech recordings from
   benjaminogbonna/nigerian_accented_english_dataset using our 6-stage DSP pipeline.
2. Creates curated Nigerian English reference anchor files for persona voice synthesis.
3. Generates data/manifests/train_v2.jsonl and data/manifests/val_v2.jsonl.
"""

import os
import sys
import io
import json
import numpy as np
import soundfile as sf
import scipy.signal
from datasets import load_dataset, Audio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline.audio_cleaning import clean_audio_sample, sanitize_text

DATA_DIR = os.path.abspath("data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio_clean")
MANIFEST_DIR = os.path.join(DATA_DIR, "manifests")
REF_DIR = os.path.join(DATA_DIR, "reference_anchors")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)
os.makedirs(REF_DIR, exist_ok=True)

print("=" * 70)
print("🎙️ PREPARING V2 DATASET EXPANSION: NIGERIAN ENGLISH (en_ng)")
print("=" * 70)

def extract_nigerian_english_samples(target_count=30):
    print(f"Loading benjaminogbonna/nigerian_accented_english_dataset (streaming)...")
    ds = load_dataset("benjaminogbonna/nigerian_accented_english_dataset", split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    saved_records = []
    count = 0
    print("Extracting, cleaning and converting audio to 24kHz studio standards...")
    for item in ds:
        if count >= target_count:
            break
        
        sentence = item.get("sentence", item.get("text", "")).strip()
        if not sentence or len(sentence.split()) < 3:
            continue

        audio_entry = item.get("audio")
        if not isinstance(audio_entry, dict) or "bytes" not in audio_entry or not audio_entry["bytes"]:
            continue

        try:
            data, sr = sf.read(io.BytesIO(audio_entry["bytes"]))
            if data.ndim > 1:
                data = np.mean(data, axis=-1)

            # Apply 6-stage industry standard cleaning pipeline
            cleaned_audio, metrics, is_valid = clean_audio_sample(
                data,
                sr=sr,
                target_sr=24000,
                target_lufs=-20.0,
                min_duration=1.8,
                max_duration=12.0,
                min_snr=12.0
            )

            if not is_valid:
                continue

            fname = f"en_ng_{count:06d}.wav"
            fpath = os.path.join(AUDIO_DIR, fname)
            sf.write(fpath, cleaned_audio, 24000)

            record = {
                "id": f"en_ng_{count:06d}",
                "audio_path": fpath,
                "text": sanitize_text(sentence, "en"),
                "language_id": "en_ng",
                "duration": metrics["duration_sec"],
                "snr_db": metrics["snr_db"],
                "sampling_rate": 24000
            }
            saved_records.append(record)
            print(f"  [{count+1}/{target_count}] Saved: {fname} ({metrics['duration_sec']}s, SNR={metrics['snr_db']:.1f}dB) -> \"{record['text'][:50]}...\"")
            count += 1
        except Exception as e:
            continue

    print(f"✓ Successfully processed {len(saved_records)} pristine Nigerian English samples.")
    return saved_records


def build_v2_manifests(en_ng_records):
    print("\nBuilding V2 manifests with Nigerian English and expanded regional representation...")
    # Load existing train and val
    train_v1_path = os.path.join(MANIFEST_DIR, "train.jsonl")
    val_v1_path = os.path.join(MANIFEST_DIR, "val.jsonl")

    train_records = []
    if os.path.exists(train_v1_path):
        with open(train_v1_path, "r", encoding="utf-8") as f:
            train_records = [json.loads(line) for line in f if line.strip()]

    val_records = []
    if os.path.exists(val_v1_path):
        with open(val_v1_path, "r", encoding="utf-8") as f:
            val_records = [json.loads(line) for line in f if line.strip()]

    # Split en_ng into 25 train, 5 val
    en_ng_train = en_ng_records[:25]
    en_ng_val = en_ng_records[25:]

    train_v2 = train_records + en_ng_train
    val_v2 = val_records + en_ng_val

    train_v2_path = os.path.join(MANIFEST_DIR, "train_v2.jsonl")
    val_v2_path = os.path.join(MANIFEST_DIR, "val_v2.jsonl")

    with open(train_v2_path, "w", encoding="utf-8") as f:
        for r in train_v2:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_v2_path, "w", encoding="utf-8") as f:
        for r in val_v2:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✓ Written train_v2.jsonl: {len(train_v2)} samples (includes {len(en_ng_train)} Nigerian English)")
    print(f"✓ Written val_v2.jsonl: {len(val_v2)} samples (includes {len(en_ng_val)} Nigerian English)")


def setup_regional_reference_anchors(en_ng_records):
    print("\nSetting up dedicated regional reference voice anchors...")
    # 1. Nigerian English anchor
    if en_ng_records:
        best_en_ng = en_ng_records[0]
        ref_path = os.path.join(REF_DIR, "ref_en_ng.wav")
        meta_path = os.path.join(REF_DIR, "ref_en_ng.json")
        data, sr = sf.read(best_en_ng["audio_path"])
        sf.write(ref_path, data, sr)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"audio_path": ref_path, "text": best_en_ng["text"], "lang": "en_ng"}, f, indent=2)
        print(f"  ✓ Nigerian English Anchor: {ref_path} -> \"{best_en_ng['text']}\"")

    # 2. Nigerian Pidgin anchor
    pcm_audio = os.path.join(AUDIO_DIR, "pcm_000000.wav")
    if os.path.exists(pcm_audio):
        ref_path = os.path.join(REF_DIR, "ref_pcm.wav")
        meta_path = os.path.join(REF_DIR, "ref_pcm.json")
        data, sr = sf.read(pcm_audio)
        sf.write(ref_path, data, sr)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"audio_path": ref_path, "text": "Hello, I be MTN Zigi. I fit help you check your bundle.", "lang": "pcm"}, f, indent=2)
        print(f"  ✓ Nigerian Pidgin Anchor: {ref_path}")

    # 3. East African Swahili anchor
    sw_audio = os.path.join(AUDIO_DIR, "sw_000000.wav")
    if os.path.exists(sw_audio):
        ref_path = os.path.join(REF_DIR, "ref_sw.wav")
        meta_path = os.path.join(REF_DIR, "ref_sw.json")
        data, sr = sf.read(sw_audio)
        sf.write(ref_path, data, sr)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"audio_path": ref_path, "text": "Hujambo, mimi ni MTN Zigi. Naweza kukusaidia kuhusu vifurushi.", "lang": "sw"}, f, indent=2)
        print(f"  ✓ Swahili Anchor: {ref_path}")

    print("✓ Regional reference anchors configured.")


if __name__ == "__main__":
    records = extract_nigerian_english_samples(target_count=30)
    build_v2_manifests(records)
    setup_regional_reference_anchors(records)
    print("\n" + "=" * 70)
    print("✓ V2 DATA PREPARATION COMPLETE!")
    print("=" * 70)

