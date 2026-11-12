import soundfile as sf
import numpy as np
import librosa
import glob
import os

print("=" * 85)
print("AUDIT: ANALYZING AUDIO QUALITY & VOICING (V1 vs V2)")
print("=" * 85)

def analyze_file(fpath):
    y, sr = sf.read(fpath)
    if y.ndim > 1:
        y = np.mean(y, axis=-1)
    
    # 1. Peak & RMS
    peak = np.abs(y).max()
    rms = np.sqrt(np.mean(y**2))
    
    # 2. Spectral Flatness (High = noise/static, Low = voiced speech)
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    # 3. Spectral Centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # 4. Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    return {
        "dur": len(y) / sr,
        "peak": peak,
        "rms": rms,
        "flatness": flatness,
        "centroid": centroid,
        "zcr": zcr
    }

files = [
    ("Foundation English (V1)", "outputs/synthesized_samples/zigi_en_foundation.wav"),
    ("V1 Fine-tuned English", "outputs/v1_baseline/zigi_en_finetuned.wav"),
    ("V2 Nigerian English", "outputs/v2_improved/zigi_en_ng_v2.wav"),
    ("V2 Nigerian Pidgin", "outputs/v2_improved/zigi_pcm_v2.wav"),
    ("V2 Yoruba", "outputs/v2_improved/zigi_yo_v2.wav"),
    ("V2 Igbo", "outputs/v2_improved/zigi_ig_v2.wav"),
    ("V2 Hausa", "outputs/v2_improved/zigi_ha_v2.wav"),
    ("V2 Swahili", "outputs/v2_improved/zigi_sw_v2.wav"),
    ("V2 isiZulu", "outputs/v2_improved/zigi_zu_v2.wav"),
    ("V2 Persian", "outputs/v2_improved/zigi_fa_v2.wav"),
]

print(f"{'Label':25s} | {'Dur':5s} | {'Peak':6s} | {'RMS':6s} | {'Flatness':8s} | {'ZCR':6s} | Verdict")
print("-" * 85)

for label, fp in files:
    if os.path.exists(fp):
        res = analyze_file(fp)
        verdict = "VOICED SPEECH (CLEAR)" if (res["flatness"] < 0.045 and res["rms"] > 0.08) else "NOISE/WHISPER/CORRUPT"
        print(f"{label:25s} | {res['dur']:4.1f}s | {res['peak']:6.4f} | {res['rms']:6.4f} | {res['flatness']:8.5f} | {res['zcr']:6.4f} | {verdict}")
    else:
        print(f"{label:25s} | File not found: {fp}")

print("=" * 85)
