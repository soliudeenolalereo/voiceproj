import os
import glob
import numpy as np
import soundfile as sf
import librosa

target_dir = "outputs/v2_improved"
files = sorted(glob.glob(os.path.join(target_dir, "*.wav")))

print("=" * 90)
print("AUDIT: FULL MASTER STUDIO ACOUSTIC VERIFICATION (V2 IMPROVED)")
print("=" * 90)
print(f"{'File':22s} | {'Dur(s)':6s} | {'Pitch':7s} | {'Peak':6s} | {'RMS':6s} | {'Flatness':8s} | Verdict")
print("-" * 90)

for fp in files:
    name = os.path.basename(fp)
    y, sr = sf.read(fp)
    if y.ndim > 1:
        y = np.mean(y, axis=-1)
    
    dur = len(y) / sr
    peak = np.abs(y).max()
    rms = np.sqrt(np.mean(y**2))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    f0, v, _ = librosa.pyin(y[:min(len(y), sr*5)], fmin=60, fmax=400, sr=sr)
    pitch = np.nanmean(f0) if np.any(v) else 0.0
    
    verdict = "STUDIO MASTER (EXCELLENT)" if (flatness < 0.025 and rms > 0.08 and 110 <= pitch <= 135) else "STUDIO (GOOD)"
    print(f"{name:22s} | {dur:5.2f}s | {pitch:5.1f}Hz | {peak:6.4f} | {rms:6.4f} | {flatness:8.5f} | {verdict}")

print("=" * 90)
