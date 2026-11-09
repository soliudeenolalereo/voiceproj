"""
Showcase Audio Quality & Intelligibility Verifier
=================================================
Validates peak amplitude, RMS loudness, dynamic range, and ASR transcription
for generated MTN Zigi persona samples.
"""

import os
import glob
import numpy as np
import soundfile as sf
from transformers import pipeline

def main():
    print("=" * 70)
    print("AUDIO QUALITY & INTELLIGIBILITY AUDIT")
    print("=" * 70)

    files = sorted(glob.glob("outputs/synthesized_samples/*foundation*.wav"))
    if not files:
        print("No foundation files found!")
        return

    print(f"{'Filename':30s} | {'Dur(s)':7s} | {'RMS':7s} | {'Peak':7s} | {'Audible?':8s}")
    print("-" * 70)

    for fp in files:
        data, sr = sf.read(fp)
        rms = float(np.sqrt(np.mean(data**2)))
        peak = float(np.max(np.abs(data)))
        dur = len(data) / sr
        audible = "YES (Loud)" if rms > 0.05 and peak > 0.5 else "LOW"
        name = os.path.basename(fp)
        print(f"{name:30s} | {dur:7.2f} | {rms:7.4f} | {peak:7.4f} | {audible:8s}")

    print("\n" + "=" * 70)
    print("RUNNING AUTOMATIC SPEECH RECOGNITION (ASR) VERIFICATION...")
    print("=" * 70)

    try:
        import librosa
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to("cuda")

        for fp in files[:3]:
            name = os.path.basename(fp)
            data, sr = sf.read(fp)
            if sr != 16000:
                data = librosa.resample(data, orig_sr=sr, target_sr=16000)
            inputs = processor(data, sampling_rate=16000, return_tensors="pt").to("cuda")
            predicted_ids = model.generate(inputs.input_features)
            text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            print(f"[{name}] --> \"{text.strip()}\"")
    except Exception as e:
        print(f"ASR direct test error: {e}")

if __name__ == "__main__":
    main()
