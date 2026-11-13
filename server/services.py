"""
Voice AI Backend Services (FastAPI Server)
===========================================
Manages model lifecycle (singleton), TTS synthesis for V1 and V2,
ASR transcription (with manual text override support), acoustic metric analysis,
and near real-time sentence/clause streaming.
"""

import os
import sys
import uuid
import time
import io
import base64
import numpy as np
import soundfile as sf
import scipy.signal
import torch
import librosa

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from pipeline.tts_engine_v2 import TTSEngineV2, studio_master_dsp
from pipeline.asr_engine_v2 import ASREngineV2

AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio_clean")
OUTPUT_API_DIR = os.path.join(BASE_DIR, "outputs", "api_generated")
os.makedirs(OUTPUT_API_DIR, exist_ok=True)

# Global singleton containers
_tts_engine = None
_asr_engine = None


def get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        print("[Services] Initializing TTSEngineV2 singleton on CUDA...")
        _tts_engine = TTSEngineV2()
    return _tts_engine


def get_asr_engine():
    global _asr_engine
    if _asr_engine is None:
        print("[Services] Initializing ASREngineV2 singleton on CUDA...")
        _asr_engine = ASREngineV2()
    return _asr_engine


def calc_audio_metrics(wav_data: np.ndarray, sr: int = 24000) -> dict:
    """Computes comprehensive acoustic and signal metrics for synthesized audio."""
    if wav_data.ndim > 1:
        wav_data = np.mean(wav_data, axis=-1)

    dur = float(len(wav_data) / sr)
    peak = float(np.abs(wav_data).max())
    peak_dbfs = float(20 * np.log10(peak)) if peak > 1e-6 else -99.0
    rms = float(np.sqrt(np.mean(wav_data**2)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=wav_data)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(wav_data)))

    # Pitch tracking on first 5 seconds
    sample_len = min(len(wav_data), sr * 5)
    try:
        f0, voiced_flag, _ = librosa.pyin(wav_data[:sample_len], fmin=60, fmax=400, sr=sr)
        pitch = float(np.nanmean(f0)) if np.any(voiced_flag) else 0.0
    except Exception:
        pitch = 0.0

    return {
        "duration_sec": round(dur, 2),
        "peak_amplitude": round(peak, 4),
        "peak_dbfs": round(peak_dbfs, 2),
        "rms_loudness": round(rms, 4),
        "spectral_flatness": round(flatness, 5),
        "zero_crossing_rate": round(zcr, 4),
        "pitch_hz": round(pitch, 1),
        "sample_rate": sr
    }


def synthesize_speech(
    text: str,
    version: str = "v2_improved",
    language: str = "en_ng",
    guidance_scale: float = None,
    class_temperature: float = None,
    num_step: int = None,
    speed: float = None,
    apply_mastering: bool = None,
    anchor_type: str = "auto"
) -> dict:
    """
    Synthesizes speech using either V1 Baseline or V2 Master Studio presets,
    with full granular parameter override capabilities.
    """
    engine = get_tts_engine()

    # Apply version presets if not explicitly overridden
    is_v1 = (version.lower() == "v1_baseline")
    if is_v1:
        default_guidance = 1.00
        default_temp = 0.70
        default_steps = 20
        default_speed = 1.00
        default_mastering = False
        default_anchor = "golden_baritone"
    else:  # v2_improved
        default_guidance = 1.05
        default_temp = 0.50
        default_steps = 10  # Fast 10-step diffusion for sub-second latency
        default_speed = 1.00
        default_mastering = True
        default_anchor = "auto"

    cfg_guidance = guidance_scale if guidance_scale is not None else default_guidance
    cfg_temp = class_temperature if class_temperature is not None else default_temp
    cfg_steps = num_step if num_step is not None else default_steps
    cfg_speed = speed if speed is not None else default_speed
    cfg_mastering = apply_mastering if apply_mastering is not None else default_mastering
    cfg_anchor = anchor_type if anchor_type != "auto" else default_anchor

    omni_lang_map = {
        "en_ng": "en", "en": "en", "pcm": "en", "yo": "yo",
        "ig": "ig", "ha": "ha", "sw": "sw", "zu": "zu", "fa": "fa"
    }
    target_lang = omni_lang_map.get(language, "en")

    t0 = time.time()
    gen_kwargs = {
        "text": text,
        "language": target_lang,
        "speed": float(cfg_speed),
        "num_step": int(cfg_steps),
        "guidance_scale": float(cfg_guidance),
        "class_temperature": float(cfg_temp),
        "postprocess_output": False,
    }

    # Use pre-cached prompt embeddings when available for zero-overhead inference
    if not is_v1 and cfg_anchor == "auto":
        cached_prompt = engine.get_cached_prompt(language)
        if cached_prompt is not None:
            gen_kwargs["voice_clone_prompt"] = cached_prompt
        anchor_name = "Auto Unified Voice (Pre-cached VRAM Embeddings)"
    else:
        if cfg_anchor == "nigerian_male" or (cfg_anchor == "auto" and language == "en_ng"):
            ref_audio = engine.nigerian_male_audio
            ref_text = engine.nigerian_male_text
            anchor_name = "Nigerian English Male Baritone (119 Hz)"
        else:
            ref_audio = engine.golden_audio
            ref_text = engine.golden_text
            anchor_name = "Golden Broadcast Studio Baritone (127 Hz)"

        if ref_audio is not None and ref_text is not None and os.path.exists(ref_audio):
            gen_kwargs["ref_audio"] = ref_audio
            gen_kwargs["ref_text"] = ref_text

    with torch.inference_mode():
        audios = engine.model.generate(**gen_kwargs)
        raw_wav = audios[0]

    if cfg_mastering:
        final_wav = studio_master_dsp(raw_wav, 24000)
    else:
        peak = np.abs(raw_wav).max()
        final_wav = (raw_wav / peak * 0.90) if peak > 1e-5 else raw_wav

    latency_sec = round(time.time() - t0, 2)

    # Save output file
    file_id = f"{version}_{language}_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(OUTPUT_API_DIR, file_id)
    sf.write(out_path, final_wav, 24000)

    metrics = calc_audio_metrics(final_wav, 24000)
    metrics["latency_sec"] = latency_sec

    return {
        "file_id": file_id,
        "audio_url": f"/api/audio/{file_id}",
        "file_path": out_path,
        "version": version,
        "language": language,
        "metrics": metrics,
        "config": {
            "guidance_scale": cfg_guidance,
            "class_temperature": cfg_temp,
            "num_step": cfg_steps,
            "speed": cfg_speed,
            "apply_mastering": cfg_mastering,
            "anchor_name": anchor_name
        }
    }


def synthesize_speech_stream(
    text: str,
    language: str = "en_ng",
    guidance_scale: float = 1.05,
    class_temperature: float = 0.50,
    num_step: int = 10,
    speed: float = 1.00,
    apply_mastering: bool = True
):
    """
    Sentence/Clause progressive streaming generator.
    Yields NDJSON chunk dicts containing base64 audio and latency metrics.
    Enables Time-To-First-Byte (TTFB) < 350ms!
    """
    engine = get_tts_engine()
    t_stream_start = time.time()

    for chunk in engine.synthesize_stream(
        text=text,
        lang_code=language,
        speed=speed,
        num_step=num_step,
        guidance_scale=guidance_scale,
        class_temperature=class_temperature,
        apply_mastering=apply_mastering
    ):
        # Convert numpy wav into in-memory WAV base64
        buf = io.BytesIO()
        sf.write(buf, chunk["audio"], chunk["sample_rate"], format="WAV")
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode("ascii")

        cum_latency = round(time.time() - t_stream_start, 3)

        yield {
            "chunk_index": chunk["chunk_index"],
            "total_chunks": chunk["total_chunks"],
            "text": chunk["text"],
            "audio_base64": audio_b64,
            "duration_sec": chunk["duration_sec"],
            "latency_sec": chunk["latency_sec"],
            "cumulative_latency_sec": cum_latency,
            "is_first": chunk["is_first"],
            "is_last": chunk["is_last"],
            "sample_rate": chunk["sample_rate"]
        }


def compare_v1_vs_v2(text: str, language: str = "en_ng") -> dict:
    """Generates both V1 and V2 side-by-side for direct A/B testing."""
    v1_result = synthesize_speech(text=text, version="v1_baseline", language=language)
    v2_result = synthesize_speech(text=text, version="v2_improved", language=language)
    return {
        "text": text,
        "language": language,
        "v1_baseline": v1_result,
        "v2_improved": v2_result
    }


def transcribe_audio(audio_path: str, language: str = "auto") -> dict:
    """Transcribes an audio file with the dialect-aware ASR engine."""
    asr = get_asr_engine()
    return asr.transcribe(audio_path, lang=language)
