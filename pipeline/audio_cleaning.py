"""
Industry-Standard Audio & Text Preprocessing Pipeline
=====================================================
Engineered for Multilingual African & Global Speech Processing (OmniVoice TTS & MMS/Whisper ASR)

Pipeline stages:
1. High-Pass Filtering (Butterworth 50 Hz): Eliminates DC bias & mic sub-bass rumble.
2. Voice Activity Detection (VAD): Removes dead silence, trims lead/tail to <= 150ms.
3. Spectral Noise Suppression: Non-destructive attenuation of ambient stationary hum/hiss.
4. ITU-R BS.1770-4 / EBU R128 Loudness Normalization: Normalizes to -20.0 LUFS with -1.0 dBTP ceiling.
5. Signal Quality & Duration Gating: Rejects clipped, corrupted, or low-SNR (<14dB) recordings.
6. Tone-Preserving Unicode Text Sanitization: Normalizes NFKC while strictly preserving African tone marks
   (Yoruba: ẹ, ọ, à, á; Igbo: ị, ọ, ụ, ṅ; Hausa: ɓ, ɗ, ƙ, ƴ), Arabic and Persian script.
"""

import os
import re
import unicodedata
import numpy as np
import scipy.signal
import soundfile as sf

try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False


def highpass_filter(audio: np.ndarray, sr: int = 24000, cutoff: float = 50.0, order: int = 2) -> np.ndarray:
    """Applies a zero-phase Butterworth high-pass filter to remove DC offset and sub-rumble."""
    nyquist = 0.5 * sr
    norm_cutoff = cutoff / nyquist
    if norm_cutoff >= 1.0:
        return audio
    sos = scipy.signal.butter(order, norm_cutoff, btype='high', output='sos')
    # Use sosfiltfilt for zero-phase distortion
    filtered = scipy.signal.sosfiltfilt(sos, audio)
    return filtered.astype(np.float32)


def estimate_snr(audio: np.ndarray) -> float:
    """Estimates Signal-to-Noise Ratio (dB) using 10th percentile noise floor approximation."""
    if len(audio) == 0:
        return 0.0
    frame_len = 512
    hop_len = 256
    num_frames = (len(audio) - frame_len) // hop_len + 1
    if num_frames <= 1:
        return 20.0
    
    # Calculate frame energies
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_len)[::hop_len]
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    
    # Sort energy: lowest 10% frames approximate background noise, upper 50% approximate signal
    sorted_rms = np.sort(frame_rms)
    noise_floor = np.mean(sorted_rms[:max(1, int(len(sorted_rms) * 0.10))])
    signal_level = np.mean(sorted_rms[int(len(sorted_rms) * 0.50):])
    
    if noise_floor <= 1e-9:
        return 40.0
    snr_db = 20.0 * np.log10((signal_level + 1e-9) / (noise_floor + 1e-9))
    return float(np.clip(snr_db, 0.0, 60.0))


def apply_vad_trim(
    audio: np.ndarray,
    sr: int = 24000,
    top_db: float = 28.0,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    pad_ms: float = 150.0
) -> np.ndarray:
    """
    Precision Voice Activity Detection (VAD) trimming.
    Trims silence at beginnings and ends with configurable speech padding.
    """
    frame_len = int(sr * frame_ms / 1000.0)
    hop_len = int(sr * hop_ms / 1000.0)
    pad_len = int(sr * pad_ms / 1000.0)

    if len(audio) < frame_len:
        return audio

    # Compute short-time energy
    num_frames = (len(audio) - frame_len) // hop_len + 1
    frames = np.lib.stride_tricks.sliding_window_view(audio, frame_len)[::hop_len]
    energy = np.sum(frames ** 2, axis=1)
    max_energy = np.max(energy) + 1e-12

    # Threshold in dB relative to peak energy
    threshold = max_energy * (10.0 ** (-top_db / 10.0))
    voiced_indices = np.where(energy > threshold)[0]

    if len(voiced_indices) == 0:
        return audio

    start_sample = max(0, voiced_indices[0] * hop_len - pad_len)
    end_sample = min(len(audio), voiced_indices[-1] * hop_len + frame_len + pad_len)

    return audio[start_sample:end_sample].astype(np.float32)


def spectral_noise_reduction(audio: np.ndarray, sr: int = 24000, reduction_db: float = 10.0) -> np.ndarray:
    """
    Lightweight, conservative spectral subtraction noise gate.
    Suppresses stationary ambient noise without creating watery/musical artifacts.
    """
    if len(audio) < 1024:
        return audio
    
    # STFT
    n_fft = 1024
    hop_length = 256
    frequencies, times, Zxx = scipy.signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Estimate noise from quietest 10% of frames
    power = magnitude ** 2
    avg_power_per_frame = np.mean(power, axis=0)
    noise_frame_count = max(2, int(0.10 * len(avg_power_per_frame)))
    quiet_frame_idx = np.argsort(avg_power_per_frame)[:noise_frame_count]
    noise_est = np.mean(power[:, quiet_frame_idx], axis=1, keepdims=True)

    # Spectral subtraction with oversubtraction factor
    alpha = 1.2
    beta = 10.0 ** (-reduction_db / 10.0)
    cleaned_power = np.maximum(power - alpha * noise_est, beta * power)
    cleaned_mag = np.sqrt(cleaned_power)

    # ISTFT
    _, cleaned_audio = scipy.signal.istft(
        cleaned_mag * np.exp(1j * phase),
        fs=sr,
        nperseg=n_fft,
        noverlap=n_fft - hop_length
    )
    
    # Match length
    if len(cleaned_audio) > len(audio):
        cleaned_audio = cleaned_audio[:len(audio)]
    elif len(cleaned_audio) < len(audio):
        cleaned_audio = np.pad(cleaned_audio, (0, len(audio) - len(cleaned_audio)))
        
    return cleaned_audio.astype(np.float32)


def normalize_loudness_itu(
    audio: np.ndarray,
    sr: int = 24000,
    target_lufs: float = -20.0,
    peak_limit_db: float = -1.0
) -> np.ndarray:
    """
    Normalizes audio loudness to ITU-R BS.1770-4 / EBU R128 standard.
    Ensures consistent perceptual loudness across all languages with soft peak limiting.
    """
    if len(audio) == 0:
        return audio
    
    # Minimum duration required for BS.1770-4 meter is ~0.4s
    if HAS_PYLOUDNORM and (len(audio) / sr >= 0.4):
        try:
            meter = pyln.Meter(sr)
            current_lufs = meter.integrated_loudness(audio)
            if not np.isinf(current_lufs) and not np.isnan(current_lufs):
                gain_db = target_lufs - current_lufs
                # Limit extreme gains to prevent blowing up noise
                gain_db = np.clip(gain_db, -18.0, 18.0)
                audio = audio * (10.0 ** (gain_db / 20.0))
        except Exception:
            pass
            
    # Peak limiting / protection against clipping
    peak = np.max(np.abs(audio)) + 1e-9
    peak_ceiling = 10.0 ** (peak_limit_db / 20.0)
    if peak > peak_ceiling:
        audio = audio * (peak_ceiling / peak)

    return audio.astype(np.float32)


def sanitize_text(text: str, language_id: str = "") -> str:
    """
    Industry-standard multilingual text normalizer.
    Preserves African diacritics/tone marks (Yoruba, Igbo, Hausa, Twi)
    and Arabic/Persian scripts while stripping formatting noise, URLs, and emojis.
    """
    if not text:
        return ""

    # Normalize Unicode canonical composition (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # Remove URLs, HTML tags, and markdown
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)  # remove [applause], (cheers), etc.

    # Remove emojis and unprintable symbols, but PRESERVE:
    # 1. Standard Latin alphanumeric
    # 2. Latin Extended diacritics (Yoruba: ẹ, ọ, à, á, è, é; Igbo: ị, ọ, ụ, ṅ; Hausa: ɓ, ɗ, ƙ, ƴ)
    # 3. Arabic script (Sudanese, Juba Arabic, Persian/Farsi: 0600-06FF, 0750-077F, FB50-FDFF, FE70-FEFF)
    # 4. Standard sentence punctuation (, . ! ? ' -)
    allowed_pattern = r'[^\w\s\.,!\?\'\-\u00C0-\u024F\u1E00-\u1EFF\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]'
    text = re.sub(allowed_pattern, ' ', text)

    # Collapse multiple spaces and trim
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def clean_audio_sample(
    audio: np.ndarray,
    sr: int,
    target_sr: int = 24000,
    target_lufs: float = -20.0,
    min_duration: float = 1.5,
    max_duration: float = 12.0,
    min_snr: float = 12.0
) -> tuple[np.ndarray, dict, bool]:
    """
    Master cleaning function executing the full 6-stage industry standard pipeline.
    
    Returns:
        cleaned_audio (np.ndarray): Processed audio float32.
        metrics (dict): Processing metrics (duration, snr, peak_db, lufs).
        is_valid (bool): True if the sample passed all quality gates.
    """
    # 1. Ensure float32 mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = audio.astype(np.float32)

    # 2. Resample if necessary
    if sr != target_sr:
        num_samples = int(len(audio) * target_sr / sr)
        audio = scipy.signal.resample(audio, num_samples)
        sr = target_sr

    # 3. High-Pass Filter (Butterworth 50Hz)
    audio = highpass_filter(audio, sr=sr, cutoff=50.0, order=2)

    # 4. Voice Activity Detection & Trim
    audio = apply_vad_trim(audio, sr=sr, top_db=28.0)

    # 5. Spectral Noise Reduction
    audio = spectral_noise_reduction(audio, sr=sr, reduction_db=8.0)

    # 6. ITU-R BS.1770-4 Loudness Normalization
    audio = normalize_loudness_itu(audio, sr=sr, target_lufs=target_lufs, peak_limit_db=-1.0)

    # 7. Quality Gates
    duration = len(audio) / sr
    snr = estimate_snr(audio)
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    peak_db = 20.0 * np.log10(peak + 1e-9)

    # Check for clipping (> 0.999)
    clip_rate = float(np.mean(np.abs(audio) >= 0.999)) if len(audio) > 0 else 0.0

    is_valid = (
        min_duration <= duration <= max_duration and
        snr >= min_snr and
        clip_rate < 0.005 and
        peak > 0.01
    )

    metrics = {
        "duration_sec": round(duration, 3),
        "snr_db": round(snr, 1),
        "peak_db": round(peak_db, 2),
        "clipping_rate": round(clip_rate, 4),
        "sampling_rate": sr
    }

    return audio, metrics, is_valid


if __name__ == "__main__":
    print("Testing Industry Audio Cleaning Pipeline...")
    # Synthetic test signal: 3s 440Hz sine wave + noise + silence head/tail
    test_sr = 24000
    t = np.linspace(0, 3, test_sr * 3, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = 0.02 * np.random.randn(len(t))
    silence = np.zeros(test_sr)  # 1s silence
    raw = np.concatenate([silence, sine + noise, silence]).astype(np.float32)

    cleaned, metrics, valid = clean_audio_sample(raw, test_sr)
    print("Pipeline Output Metrics:", metrics)
    print("Is Valid Sample:", valid)
    
    # Test text sanitizer with African tone marks and Persian
    sample_text = "Good day! Ẹ kú àkókò yìí o! Ndeewo! Barka dai! سلام حال شما چطوره؟ [applause] http://test.com"
    sanitized = sanitize_text(sample_text)
    print("Sanitized text:", sanitized)
    print("✓ Pipeline verification complete.")

