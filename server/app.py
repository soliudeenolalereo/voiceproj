"""
FastAPI Voice AI Application & Experimentation Studio
=====================================================
REST API & Interactive Web Dashboard for:
1. Real-time TTS generation with adjustable acoustic parameters.
2. Direct A/B side-by-side comparison between V1 Baseline and V2 Improved.
3. Speech-to-Text (ASR) transcription with Manual Text Override capability.
4. Audio streaming and playback.
"""

import os
import sys
import shutil
import uuid
from typing import Optional, Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from server.services import (
    synthesize_speech,
    compare_v1_vs_v2,
    transcribe_audio,
    get_tts_engine,
    get_asr_engine,
    OUTPUT_API_DIR
)

app = FastAPI(
    title="MTN Zigi Voice AI Engine & Experimentation Studio",
    description="Multilingual TTS (V1 Baseline vs V2 Master Studio) & Dialect-Aware ASR with Manual Text Injection",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---
class TTSRequest(BaseModel):
    text: str = Field(..., description="Prompt text to synthesize")
    version: Literal["v1_baseline", "v2_improved"] = Field("v2_improved", description="Model version")
    language: str = Field("en_ng", description="Target language code (en_ng, pcm, yo, ha, sw, ig, zu, fa, en)")
    guidance_scale: Optional[float] = Field(None, ge=0.5, le=3.0, description="Classifier-free guidance scale")
    class_temperature: Optional[float] = Field(None, ge=0.0, le=1.5, description="Codebook sampling temperature")
    num_step: Optional[int] = Field(None, ge=10, le=50, description="Number of diffusion codebook steps")
    speed: Optional[float] = Field(None, ge=0.5, le=2.0, description="Speaking rate multiplier")
    apply_mastering: Optional[bool] = Field(None, description="Enable 65Hz high-pass and silence trim")
    anchor_type: Optional[str] = Field("auto", description="Anchor type: auto, golden_baritone, nigerian_male")


class CompareRequest(BaseModel):
    text: str = Field(..., description="Text prompt to synthesize on both V1 and V2")
    language: str = Field("en_ng", description="Target language code")


class PipelineRequest(BaseModel):
    manual_text: Optional[str] = Field(None, description="External ASR or manual text override")
    audio_path: Optional[str] = Field(None, description="Local path to audio for local ASR")
    language: Optional[str] = Field("auto", description="ASR language code")
    tts_version: Literal["v1_baseline", "v2_improved"] = Field("v2_improved", description="TTS version to synthesize")
    tts_language: Optional[str] = Field("en_ng", description="Language to use for TTS reply")


# --- API Endpoints ---

@app.get("/health")
def health_check():
    """Returns GPU status, VRAM usage, and loaded models."""
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    vram_alloc_mb = round(torch.cuda.memory_allocated() / (1024**2), 1) if cuda_available else 0
    vram_res_mb = round(torch.cuda.memory_reserved() / (1024**2), 1) if cuda_available else 0

    return {
        "status": "online",
        "device": device_name,
        "cuda_available": cuda_available,
        "vram_allocated_mb": vram_alloc_mb,
        "vram_reserved_mb": vram_res_mb,
        "versions_available": ["v1_baseline", "v2_improved"],
        "languages_supported": ["en_ng", "pcm", "yo", "ha", "sw", "ig", "zu", "fa", "en"]
    }


@app.post("/api/tts/synthesize")
def api_synthesize(req: TTSRequest):
    """
    Synthesizes speech with customizable parameters and returns audio URL + acoustic metrics.
    """
    try:
        result = synthesize_speech(
            text=req.text,
            version=req.version,
            language=req.language,
            guidance_scale=req.guidance_scale,
            class_temperature=req.class_temperature,
            num_step=req.num_step,
            speed=req.speed,
            apply_mastering=req.apply_mastering,
            anchor_type=req.anchor_type
        )
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tts/compare")
def api_compare(req: CompareRequest):
    """
    Synthesizes both V1 Baseline and V2 Improved for side-by-side A/B evaluation.
    """
    try:
        result = compare_v1_vs_v2(text=req.text, language=req.language)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/asr/transcribe")
async def api_transcribe(
    file: Optional[UploadFile] = File(None),
    audio_path: Optional[str] = Form(None),
    language: str = Form("auto")
):
    """
    Transcribes an uploaded audio file or server path using dialect-aware ASR.
    """
    try:
        if file is not None:
            upload_path = os.path.join(OUTPUT_API_DIR, f"upload_{uuid.uuid4().hex[:8]}_{file.filename}")
            with open(upload_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            target_path = upload_path
        elif audio_path and os.path.exists(audio_path):
            target_path = audio_path
        else:
            raise HTTPException(status_code=400, detail="Must provide an audio file upload or valid audio_path.")

        asr_res = transcribe_audio(target_path, language=language)
        return {
            "status": "success",
            "audio_file": os.path.basename(target_path),
            "data": asr_res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipeline")
async def api_pipeline(
    file: Optional[UploadFile] = File(None),
    manual_text: Optional[str] = Form(None),
    audio_path: Optional[str] = Form(None),
    language: str = Form("auto"),
    tts_version: str = Form("v2_improved"),
    tts_language: str = Form("en_ng")
):
    """
    Full End-to-End Voice Assistant Pipeline.
    Supports injecting external manual text (from Google, Whisper-Large, Azure, etc.)
    or transcribing via internal local ASR if manual text is omitted.
    """
    try:
        text_source = "manual_override"
        transcription_text = ""

        # Step 1: Resolve Text (Manual Injection vs Local ASR)
        if manual_text and manual_text.strip():
            transcription_text = manual_text.strip()
            text_source = "manual_external_asr"
        else:
            # Fall back to local ASR
            if file is not None:
                upload_path = os.path.join(OUTPUT_API_DIR, f"pipeline_{uuid.uuid4().hex[:8]}_{file.filename}")
                with open(upload_path, "wb") as f:
                    shutil.copyfileobj(file.file, f)
                target_path = upload_path
            elif audio_path and os.path.exists(audio_path):
                target_path = audio_path
            else:
                raise HTTPException(status_code=400, detail="Please provide either 'manual_text' or an audio file for ASR.")

            asr_res = transcribe_audio(target_path, language=language)
            transcription_text = asr_res["text"]
            text_source = f"local_asr ({asr_res['engine']})"

        if not transcription_text:
            raise HTTPException(status_code=400, detail="No speech text could be resolved.")

        # Step 2: Synthesize Response via selected TTS version (V1 vs V2)
        tts_result = synthesize_speech(
            text=transcription_text,
            version=tts_version,
            language=tts_language
        )

        return {
            "status": "success",
            "text_used": transcription_text,
            "text_source": text_source,
            "tts_version": tts_version,
            "tts_language": tts_language,
            "audio_result": tts_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    """Serves synthesized or uploaded audio files for direct in-browser playback."""
    # Search in api_generated, outputs/v2_improved, outputs/v1_baseline, or outputs/synthesized_samples
    candidate_dirs = [
        OUTPUT_API_DIR,
        os.path.join(BASE_DIR, "outputs", "v2_improved"),
        os.path.join(BASE_DIR, "outputs", "v1_baseline"),
        os.path.join(BASE_DIR, "outputs", "synthesized_samples")
    ]
    for d in candidate_dirs:
        fp = os.path.join(d, filename)
        if os.path.exists(fp):
            return FileResponse(fp, media_type="audio/wav")

    raise HTTPException(status_code=404, detail=f"Audio file '{filename}' not found.")


@app.get("/api/samples")
def list_samples():
    """Lists pre-generated showcase audio samples for instant auditioning."""
    v1_dir = os.path.join(BASE_DIR, "outputs", "v1_baseline")
    v2_dir = os.path.join(BASE_DIR, "outputs", "v2_improved")

    v1_files = sorted(os.listdir(v1_dir)) if os.path.exists(v1_dir) else []
    v2_files = sorted(os.listdir(v2_dir)) if os.path.exists(v2_dir) else []

    return {
        "v1_baseline": [{"file": f, "url": f"/api/audio/{f}"} for f in v1_files if f.endswith(".wav")],
        "v2_improved": [{"file": f, "url": f"/api/audio/{f}"} for f in v2_files if f.endswith(".wav")]
    }


# --- Interactive Single-Page Web UI ---
@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
def index_ui():
    """Serves the interactive Voice AI Testing & Experimentation Dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MTN Zigi Voice AI Studio - V1 vs V2 Experimenter</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f172a;
      --card-bg: #1e293b;
      --accent: #ffcc00; /* MTN Yellow */
      --accent-hover: #e6b800;
      --text: #f8fafc;
      --muted: #94a3b8;
      --border: #334155;
      --success: #10b981;
      --v1: #3b82f6;
      --v2: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', sans-serif; }
    body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
    .container { max-width: 1200px; margin: 0 auto; }
    header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
    h1 { font-size: 1.6rem; font-weight: 700; display: flex; align-items: center; gap: 10px; }
    .badge { font-size: 0.75rem; padding: 4px 10px; border-radius: 999px; font-weight: 600; background: var(--border); }
    .badge-v1 { background: rgba(59, 130, 246, 0.2); color: var(--v1); border: 1px solid var(--v1); }
    .badge-v2 { background: rgba(16, 185, 129, 0.2); color: var(--v2); border: 1px solid var(--v2); }
    .badge-mtn { background: var(--accent); color: #000; }
    
    .tabs { display: flex; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
    .tab-btn { background: none; border: none; color: var(--muted); font-size: 0.95rem; font-weight: 600; padding: 8px 16px; border-radius: 8px; cursor: pointer; transition: all 0.2s; }
    .tab-btn.active { background: var(--card-bg); color: var(--accent); border: 1px solid var(--border); }
    
    .panel { display: none; }
    .panel.active { display: block; }

    .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media(max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }

    .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
    .card-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 16px; display: flex; align-items: center; justify-content: space-between; }
    
    label { display: block; font-size: 0.85rem; font-weight: 500; color: var(--muted); margin-bottom: 6px; }
    input[type="text"], textarea, select {
      width: 100%; background: #0f172a; border: 1px solid var(--border); color: var(--text); border-radius: 8px; padding: 10px 14px; font-size: 0.95rem; outline: none; margin-bottom: 14px; transition: border-color 0.2s;
    }
    input[type="text"]:focus, textarea:focus, select:focus { border-color: var(--accent); }
    textarea { resize: vertical; min-height: 80px; }

    .presets { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 14px; }
    .preset-btn { background: #0f172a; border: 1px solid var(--border); color: var(--text); font-size: 0.75rem; padding: 4px 8px; border-radius: 6px; cursor: pointer; }
    .preset-btn:hover { border-color: var(--accent); color: var(--accent); }

    .slider-group { margin-bottom: 14px; }
    .slider-header { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--muted); margin-bottom: 4px; }
    .slider-val { font-weight: 600; color: var(--accent); }
    input[type="range"] { width: 100%; accent-color: var(--accent); cursor: pointer; }

    .btn { background: var(--accent); color: #000; font-weight: 600; font-size: 0.95rem; border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer; transition: all 0.2s; width: 100%; display: flex; align-items: center; justify-content: center; gap: 8px; }
    .btn:hover { background: var(--accent-hover); }
    .btn-secondary { background: var(--border); color: var(--text); }
    .btn-secondary:hover { background: #475569; }

    .audio-player-card { background: #0f172a; border: 1px solid var(--border); border-radius: 10px; padding: 16px; margin-top: 16px; }
    audio { width: 100%; margin-top: 10px; }

    .metrics-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 12px; }
    .metric-box { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 10px; text-align: center; }
    .metric-lbl { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; }
    .metric-num { font-size: 1rem; font-weight: 700; color: var(--accent); margin-top: 2px; }

    .compare-box { border-left: 4px solid var(--v1); padding-left: 14px; margin-bottom: 16px; }
    .compare-box.v2 { border-left-color: var(--v2); }

    .loading-spinner { display: none; text-align: center; margin: 16px 0; color: var(--accent); font-weight: 500; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>🎙️ MTN Zigi Voice AI Studio <span class="badge badge-mtn">FastAPI</span></h1>
        <p style="color: var(--muted); font-size: 0.85rem; margin-top: 4px;">Multilingual TTS Tuning, ASR Lab & V1 Baseline vs V2 Master Studio A/B Testing</p>
      </div>
      <div id="system-status" style="font-size: 0.8rem; text-align: right; color: var(--muted);">
        Connecting to CUDA...
      </div>
    </header>

    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('tts-tab')">TTS Experimenter</button>
      <button class="tab-btn" onclick="switchTab('compare-tab')">V1 vs V2 A/B Tester</button>
      <button class="tab-btn" onclick="switchTab('asr-tab')">ASR & Manual Text Lab</button>
      <button class="tab-btn" onclick="switchTab('showcase-tab')">Pre-Generated Showcase</button>
    </div>

    <!-- TAB 1: TTS EXPERIMENTER -->
    <div id="tts-tab" class="panel active">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>Synthesis Controls</span>
            <span id="version-badge" class="badge badge-v2">V2 Master Studio</span>
          </div>

          <label>Model Preset Version</label>
          <select id="tts-version" onchange="onVersionChange()">
            <option value="v2_improved" selected>V2 Improved (Master Studio - High-Pass + Silence Trim)</option>
            <option value="v1_baseline">V1 Baseline (Foundation 100-step - Raw Output)</option>
          </select>

          <label>Target Language</label>
          <select id="tts-lang">
            <option value="en_ng" selected>Nigerian English (en_ng) - Authentic Male Baritone</option>
            <option value="pcm">Nigerian Pidgin (pcm) - Lagos Cadence</option>
            <option value="yo">Yoruba (yo)</option>
            <option value="ha">Hausa (ha)</option>
            <option value="sw">Swahili (sw)</option>
            <option value="ig">Igbo (ig)</option>
            <option value="zu">isiZulu (zu)</option>
            <option value="fa">Persian / Farsi (fa)</option>
            <option value="en">Standard English (en)</option>
          </select>

          <label>Prompt Text</label>
          <div class="presets">
            <button class="preset-btn" onclick="setPrompt('en_ng', 'Welcome to MTN Nigeria. I am Zigi, your digital assistant. How may I assist you today?')">EN_NG Preset</button>
            <button class="preset-btn" onclick="setPrompt('pcm', 'Hello, I be MTN Zigi. I fit help you check your bundle and airtime mata. Abeg, feel free to ask me anything.')">Pidgin Preset</button>
            <button class="preset-btn" onclick="setPrompt('ha', 'Barka dai, ni ne MTN Zigi. Zan iya taimaka maka da bayani game da kundin tsarinka da katin magana.')">Hausa Preset</button>
            <button class="preset-btn" onclick="setPrompt('yo', 'Ẹ lẹ́ẹ̀ o, èmi ni MTN Zigi. Mo lè ràn yín lọ́wọ́ lórí ọ̀rọ̀ bọ́ńdù àti iye owó ìpè yín.')">Yoruba Preset</button>
          </div>
          <textarea id="tts-text">Welcome to MTN Nigeria. I am Zigi, your digital assistant. You can check your account balance, borrow airtime, or purchase a data bundle. How may I assist you today?</textarea>

          <div class="slider-group">
            <div class="slider-header">
              <span>Classifier-Free Guidance Scale</span>
              <span id="val-guidance" class="slider-val">1.05</span>
            </div>
            <input type="range" id="param-guidance" min="0.8" max="2.2" step="0.05" value="1.05" oninput="updateSlider('guidance')">
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Class Temperature (Prosody Variance)</span>
              <span id="val-temp" class="slider-val">0.50</span>
            </div>
            <input type="range" id="param-temp" min="0.0" max="1.0" step="0.05" value="0.50" oninput="updateSlider('temp')">
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Diffusion Codebook Steps</span>
              <span id="val-steps" class="slider-val">28</span>
            </div>
            <input type="range" id="param-steps" min="16" max="40" step="2" value="28" oninput="updateSlider('steps')">
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Speaking Speed</span>
              <span id="val-speed" class="slider-val">1.00x</span>
            </div>
            <input type="range" id="param-speed" min="0.8" max="1.3" step="0.05" value="1.00" oninput="updateSlider('speed')">
          </div>

          <div style="margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
            <input type="checkbox" id="param-mastering" checked style="accent-color: var(--accent); cursor: pointer;">
            <label for="param-mastering" style="margin: 0; cursor: pointer; color: var(--text);">Apply 3-Stage Broadcast DSP Mastering (High-pass 65Hz + Silence Trim)</label>
          </div>

          <button id="btn-synth" class="btn" onclick="executeSynthesize()">🚀 Synthesize Audio</button>
          <div id="synth-loading" class="loading-spinner">⚙️ Running Diffusion Synthesis on RTX 2060...</div>
        </div>

        <div>
          <div class="card">
            <div class="card-title">Audio Output & Analysis</div>
            <div id="player-placeholder" style="color: var(--muted); text-align: center; padding: 40px 0;">
              Synthesize a sample on the left to see the audio player and acoustic metrics here.
            </div>

            <div id="player-container" style="display: none;">
              <div class="audio-player-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                  <strong id="res-filename" style="color: var(--accent);">sample.wav</strong>
                  <span id="res-badge" class="badge badge-v2">V2 Improved</span>
                </div>
                <audio id="tts-audio-elem" controls></audio>
              </div>

              <div class="metrics-grid">
                <div class="metric-box">
                  <div class="metric-lbl">Duration</div>
                  <div id="m-dur" class="metric-num">0.0s</div>
                </div>
                <div class="metric-box">
                  <div class="metric-lbl">Pitch (F0)</div>
                  <div id="m-pitch" class="metric-num">0 Hz</div>
                </div>
                <div class="metric-box">
                  <div class="metric-lbl">Flatness (Noise)</div>
                  <div id="m-flat" class="metric-num">0.0000</div>
                </div>
                <div class="metric-box">
                  <div class="metric-lbl">Peak / RMS</div>
                  <div id="m-rms" class="metric-num">0.00</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 2: V1 vs V2 A/B TESTER -->
    <div id="compare-tab" class="panel">
      <div class="card">
        <div class="card-title">Direct Side-by-Side A/B Audio Comparison</div>
        <p style="color: var(--muted); font-size: 0.85rem; margin-bottom: 14px;">
          Synthesize any text prompt simultaneously across both <strong>V1 Baseline</strong> and <strong>V2 Master Studio</strong> to test timbre, room acoustics, and enunciation.
        </p>

        <label>Comparison Text</label>
        <textarea id="compare-text">Welcome to MTN Nigeria. I am Zigi, your digital assistant. You can check your account balance, borrow airtime, or purchase a data bundle. How may I assist you today?</textarea>

        <div style="display: flex; gap: 10px; margin-bottom: 16px;">
          <div style="flex: 1;">
            <label>Language</label>
            <select id="compare-lang">
              <option value="en_ng" selected>Nigerian English (en_ng)</option>
              <option value="pcm">Nigerian Pidgin (pcm)</option>
              <option value="yo">Yoruba (yo)</option>
              <option value="ha">Hausa (ha)</option>
              <option value="sw">Swahili (sw)</option>
              <option value="fa">Persian (fa)</option>
            </select>
          </div>
          <div style="flex: 1; display: flex; align-items: flex-end;">
            <button class="btn" onclick="executeCompare()">⚡ Compare V1 Baseline vs V2 Improved</button>
          </div>
        </div>
        <div id="compare-loading" class="loading-spinner">⚙️ Synthesizing V1 and V2 models in parallel...</div>

        <div id="compare-results" style="display: none; margin-top: 20px;">
          <div class="grid-2">
            <!-- V1 Card -->
            <div class="card compare-box">
              <div class="card-title">
                <span>Model V1 (Baseline)</span>
                <span class="badge badge-v1">V1 Foundation</span>
              </div>
              <audio id="audio-v1" controls style="width: 100%;"></audio>
              <div class="metrics-grid" style="margin-top: 12px;">
                <div class="metric-box"><div class="metric-lbl">Dur</div><div id="v1-dur" class="metric-num">0s</div></div>
                <div class="metric-box"><div class="metric-lbl">Pitch</div><div id="v1-pitch" class="metric-num">0 Hz</div></div>
                <div class="metric-box"><div class="metric-lbl">Flatness</div><div id="v1-flat" class="metric-num">0.00</div></div>
                <div class="metric-box"><div class="metric-lbl">Latency</div><div id="v1-lat" class="metric-num">0s</div></div>
              </div>
            </div>

            <!-- V2 Card -->
            <div class="card compare-box v2">
              <div class="card-title">
                <span>Model V2 (Master Studio)</span>
                <span class="badge badge-v2">V2 Master</span>
              </div>
              <audio id="audio-v2" controls style="width: 100%;"></audio>
              <div class="metrics-grid" style="margin-top: 12px;">
                <div class="metric-box"><div class="metric-lbl">Dur</div><div id="v2-dur" class="metric-num">0s</div></div>
                <div class="metric-box"><div class="metric-lbl">Pitch</div><div id="v2-pitch" class="metric-num">0 Hz</div></div>
                <div class="metric-box"><div class="metric-lbl">Flatness</div><div id="v2-flat" class="metric-num">0.00</div></div>
                <div class="metric-box"><div class="metric-lbl">Latency</div><div id="v2-lat" class="metric-num">0s</div></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 3: ASR & MANUAL TEXT LAB -->
    <div id="asr-tab" class="panel">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">1. Speech Input (Audio File or Manual Text Override)</div>
          
          <label>Upload Audio for Local ASR Transcription</label>
          <input type="file" id="asr-file-input" accept="audio/*" style="margin-bottom: 14px;">

          <label>Dialect Hint</label>
          <select id="asr-lang">
            <option value="auto" selected>Auto-Detect Dialect</option>
            <option value="en_ng">Nigerian English (en_ng)</option>
            <option value="pcm">Nigerian Pidgin (pcm)</option>
            <option value="ha">Hausa (ha)</option>
            <option value="sw">Swahili (sw)</option>
            <option value="yo">Yoruba (yo)</option>
            <option value="fa">Persian (fa)</option>
          </select>

          <button class="btn btn-secondary" onclick="executeASROnly()" style="margin-bottom: 16px;">🔍 Run Local ASR Transcription</button>
          
          <div style="border-top: 1px dashed var(--border); padding-top: 14px; margin-top: 6px;">
            <label style="color: var(--accent); font-weight: 600;">Manual Text Override (External ASR / Ground Truth)</label>
            <p style="color: var(--muted); font-size: 0.75rem; margin-bottom: 6px;">
              Paste manual text obtained from external ASR engines (Google Cloud, Azure, Deepgram, Whisper-Large, etc.). When populated, the pipeline bypasses local ASR and synthesizes this text directly!
            </p>
            <textarea id="manual-text-input" placeholder="Type or paste external ASR transcription here..."></textarea>
          </div>

          <label>Synthesize Response Model Version</label>
          <select id="pipeline-tts-version">
            <option value="v2_improved" selected>V2 Improved (Master Studio)</option>
            <option value="v1_baseline">V1 Baseline</option>
          </select>

          <button class="btn" onclick="executePipeline()">⚡ Run Pipeline & Synthesize Voice</button>
          <div id="pipeline-loading" class="loading-spinner">⚙️ Processing speech pipeline...</div>
        </div>

        <div class="card">
          <div class="card-title">2. Transcription & Synthesized Reply</div>
          
          <label>Resolved Transcription Text</label>
          <div id="resolved-text-box" style="background: #0f172a; border: 1px solid var(--border); border-radius: 8px; padding: 14px; min-height: 80px; margin-bottom: 14px; color: var(--text);">
            Waiting for audio upload or manual text...
          </div>
          <div id="resolved-source-badge" style="font-size: 0.75rem; color: var(--muted); margin-bottom: 14px;">Source: None</div>

          <div id="pipeline-player-container" style="display: none;">
            <div class="audio-player-card">
              <strong>Synthesized Speech Reply</strong>
              <audio id="pipeline-audio-elem" controls style="width: 100%; margin-top: 10px;"></audio>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- TAB 4: PRE-GENERATED SHOWCASE -->
    <div id="showcase-tab" class="panel">
      <div class="card">
        <div class="card-title">Audition Pre-Generated Master Samples</div>
        <p style="color: var(--muted); font-size: 0.85rem; margin-bottom: 14px;">
          Instant in-browser playback of the official pre-synthesized audio files across all 8 supported dialects.
        </p>

        <h3 style="font-size: 1rem; color: var(--v2); margin: 16px 0 8px 0;">V2 Improved (Master Studio Edition - 114 to 128 Hz Baritone)</h3>
        <div class="grid-2" id="v2-samples-grid"></div>

        <h3 style="font-size: 1rem; color: var(--v1); margin: 24px 0 8px 0;">V1 Baseline (Foundation 100-step Checkpoint)</h3>
        <div class="grid-2" id="v1-samples-grid"></div>
      </div>
    </div>
  </div>

  <script>
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById(tabId).classList.add('active');
    }

    function updateSlider(name) {
      const el = document.getElementById('param-' + name);
      const valEl = document.getElementById('val-' + name);
      valEl.innerText = name === 'speed' ? el.value + 'x' : el.value;
    }

    function onVersionChange() {
      const ver = document.getElementById('tts-version').value;
      const badge = document.getElementById('version-badge');
      if (ver === 'v1_baseline') {
        badge.innerText = 'V1 Baseline';
        badge.className = 'badge badge-v1';
        document.getElementById('param-guidance').value = 1.00;
        document.getElementById('param-temp').value = 0.70;
        document.getElementById('param-steps').value = 24;
        document.getElementById('param-mastering').checked = false;
      } else {
        badge.innerText = 'V2 Master Studio';
        badge.className = 'badge badge-v2';
        document.getElementById('param-guidance').value = 1.05;
        document.getElementById('param-temp').value = 0.50;
        document.getElementById('param-steps').value = 28;
        document.getElementById('param-mastering').checked = true;
      }
      updateSlider('guidance');
      updateSlider('temp');
      updateSlider('steps');
    }

    function setPrompt(lang, text) {
      document.getElementById('tts-lang').value = lang;
      document.getElementById('tts-text').value = text;
    }

    async function executeSynthesize() {
      const loading = document.getElementById('synth-loading');
      loading.style.display = 'block';
      const body = {
        text: document.getElementById('tts-text').value,
        version: document.getElementById('tts-version').value,
        language: document.getElementById('tts-lang').value,
        guidance_scale: parseFloat(document.getElementById('param-guidance').value),
        class_temperature: parseFloat(document.getElementById('param-temp').value),
        num_step: parseInt(document.getElementById('param-steps').value),
        speed: parseFloat(document.getElementById('param-speed').value),
        apply_mastering: document.getElementById('param-mastering').checked
      };

      try {
        const res = await fetch('/api/tts/synthesize', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body)
        });
        const data = await res.json();
        if (data.status === 'success') {
          const item = data.data;
          document.getElementById('player-placeholder').style.display = 'none';
          document.getElementById('player-container').style.display = 'block';
          document.getElementById('res-filename').innerText = item.file_id;
          document.getElementById('res-badge').innerText = item.version;
          document.getElementById('res-badge').className = item.version === 'v1_baseline' ? 'badge badge-v1' : 'badge badge-v2';
          
          const audio = document.getElementById('tts-audio-elem');
          audio.src = item.audio_url;
          audio.play();

          document.getElementById('m-dur').innerText = item.metrics.duration_sec + 's';
          document.getElementById('m-pitch').innerText = item.metrics.pitch_hz + ' Hz';
          document.getElementById('m-flat').innerText = item.metrics.spectral_flatness;
          document.getElementById('m-rms').innerText = item.metrics.rms_loudness;
        }
      } catch(e) {
        alert('Synthesis failed: ' + e);
      } finally {
        loading.style.display = 'none';
      }
    }

    async function executeCompare() {
      const loading = document.getElementById('compare-loading');
      const results = document.getElementById('compare-results');
      loading.style.display = 'block';
      results.style.display = 'none';

      const body = {
        text: document.getElementById('compare-text').value,
        language: document.getElementById('compare-lang').value
      };

      try {
        const res = await fetch('/api/tts/compare', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(body)
        });
        const data = await res.json();
        if (data.status === 'success') {
          const v1 = data.data.v1_baseline;
          const v2 = data.data.v2_improved;

          document.getElementById('audio-v1').src = v1.audio_url;
          document.getElementById('v1-dur').innerText = v1.metrics.duration_sec + 's';
          document.getElementById('v1-pitch').innerText = v1.metrics.pitch_hz + ' Hz';
          document.getElementById('v1-flat').innerText = v1.metrics.spectral_flatness;
          document.getElementById('v1-lat').innerText = v1.metrics.latency_sec + 's';

          document.getElementById('audio-v2').src = v2.audio_url;
          document.getElementById('v2-dur').innerText = v2.metrics.duration_sec + 's';
          document.getElementById('v2-pitch').innerText = v2.metrics.pitch_hz + ' Hz';
          document.getElementById('v2-flat').innerText = v2.metrics.spectral_flatness;
          document.getElementById('v2-lat').innerText = v2.metrics.latency_sec + 's';

          results.style.display = 'block';
        }
      } catch(e) {
        alert('Comparison failed: ' + e);
      } finally {
        loading.style.display = 'none';
      }
    }

    async function executeASROnly() {
      const fileInput = document.getElementById('asr-file-input');
      if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please choose an audio file to transcribe.');
        return;
      }
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('language', document.getElementById('asr-lang').value);

      try {
        const res = await fetch('/api/asr/transcribe', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'success') {
          document.getElementById('resolved-text-box').innerText = data.data.text;
          document.getElementById('resolved-source-badge').innerText = 'Source: Local ASR (' + data.data.engine + ')';
        }
      } catch(e) {
        alert('ASR failed: ' + e);
      }
    }

    async function executePipeline() {
      const loading = document.getElementById('pipeline-loading');
      loading.style.display = 'block';

      const formData = new FormData();
      const fileInput = document.getElementById('asr-file-input');
      const manualText = document.getElementById('manual-text-input').value;

      if (fileInput.files && fileInput.files.length > 0) {
        formData.append('file', fileInput.files[0]);
      }
      formData.append('manual_text', manualText);
      formData.append('language', document.getElementById('asr-lang').value);
      formData.append('tts_version', document.getElementById('pipeline-tts-version').value);
      formData.append('tts_language', document.getElementById('asr-lang').value === 'auto' ? 'en_ng' : document.getElementById('asr-lang').value);

      try {
        const res = await fetch('/api/pipeline', { method: 'POST', body: formData });
        const data = await res.json();
        if (data.status === 'success') {
          document.getElementById('resolved-text-box').innerText = data.text_used;
          document.getElementById('resolved-source-badge').innerText = 'Source: ' + data.text_source;
          
          document.getElementById('pipeline-player-container').style.display = 'block';
          const audio = document.getElementById('pipeline-audio-elem');
          audio.src = data.audio_result.audio_url;
          audio.play();
        }
      } catch(e) {
        alert('Pipeline execution failed: ' + e);
      } finally {
        loading.style.display = 'none';
      }
    }

    // Populate pre-generated showcase
    async function loadShowcase() {
      try {
        const res = await fetch('/api/samples');
        const data = await res.json();
        
        const v2Grid = document.getElementById('v2-samples-grid');
        v2Grid.innerHTML = data.v2_improved.map(item => `
          <div class="metric-box" style="text-align: left; margin-bottom: 8px;">
            <strong style="color: var(--accent); font-size: 0.85rem;">${item.file}</strong>
            <audio controls src="${item.url}" style="width: 100%; margin-top: 6px;"></audio>
          </div>
        `).join('');

        const v1Grid = document.getElementById('v1-samples-grid');
        v1Grid.innerHTML = data.v1_baseline.map(item => `
          <div class="metric-box" style="text-align: left; margin-bottom: 8px;">
            <strong style="color: var(--v1); font-size: 0.85rem;">${item.file}</strong>
            <audio controls src="${item.url}" style="width: 100%; margin-top: 6px;"></audio>
          </div>
        `).join('');
      } catch(e) {}
    }

    // Check health on start
    async function checkHealth() {
      try {
        const res = await fetch('/health');
        const d = await res.json();
        document.getElementById('system-status').innerHTML = `
          🟢 Online: <strong>${d.device}</strong> | VRAM: ${d.vram_allocated_mb} MB
        `;
      } catch(e) {
        document.getElementById('system-status').innerText = '🔴 Server Offline';
      }
    }

    checkHealth();
    loadShowcase();
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

