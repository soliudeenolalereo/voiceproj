"""
FastAPI Voice AI Application & Experimentation Studio
=====================================================
REST API & Interactive Web Dashboard for:
1. Near Real-Time Conversational Streaming TTS (TTFB < 350ms, sentence/clause chunking).
2. Granular TTS generation with adjustable acoustic parameters.
3. Direct A/B side-by-side comparison between V1 Baseline and V2 Master Studio.
4. Speech-to-Text (ASR) transcription with Manual Text Override capability.
5. Audio streaming and playback.
"""

import os
import sys
import shutil
import uuid
import json
from typing import Optional, Literal
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from server.services import (
    synthesize_speech,
    synthesize_speech_stream,
    compare_v1_vs_v2,
    transcribe_audio,
    get_tts_engine,
    get_asr_engine,
    OUTPUT_API_DIR
)

app = FastAPI(
    title="MTN Zigi Voice AI Engine & Experimentation Studio",
    description="Multilingual TTS (V1 Baseline vs V2 Master Studio) with Near Real-Time Streaming & Dialect ASR",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Schemas ---
class TTSStreamRequest(BaseModel):
    text: str = Field(..., description="Prompt text to synthesize progressively in chunks")
    language: str = Field("en_ng", description="Target language code (en_ng, pcm, yo, ha, sw, ig, zu, fa, en)")
    guidance_scale: Optional[float] = Field(1.05, ge=0.5, le=3.0, description="Guidance scale")
    class_temperature: Optional[float] = Field(0.50, ge=0.0, le=1.5, description="Sampling temperature")
    num_step: Optional[int] = Field(10, ge=4, le=50, description="Diffusion steps (8-10 for real-time)")
    speed: Optional[float] = Field(1.00, ge=0.5, le=2.0, description="Speaking speed factor")
    apply_mastering: Optional[bool] = Field(True, description="Apply 65Hz high-pass and peak normalization")


class TTSRequest(BaseModel):
    text: str = Field(..., description="Prompt text to synthesize")
    version: Literal["v1_baseline", "v2_improved"] = Field("v2_improved", description="Model version")
    language: str = Field("en_ng", description="Target language code (en_ng, pcm, yo, ha, sw, ig, zu, fa, en)")
    guidance_scale: Optional[float] = Field(None, ge=0.5, le=3.0, description="Classifier-free guidance scale")
    class_temperature: Optional[float] = Field(None, ge=0.0, le=1.5, description="Codebook sampling temperature")
    num_step: Optional[int] = Field(None, ge=4, le=50, description="Number of diffusion codebook steps")
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
        "languages_supported": ["en_ng", "pcm", "yo", "ha", "sw", "ig", "zu", "fa", "en"],
        "real_time_streaming_supported": True
    }


@app.post("/api/tts/stream")
def api_stream_tts(req: TTSStreamRequest):
    """
    Streaming TTS Endpoint (NDJSON):
    Progressively yields audio chunks as each sentence/clause finishes.
    Enables immediate playback with TTFB < 350ms!
    """
    def event_stream():
        for chunk in synthesize_speech_stream(
            text=req.text,
            language=req.language,
            guidance_scale=req.guidance_scale or 1.05,
            class_temperature=req.class_temperature or 0.50,
            num_step=req.num_step or 10,
            speed=req.speed or 1.00,
            apply_mastering=req.apply_mastering if req.apply_mastering is not None else True
        ):
            yield json.dumps(chunk) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


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
        return {"status": "success", "data": asr_res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pipeline")
async def api_pipeline(
    file: Optional[UploadFile] = File(None),
    manual_text: Optional[str] = Form(None),
    language: str = Form("auto"),
    tts_version: str = Form("v2_improved"),
    tts_language: str = Form("en_ng")
):
    """
    End-to-End Voice Assistant Pipeline:
    Allows user to pass:
    1. MANUAL TEXT override (e.g. from Whisper-Large, Google STT, Azure).
    2. AUDIO FILE (processed via local Whisper-Tiny ASR).
    """
    try:
        text_source = "manual_external_asr"
        transcription_text = ""

        # Step 1: Resolve Text
        if manual_text and manual_text.strip():
            transcription_text = manual_text.strip()
            text_source = "manual_external_asr"
        elif file is not None:
            upload_path = os.path.join(OUTPUT_API_DIR, f"pipe_in_{uuid.uuid4().hex[:8]}_{file.filename}")
            with open(upload_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            asr_res = transcribe_audio(upload_path, language=language)
            transcription_text = asr_res["text"]
            text_source = f"local_asr_{asr_res['engine']}"
        else:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'manual_text' (from external ASR) or upload an audio file."
            )

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
  <title>MTN Zigi Voice AI Studio - Near Real-Time Conversational Suite</title>
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
    .badge-speed { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
    
    .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
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

    /* Streaming HUD styles */
    .telemetry-hud {
      background: #090e17;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 16px;
      margin-top: 14px;
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }
    .hud-box { text-align: center; }
    .hud-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; }
    .hud-val { font-size: 1.3rem; font-weight: 700; color: var(--success); margin-top: 4px; font-variant-numeric: tabular-nums; }
    
    .chunks-list { margin-top: 16px; display: flex; flex-direction: column; gap: 8px; max-height: 350px; overflow-y: auto; }
    .chunk-card {
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 14px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: all 0.2s;
    }
    .chunk-card.playing { border-color: var(--accent); background: rgba(255, 204, 0, 0.05); }
    .chunk-card.done { border-color: var(--success); opacity: 0.85; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>🎙️ MTN Zigi Voice AI Studio <span class="badge badge-mtn">FastAPI 2.1</span></h1>
        <p style="color: var(--muted); font-size: 0.85rem; margin-top: 4px;">Near Real-Time Conversational Streaming (TTFB &lt; 350ms), Multilingual TTS Tuning &amp; Dialect ASR Lab</p>
      </div>
      <div id="system-status" style="font-size: 0.8rem; text-align: right; color: var(--muted);">
        Connecting to CUDA...
      </div>
    </header>

    <div class="tabs">
      <button class="tab-btn active" onclick="switchTab('stream-tab')">⚡ Near Real-Time Stream</button>
      <button class="tab-btn" onclick="switchTab('tts-tab')">🎛️ Synthesis Studio</button>
      <button class="tab-btn" onclick="switchTab('compare-tab')">⚖️ V1 vs V2 Comparator</button>
      <button class="tab-btn" onclick="switchTab('asr-tab')">🎙️ ASR &amp; Manual Text Lab</button>
      <button class="tab-btn" onclick="switchTab('samples-tab')">📁 Showcase Library</button>
    </div>

    <!-- ============================================================== -->
    <!-- TAB 1: NEAR REAL-TIME STREAMING STUDIO                         -->
    <!-- ============================================================== -->
    <div id="stream-tab" class="panel active">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>⚡ Conversational Streaming Engine</span>
            <span class="badge badge-speed">Sentence Chunking Enabled</span>
          </div>

          <label>Target Language</label>
          <select id="stream-lang">
            <option value="en_ng">Nigerian English (en_ng) - Native Male Baritone</option>
            <option value="pcm">Nigerian Pidgin (pcm) - MTN Zigi Persona</option>
            <option value="yo">Yoruba (yo) - MTN Zigi Persona</option>
            <option value="ha">Hausa (ha) - MTN Zigi Persona</option>
            <option value="sw">Swahili (sw) - MTN Zigi Persona</option>
            <option value="ig">Igbo (ig) - MTN Zigi Persona</option>
            <option value="zu">isiZulu (zu) - MTN Zigi Persona</option>
            <option value="fa">Persian (fa) - MTN Zigi Persona</option>
            <option value="en">Standard English (en)</option>
          </select>

          <label>Multi-Sentence Conversational Prompt</label>
          <textarea id="stream-text" rows="3">Hello! Welcome to MTN customer care service. Your balance is 500 Naira. Dial star 310 hash to recharge.</textarea>

          <div class="presets">
            <button class="preset-btn" onclick="setStreamPrompt('en_ng', 'Hello! Welcome to MTN customer care service. Your balance is 500 Naira. Dial star 310 hash to recharge.')">Telecom Balance</button>
            <button class="preset-btn" onclick="setStreamPrompt('en_ng', 'Good day. We detected an active 5G data bundle on your line. You can manage subscriptions anytime by texting STOP to 312.')">5G Data</button>
            <button class="preset-btn" onclick="setStreamPrompt('pcm', 'How far! Welcome to MTN customer care. Your airtime balance na 500 Naira. Make you dial star 310 hash.')">Pidgin Care</button>
            <button class="preset-btn" onclick="setStreamPrompt('yo', 'Bawo ni! E kaabo si ile ise MTN. Owo to wa ninu apo re je aadota Naira.')">Yoruba Care</button>
            <button class="preset-btn" onclick="setStreamPrompt('ha', 'Sannu da zuwa cibiyar MTN. Ragowar kudin kiran ka Naira dari biyar ne.')">Hausa Care</button>
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Diffusion Steps (Fewer = Lower Latency)</span>
              <span id="stream-steps-val" class="slider-val">10</span>
            </div>
            <input type="range" id="stream-steps" min="6" max="24" step="2" value="10" oninput="document.getElementById('stream-steps-val').innerText = this.value">
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Speaking Speed</span>
              <span id="stream-speed-val" class="slider-val">1.00x</span>
            </div>
            <input type="range" id="stream-speed" min="0.80" max="1.30" step="0.05" value="1.00" oninput="document.getElementById('stream-speed-val').innerText = parseFloat(this.value).toFixed(2) + 'x'">
          </div>

          <button class="btn" id="stream-btn" onclick="startStreamingPlayback()">
            ⚡ Stream Audio Live (Instant TTFB)
          </button>
          <div id="stream-spinner" class="loading-spinner">⚡ Streaming audio chunks in real time...</div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Live Playback &amp; Telemetry HUD</span>
            <span id="stream-status-badge" class="badge">Idle</span>
          </div>

          <div class="telemetry-hud">
            <div class="hud-box">
              <div class="hud-label">⚡ TTFB</div>
              <div id="hud-ttfb" class="hud-val">-- ms</div>
            </div>
            <div class="hud-box">
              <div class="hud-label">⏱️ Total Latency</div>
              <div id="hud-lat" class="hud-val" style="color: var(--accent);">-- s</div>
            </div>
            <div class="hud-box">
              <div class="hud-label">🔊 Audio Length</div>
              <div id="hud-dur" class="hud-val" style="color: var(--text);">-- s</div>
            </div>
            <div class="hud-box">
              <div class="hud-label">🚀 Speed Factor</div>
              <div id="hud-rtf" class="hud-val" style="color: #38bdf8;">-- x</div>
            </div>
          </div>

          <div style="margin-top: 16px;">
            <label>Progressive Sentence Pipeline</label>
            <div id="chunks-feed" class="chunks-list">
              <div style="color: var(--muted); font-size: 0.85rem; text-align: center; padding: 24px;">
                Click "Stream Audio Live" to watch chunks generate and play back with sub-second TTFB.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ============================================================== -->
    <!-- TAB 2: TTS EXPERIMENTER & PARAMETER TUNING                     -->
    <!-- ============================================================== -->
    <div id="tts-tab" class="panel">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">
            <span>Acoustic Parameter Tuning</span>
            <span class="badge badge-v2">Master Studio</span>
          </div>

          <label>Model Version</label>
          <select id="tts-version">
            <option value="v2_improved">v2_improved (Clean Baritone + Native Nigerian English + Studio DSP)</option>
            <option value="v1_baseline">v1_baseline (Original Baseline Checkpoint)</option>
          </select>

          <label>Language</label>
          <select id="tts-lang">
            <option value="en_ng">Nigerian English (en_ng)</option>
            <option value="pcm">Nigerian Pidgin (pcm)</option>
            <option value="yo">Yoruba (yo)</option>
            <option value="ha">Hausa (ha)</option>
            <option value="sw">Swahili (sw)</option>
            <option value="ig">Igbo (ig)</option>
            <option value="zu">isiZulu (zu)</option>
            <option value="fa">Persian (fa)</option>
            <option value="en">English (en)</option>
          </select>

          <label>Prompt Text</label>
          <textarea id="tts-text">Welcome to MTN customer care service. Your balance is 500 Naira.</textarea>

          <div class="slider-group">
            <div class="slider-header">
              <span>Guidance Scale (CFG)</span>
              <span id="val-guidance" class="slider-val">1.05</span>
            </div>
            <input type="range" id="param-guidance" min="1.00" max="2.50" step="0.05" value="1.05" oninput="document.getElementById('val-guidance').innerText = this.value">
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Prosody Temperature</span>
              <span id="val-temp" class="slider-val">0.50</span>
            </div>
            <input type="range" id="param-temp" min="0.00" max="1.00" step="0.05" value="0.50" oninput="document.getElementById('val-temp').innerText = this.value">
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Diffusion Steps</span>
              <span id="val-steps" class="slider-val">10</span>
            </div>
            <input type="range" id="param-steps" min="6" max="32" step="2" value="10" oninput="document.getElementById('val-steps').innerText = this.value">
          </div>

          <div class="slider-group">
            <div class="slider-header">
              <span>Speaking Speed</span>
              <span id="val-speed" class="slider-val">1.00x</span>
            </div>
            <input type="range" id="param-speed" min="0.80" max="1.30" step="0.05" value="1.00" oninput="document.getElementById('val-speed').innerText = parseFloat(this.value).toFixed(2) + 'x'">
          </div>

          <div style="margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
            <input type="checkbox" id="param-mastering" checked style="accent-color: var(--accent); width: 16px; height: 16px;">
            <label for="param-mastering" style="margin-bottom: 0; cursor: pointer;">Enable Studio DSP Mastering (65Hz Filter + Normalization)</label>
          </div>

          <button class="btn" onclick="executeSynthesize()">Synthesize Speech</button>
          <div id="synth-loading" class="loading-spinner">Synthesizing audio on CUDA...</div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>Audio Playback &amp; Acoustic Analytics</span>
            <span id="res-badge" class="badge badge-v2">Awaiting</span>
          </div>

          <div id="player-placeholder" style="text-align: center; padding: 40px; color: var(--muted);">
            Adjust parameters on the left and click <strong>Synthesize Speech</strong> to audition.
          </div>

          <div id="player-container" style="display: none;">
            <div class="audio-player-card">
              <strong id="res-filename" style="font-size: 0.85rem; color: var(--accent);">result.wav</strong>
              <audio id="tts-audio-elem" controls></audio>
            </div>

            <div class="metrics-grid">
              <div class="metric-box">
                <div class="metric-lbl">Duration</div>
                <div id="m-dur" class="metric-num">--</div>
              </div>
              <div class="metric-box">
                <div class="metric-lbl">Pitch (F0)</div>
                <div id="m-pitch" class="metric-num">--</div>
              </div>
              <div class="metric-box">
                <div class="metric-lbl">Flatness</div>
                <div id="m-flat" class="metric-num">--</div>
              </div>
              <div class="metric-box">
                <div class="metric-lbl">RMS Energy</div>
                <div id="m-rms" class="metric-num">--</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ============================================================== -->
    <!-- TAB 3: V1 VS V2 SIDE-BY-SIDE COMPARATOR                        -->
    <!-- ============================================================== -->
    <div id="compare-tab" class="panel">
      <div class="card">
        <div class="card-title">Side-by-Side A/B Evaluation Studio</div>
        <div class="grid-2">
          <div>
            <label>Comparison Text</label>
            <input type="text" id="compare-text" value="Welcome to MTN customer care service. How may I assist you today?">
          </div>
          <div>
            <label>Language</label>
            <select id="compare-lang">
              <option value="en_ng">Nigerian English (en_ng)</option>
              <option value="pcm">Nigerian Pidgin (pcm)</option>
              <option value="yo">Yoruba (yo)</option>
              <option value="ha">Hausa (ha)</option>
              <option value="sw">Swahili (sw)</option>
              <option value="en">English (en)</option>
            </select>
          </div>
        </div>
        <button class="btn" onclick="executeCompare()">⚡ Compare V1 Baseline vs V2 Improved</button>
        <div id="compare-loading" class="loading-spinner">Generating dual audio tracks on CUDA...</div>
      </div>

      <div id="compare-results" style="display: none;">
        <div class="grid-2">
          <div class="card compare-box">
            <div class="card-title">
              <span>🔵 V1 Baseline Audio</span>
              <span class="badge badge-v1">Checkpoint-50</span>
            </div>
            <audio id="audio-v1" controls></audio>
            <div class="metrics-grid">
              <div class="metric-box"><div class="metric-lbl">Duration</div><div id="v1-dur" class="metric-num">--</div></div>
              <div class="metric-box"><div class="metric-lbl">Pitch</div><div id="v1-pitch" class="metric-num">--</div></div>
              <div class="metric-box"><div class="metric-lbl">Flatness</div><div id="v1-flat" class="metric-num">--</div></div>
              <div class="metric-box"><div class="metric-lbl">Latency</div><div id="v1-lat" class="metric-num">--</div></div>
            </div>
          </div>

          <div class="card compare-box v2">
            <div class="card-title">
              <span>🟢 V2 Improved (Master Studio)</span>
              <span class="badge badge-v2">Checkpoint-100</span>
            </div>
            <audio id="audio-v2" controls></audio>
            <div class="metrics-grid">
              <div class="metric-box"><div class="metric-lbl">Duration</div><div id="v2-dur" class="metric-num">--</div></div>
              <div class="metric-box"><div class="metric-lbl">Pitch</div><div id="v2-pitch" class="metric-num">--</div></div>
              <div class="metric-box"><div class="metric-lbl">Flatness</div><div id="v2-flat" class="metric-num">--</div></div>
              <div class="metric-box"><div class="metric-lbl">Latency</div><div id="v2-lat" class="metric-num">--</div></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ============================================================== -->
    <!-- TAB 4: ASR & MANUAL TEXT PIPELINE LAB                          -->
    <!-- ============================================================== -->
    <div id="asr-tab" class="panel">
      <div class="grid-2">
        <div class="card">
          <div class="card-title">Input Sources (Manual Override vs Audio)</div>
          
          <label style="color: var(--accent); font-weight: 600;">Option A: Manual Text Injection (External ASR)</label>
          <p style="font-size: 0.75rem; color: var(--muted); margin-bottom: 6px;">Pass text directly from Whisper-Large, Google STT, Azure, or Deepgram to skip local ASR.</p>
          <textarea id="manual-text-input" placeholder="Type or paste manual transcription text here...">Customer balance is 500 Naira. Dial star 310 hash to recharge.</textarea>

          <hr style="border: 0; border-top: 1px solid var(--border); margin: 16px 0;">

          <label style="color: var(--accent); font-weight: 600;">Option B: Local ASR Audio File Upload</label>
          <input type="file" id="asr-file-input" accept=".wav,.mp3,.m4a">
          <select id="asr-lang">
            <option value="auto">Auto-Detect Dialect</option>
            <option value="en">English / Nigerian English (en)</option>
            <option value="pcm">Nigerian Pidgin (pcm)</option>
            <option value="yo">Yoruba (yo)</option>
            <option value="ha">Hausa (ha)</option>
            <option value="sw">Swahili (sw)</option>
          </select>

          <label>TTS Model for Reply</label>
          <select id="pipeline-tts-version">
            <option value="v2_improved">v2_improved (Master Studio Edition)</option>
            <option value="v1_baseline">v1_baseline</option>
          </select>

          <div style="display: flex; gap: 10px;">
            <button class="btn" onclick="executePipeline()">🚀 Run Voice Pipeline</button>
            <button class="btn btn-secondary" onclick="executeASROnly()">Transcribe Only</button>
          </div>
          <div id="pipeline-loading" class="loading-spinner">Processing voice assistant turn...</div>
        </div>

        <div class="card">
          <div class="card-title">Resolved Text &amp; Assistant Audio Response</div>
          <div style="margin-bottom: 12px;">
            <span id="resolved-source-badge" class="badge">Awaiting Input</span>
          </div>
          <div id="resolved-text-box" style="background: #0f172a; padding: 14px; border-radius: 8px; border: 1px solid var(--border); min-height: 80px; font-size: 0.95rem;">
            Run the pipeline to inspect resolved text and hear synthesized response.
          </div>

          <div id="pipeline-player-container" style="display: none; margin-top: 16px;">
            <div class="audio-player-card">
              <strong style="color: var(--accent); font-size: 0.85rem;">Synthesized Reply Audio</strong>
              <audio id="pipeline-audio-elem" controls></audio>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ============================================================== -->
    <!-- TAB 5: SHOWCASE LIBRARY                                        -->
    <!-- ============================================================== -->
    <div id="samples-tab" class="panel">
      <div class="card">
        <div class="card-title">Pre-Generated Showcase Audio Library</div>
        <p style="color: var(--muted); font-size: 0.85rem; margin-bottom: 16px;">Stream and download reference benchmarks across all supported languages.</p>

        <h3 style="color: var(--v2); font-size: 1rem; margin-bottom: 10px;">🟢 V2 Improved Samples (Broadcast Baritone &amp; Studio DSP)</h3>
        <div id="v2-samples-grid" class="grid-2"></div>

        <h3 style="color: var(--v1); font-size: 1rem; margin: 20px 0 10px 0;">🔵 V1 Baseline Samples</h3>
        <div id="v1-samples-grid" class="grid-2"></div>
      </div>
    </div>
  </div>

  <script>
    // Tab switching
    function switchTab(tabId) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById(tabId).classList.add('active');
    }

    function setStreamPrompt(lang, text) {
      document.getElementById('stream-lang').value = lang;
      document.getElementById('stream-text').value = text;
    }

    // ================================================================
    // REAL-TIME STREAMING AUDIO QUEUE (WEB AUDIO API)
    // ================================================================
    let audioCtx = null;
    let nextPlayTime = 0;
    let streamStartTime = 0;
    let firstByteRecorded = false;

    function initAudioContext() {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }
    }

    async function startStreamingPlayback() {
      initAudioContext();
      nextPlayTime = audioCtx.currentTime;
      firstByteRecorded = false;
      streamStartTime = performance.now();

      const btn = document.getElementById('stream-btn');
      const spinner = document.getElementById('stream-spinner');
      const feed = document.getElementById('chunks-feed');
      const statusBadge = document.getElementById('stream-status-badge');

      btn.disabled = true;
      spinner.style.display = 'block';
      feed.innerHTML = '';
      statusBadge.innerText = 'Streaming ⚡';
      statusBadge.className = 'badge badge-speed';

      document.getElementById('hud-ttfb').innerText = '-- ms';
      document.getElementById('hud-lat').innerText = '-- s';
      document.getElementById('hud-dur').innerText = '-- s';
      document.getElementById('hud-rtf').innerText = '-- x';

      const payload = {
        text: document.getElementById('stream-text').value,
        language: document.getElementById('stream-lang').value,
        num_step: parseInt(document.getElementById('stream-steps').value),
        speed: parseFloat(document.getElementById('stream-speed').value),
        guidance_scale: 1.05,
        class_temperature: 0.50,
        apply_mastering: true
      };

      let totalAudioDuration = 0;
      let totalLatency = 0;

      try {
        const response = await fetch('/api/tts/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\\n');
          buffer = lines.pop(); // keep last incomplete line

          for (const line of lines) {
            if (!line.trim()) continue;
            const chunk = JSON.parse(line);

            // Record TTFB on first chunk
            if (!firstByteRecorded) {
              const ttfb = Math.round(performance.now() - streamStartTime);
              document.getElementById('hud-ttfb').innerText = ttfb + ' ms';
              firstByteRecorded = true;
            }

            totalAudioDuration += chunk.duration_sec;
            totalLatency = chunk.cumulative_latency_sec;

            // Render chunk in feed
            const card = document.createElement('div');
            card.id = `chunk-card-${chunk.chunk_index}`;
            card.className = 'chunk-card';
            card.innerHTML = `
              <div>
                <strong style="color: var(--accent); font-size: 0.85rem;">[${chunk.chunk_index + 1}/${chunk.total_chunks}]</strong>
                <span style="font-size: 0.9rem; margin-left: 6px;">"${chunk.text}"</span>
                <div style="font-size: 0.75rem; color: var(--muted); margin-top: 4px;">
                  Length: ${chunk.duration_sec}s | Compute: ${chunk.latency_sec}s
                </div>
              </div>
              <span id="badge-chunk-${chunk.chunk_index}" class="badge">Queued ⏳</span>
            `;
            feed.appendChild(card);

            // Decode base64 and schedule in Web Audio API
            await scheduleChunkAudio(chunk);
          }
        }

        // Finalize Telemetry HUD
        document.getElementById('hud-lat').innerText = totalLatency.toFixed(2) + ' s';
        document.getElementById('hud-dur').innerText = totalAudioDuration.toFixed(2) + ' s';
        const rtf = (totalLatency / (totalAudioDuration || 1)).toFixed(2);
        document.getElementById('hud-rtf').innerText = rtf + ' x';

        statusBadge.innerText = 'Completed ✓';
        statusBadge.className = 'badge badge-v2';
      } catch (err) {
        alert('Streaming error: ' + err);
        statusBadge.innerText = 'Failed';
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
      }
    }

    async function scheduleChunkAudio(chunk) {
      // Decode base64 string to ArrayBuffer
      const binaryString = atob(chunk.audio_base64);
      const len = binaryString.length;
      const bytes = new Uint8Array(len);
      for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      const audioBuffer = await audioCtx.decodeAudioData(bytes.buffer);
      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioCtx.destination);

      const startTime = Math.max(audioCtx.currentTime, nextPlayTime);
      source.start(startTime);
      nextPlayTime = startTime + audioBuffer.duration;

      // Update badge states based on scheduled playback time
      const delayToStart = (startTime - audioCtx.currentTime) * 1000;
      setTimeout(() => {
        const badge = document.getElementById(`badge-chunk-${chunk.chunk_index}`);
        const card = document.getElementById(`chunk-card-${chunk.chunk_index}`);
        if (badge) {
          badge.innerText = 'Playing 🔊';
          badge.className = 'badge badge-mtn';
        }
        if (card) card.classList.add('playing');
      }, Math.max(0, delayToStart));

      setTimeout(() => {
        const badge = document.getElementById(`badge-chunk-${chunk.chunk_index}`);
        const card = document.getElementById(`chunk-card-${chunk.chunk_index}`);
        if (badge) {
          badge.innerText = 'Done ✓';
          badge.className = 'badge badge-v2';
        }
        if (card) {
          card.classList.remove('playing');
          card.classList.add('done');
        }
      }, Math.max(0, delayToStart + (audioBuffer.duration * 1000)));
    }

    // ================================================================
    // TRADITIONAL SYNTHESIS & A/B COMPARISON
    // ================================================================
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
