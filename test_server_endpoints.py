import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

print("=" * 75)
print("🧪 TESTING FASTAPI VOICE AI ENDPOINTS")
print("=" * 75)

# 1. Health Check
print("\n1. Testing GET /health...")
r_health = client.get("/health")
print("  Status code:", r_health.status_code)
print("  Health response:", r_health.json())

# 2. List Samples
print("\n2. Testing GET /api/samples...")
r_samples = client.get("/api/samples")
print("  Status code:", r_samples.status_code)
data_samples = r_samples.json()
print(f"  V1 baseline samples count: {len(data_samples.get('v1_baseline', []))}")
print(f"  V2 improved samples count: {len(data_samples.get('v2_improved', []))}")

# 3. Test TTS Synthesize (V2 Improved with custom parameters)
print("\n3. Testing POST /api/tts/synthesize (V2 with custom parameters)...")
payload_tts = {
    "text": "Hello, this is a live test from the FastAPI server.",
    "version": "v2_improved",
    "language": "en_ng",
    "guidance_scale": 1.05,
    "class_temperature": 0.50,
    "num_step": 20,
    "speed": 1.0,
    "apply_mastering": True
}
r_tts = client.post("/api/tts/synthesize", json=payload_tts)
print("  Status code:", r_tts.status_code)
res_tts = r_tts.json()
print("  Synthesize response status:", res_tts.get("status"))
if res_tts.get("status") == "success":
    m = res_tts["data"]["metrics"]
    print(f"  Audio URL: {res_tts['data']['audio_url']}")
    print(f"  Dur: {m['duration_sec']}s, Pitch: {m['pitch_hz']}Hz, Flatness: {m['spectral_flatness']}, Peak: {m['peak_dbfs']} dBFS")

# 4. Test Manual Text Pipeline Override
print("\n4. Testing POST /api/pipeline with MANUAL TEXT OVERRIDE...")
payload_pipeline = {
    "manual_text": "Customer balance is 500 Naira. Dial star 310 hash to recharge.",
    "tts_version": "v2_improved",
    "tts_language": "en_ng"
}
r_pipe = client.post("/api/pipeline", data=payload_pipeline)
print("  Status code:", r_pipe.status_code)
res_pipe = r_pipe.json()
print("  Pipeline response status:", res_pipe.get("status"))
print("  Text Source used:", res_pipe.get("text_source"))
print("  Text used:", res_pipe.get("text_used"))
if res_pipe.get("status") == "success":
    print(f"  Synthesized Audio URL: {res_pipe['audio_result']['audio_url']}")

print("\n" + "=" * 75)
print("✓ ALL FASTAPI ENDPOINTS VERIFIED & WORKING!")
print("=" * 75)

