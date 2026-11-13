import time
from server.services import synthesize_speech_stream

print("=" * 70)
print("⚡ TESTING REAL-TIME STREAMING TTS PIPELINE")
print("=" * 70)

text = "Hello! Welcome to MTN customer care service. Your balance is 500 Naira. Dial star 310 hash to recharge."
print(f"Target Prompt: {text}\n")

t0 = time.time()
ttfb = None

for chunk in synthesize_speech_stream(text=text, language="en_ng", num_step=10):
    if ttfb is None:
        ttfb = time.time() - t0
        print(f"⚡ TIME TO FIRST BYTE (TTFB): {ttfb*1000:.1f} ms!")
        print("-" * 70)
    
    print(f"Chunk {chunk['chunk_index']+1}/{chunk['total_chunks']}: \"{chunk['text']}\"")
    print(f"  Duration: {chunk['duration_sec']:.2f}s | Latency: {chunk['latency_sec']:.2f}s | Cumulative: {chunk['cumulative_latency_sec']:.2f}s | Base64 size: {len(chunk['audio_base64'])} chars")

print("-" * 70)
print(f"✓ All chunks generated in {time.time() - t0:.2f}s total.")
print("=" * 70)
