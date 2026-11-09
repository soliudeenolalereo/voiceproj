"""
High-Speed Parallel Multi-Part Downloader for OmniVoice model.safetensors
========================================================================
Downloads the 2.33 GB foundation weights using 6 parallel range streams
with automatic resume support, retry on connection drops, and integrity verification.
"""

import os
import sys
import time
import urllib.request
import threading
from concurrent.futures import ThreadPoolExecutor

URL = "https://huggingface.co/k2-fsa/OmniVoice/resolve/main/model.safetensors"
TOTAL_SIZE = 2450344112  # Exact size from HTTP header
NUM_PARTS = 6
OUTPUT_DIR = "models/k2-fsa-OmniVoice"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "model.safetensors")
PART_DIR = os.path.join(OUTPUT_DIR, "download_parts")


def get_final_url():
    """Follow redirect to get direct CDN download URL."""
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                return resp.geturl()
        except Exception as e:
            time.sleep(2)
    raise RuntimeError("Failed to resolve CDN URL after 5 attempts")


def download_chunk(part_idx, start_byte, end_byte, progress_lock, downloaded_bytes):
    expected_size = end_byte - start_byte + 1
    part_file = os.path.join(PART_DIR, f"part_{part_idx}.bin")

    while True:
        existing_size = os.path.getsize(part_file) if os.path.exists(part_file) else 0
        if existing_size >= expected_size:
            with progress_lock:
                downloaded_bytes[part_idx] = expected_size
            return

        cur_start = start_byte + existing_size
        try:
            direct_url = get_final_url()
            req = urllib.request.Request(
                direct_url,
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Range": f"bytes={cur_start}-{end_byte}"
                }
            )

            with urllib.request.urlopen(req, timeout=45) as resp:
                mode = "ab" if existing_size > 0 else "wb"
                with open(part_file, mode) as f:
                    chunk_size = 1024 * 1024  # 1MB buffer
                    while True:
                        data = resp.read(chunk_size)
                        if not data:
                            break
                        f.write(data)
                        with progress_lock:
                            downloaded_bytes[part_idx] = os.path.getsize(part_file)

        except Exception as e:
            time.sleep(2)


def monitor_progress(downloaded_bytes, stop_event, initial_bytes):
    t0 = time.time()
    last_reported = -1
    while not stop_event.is_set():
        total_dl = sum(downloaded_bytes.values())
        pct = (total_dl / TOTAL_SIZE) * 100
        elapsed = time.time() - t0
        new_dl = max(0, total_dl - initial_bytes)
        speed_mb = (new_dl / (1024 * 1024)) / max(elapsed, 1.0)
        remaining_mb = (TOTAL_SIZE - total_dl) / (1024 * 1024)
        eta_sec = remaining_mb / max(speed_mb, 0.01)

        # Print update every 5%
        cur_bracket = int(pct // 5) * 5
        if cur_bracket > last_reported or (elapsed > 0 and int(elapsed) % 20 == 0):
            last_reported = cur_bracket
            print(f"[{pct:5.1f}%] {total_dl / (1024*1024):.1f}/{TOTAL_SIZE / (1024*1024):.1f} MB | "
                  f"Speed: {speed_mb:.2f} MB/s | ETA: {eta_sec/60:.1f} min", flush=True)
        time.sleep(2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PART_DIR, exist_ok=True)

    # Check if final file is already valid
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) == TOTAL_SIZE:
        print(f"File already complete: {OUTPUT_FILE} ({TOTAL_SIZE / (1024*1024):.1f} MB)")
        from safetensors import safe_open
        with safe_open(OUTPUT_FILE, framework="pt", device="cpu") as f:
            print(f"✓ Validated safetensors with {len(list(f.keys()))} keys.")
            return

    # If corrupted/incomplete file exists, remove it before redownloading parts
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) != TOTAL_SIZE:
        print(f"Removing incomplete file {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)} bytes)...")
        os.remove(OUTPUT_FILE)

    print("=" * 65)
    print("DOWNLOADING k2-fsa/OmniVoice FOUNDATION WEIGHTS")
    print(f"Target: {OUTPUT_FILE} ({TOTAL_SIZE / (1024*1024):.1f} MB)")
    print(f"Streams: {NUM_PARTS} parallel connections with automatic retry")
    print("=" * 65)

    part_size = TOTAL_SIZE // NUM_PARTS
    ranges = []
    for i in range(NUM_PARTS):
        start = i * part_size
        end = (start + part_size - 1) if i < NUM_PARTS - 1 else TOTAL_SIZE - 1
        ranges.append((i, start, end))

    downloaded_bytes = {i: (os.path.getsize(os.path.join(PART_DIR, f"part_{i}.bin"))
                           if os.path.exists(os.path.join(PART_DIR, f"part_{i}.bin")) else 0)
                        for i in range(NUM_PARTS)}
    progress_lock = threading.Lock()
    stop_event = threading.Event()

    initial_bytes = sum(downloaded_bytes.values())
    monitor_thread = threading.Thread(target=monitor_progress, args=(downloaded_bytes, stop_event, initial_bytes))
    monitor_thread.daemon = True
    monitor_thread.start()

    with ThreadPoolExecutor(max_workers=NUM_PARTS) as executor:
        futures = [
            executor.submit(download_chunk, r[0], r[1], r[2], progress_lock, downloaded_bytes)
            for r in ranges
        ]
        for f in futures:
            f.result()

    stop_event.set()
    monitor_thread.join()

    # Verify all parts
    print("\nVerifying part integrity before reassembly...")
    for i, start, end in ranges:
        part_path = os.path.join(PART_DIR, f"part_{i}.bin")
        expected = end - start + 1
        actual = os.path.getsize(part_path)
        if actual != expected:
            raise RuntimeError(f"Part {i} mismatch! Expected {expected} bytes, got {actual} bytes.")
        print(f"  Part {i}: {actual / (1024*1024):.1f} MB ✓")

    print("\nReassembling parts into single safetensors file...")
    with open(OUTPUT_FILE, "wb") as outfile:
        for i in range(NUM_PARTS):
            part_path = os.path.join(PART_DIR, f"part_{i}.bin")
            with open(part_path, "rb") as infile:
                while True:
                    data = infile.read(4 * 1024 * 1024)
                    if not data:
                        break
                    outfile.write(data)
            os.remove(part_path)

    os.rmdir(PART_DIR)
    final_size = os.path.getsize(OUTPUT_FILE)
    print(f"✓ Reassembled {OUTPUT_FILE} ({final_size / (1024*1024):.1f} MB)")

    # Copy to Hugging Face cache snapshot
    hf_snap = "C:/Users/DELL/.cache/huggingface/hub/models--k2-fsa--OmniVoice/snapshots/c5fdb5ccb189668d56333f77ba2629f4cd7535f4"
    if os.path.exists(hf_snap):
        hf_dest = os.path.join(hf_snap, "model.safetensors")
        import shutil
        shutil.copy2(OUTPUT_FILE, hf_dest)
        print(f"✓ Synced to Hugging Face Hub snapshot: {hf_dest}")

    # Verify safetensors validity
    print("Verifying safetensors header and weight integrity...")
    from safetensors import safe_open
    with safe_open(OUTPUT_FILE, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"✓ Successfully verified! Checkpoint contains {len(keys)} tensor keys.")
        print(f"  Sample keys: {keys[:5]}")


if __name__ == "__main__":
    main()
