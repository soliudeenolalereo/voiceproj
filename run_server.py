"""
Run Script for MTN Zigi Voice AI FastAPI Server
===============================================
Usage:
  python run_server.py --port 8000
"""

import os
import sys
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start MTN Zigi Voice AI FastAPI Backend")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()

    # Detect environment and hardware
    env_name = os.path.basename(sys.prefix)
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        device_label = f"CUDA ({torch.cuda.get_device_name(0)})" if cuda_ok else "CPU (torch-cpu)"
    except Exception:
        cuda_ok = False
        device_label = "Unknown / CPU"

    print("=" * 75)
    print("🚀 STARTING MTN ZIGI VOICE AI FASTAPI SERVER")
    print(f"🐍 Python Executable:  {sys.executable}")
    print(f"📦 Conda Environment: {env_name}")
    print(f"⚡ Compute Device:     {device_label}")
    if not cuda_ok:
        print("💡 Tip: For 10x faster RTX 2060 GPU generation, switch to the GPU env: conda activate rag")
    print(f"📡 Serving Web UI:     http://localhost:{args.port}")
    print(f"📖 Swagger Docs:       http://localhost:{args.port}/docs")
    print("=" * 75)

    import uvicorn
    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=False, workers=1)
