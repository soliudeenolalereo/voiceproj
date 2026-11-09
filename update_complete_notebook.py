import json
import os

with open("voice_rtx2060_complete.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

cell13_source = [
    "# Multilingual Speech Synthesis with Local Foundation / Fine-Tuned Checkpoint\n",
    "import os\n",
    "import json\n",
    "import torch\n",
    "import numpy as np\n",
    "import soundfile as sf\n",
    "from omnivoice import OmniVoice\n",
    "from IPython.display import Audio, display\n",
    "\n",
    "# Path to the fine-tuned checkpoint (or models/k2-fsa-OmniVoice for foundation zero-shot)\n",
    "model_path = './models/checkpoints_rtx2060_foundation/checkpoint-100'\n",
    "if not os.path.exists(model_path):\n",
    "    model_path = './models/k2-fsa-OmniVoice'\n",
    "\n",
    "print(f'📡 Loading OmniVoice checkpoint from: {model_path}...')\n",
    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
    "model = OmniVoice.from_pretrained(\n",
    "    model_path,\n",
    "    torch_dtype=torch.float16 if device == 'cuda' else torch.float32,\n",
    "    attn_implementation='sdpa'\n",
    ").to(device)\n",
    "\n",
    "# Test Prompts across target persona languages\n",
    "prompts = {\n",
    "    'English': 'Hello, I am MTN Zigi. I can assist you with inquiries as regards your bundle and airtime.',\n",
    "    'Nigerian_Pidgin': 'Hello, I be MTN Zigi. I fit help you check your bundle and airtime mata.',\n",
    "    'Yoruba': 'Ẹ lẹ́ẹ̀ o, èmi ni MTN Zigi. Mo lè ràn yín lọ́wọ́ lórí ọ̀rọ̀ bọ́ńdù àti iye owó ìpè yín.',\n",
    "    'Igbo': 'Ndéèwó, mụ bụ MTN Zigi. Nwere m ike inyere gị aka maka ajụjụ gbasara bọndụ gị.',\n",
    "    'Hausa': 'Barka dai, ni ne MTN Zigi. Zan iya taimaka maka da bayani game da kundin tsarinka.',\n",
    "    'Swahili': 'Hujambo, mimi ni MTN Zigi. Naweza kukusaidia kuhusu vifurushi na salio lako.',\n",
    "    'isiZulu': 'Sawubona, ngingu MTN Zigi. Ngingakusiza ngemibuzo mayelana namabhande e-data.',\n",
    "    'Persian': 'سلام، من ام‌تی‌ان زیگی هستم. می‌توانم به شما در زمینه بسته‌های اینترنت کمک کنم.'\n",
    "}\n",
    "\n",
    "ref_audio = './data/audio_clean/xh_000023.wav'\n",
    "ref_text = 'isantya se-802.11n sikhawuleze kakhulu kunesabangaphambili abaphumeleleyo be-600mbit s'\n",
    "\n",
    "os.makedirs('outputs/synthesized_samples', exist_ok=True)\n",
    "with torch.inference_mode():\n",
    "    for lang, text in prompts.items():\n",
    "        print(f'🎙️ Synthesizing {lang}...')\n",
    "        audio = model.generate(\n",
    "            text=text,\n",
    "            ref_audio=ref_audio,\n",
    "            ref_text=ref_text,\n",
    "            num_step=24,\n",
    "            guidance_scale=1.0,\n",
    "            class_temperature=0.7,\n",
    "            postprocess_output=False\n",
    "        )\n",
    "        wav = audio[0]\n",
    "        peak = np.abs(wav).max()\n",
    "        if peak > 1e-6:\n",
    "            wav = wav / peak * 0.9\n",
    "        out_path = f'outputs/synthesized_samples/zigi_{lang.lower()}_interactive.wav'\n",
    "        sf.write(out_path, wav, 24000)\n",
    "        rms = float(np.sqrt(np.mean(wav**2)))\n",
    "        print(f'  ✓ Generated: {out_path} (RMS: {rms:.4f})')\n",
    "        display(Audio(wav, rate=24000))\n"
]

nb["cells"][13]["source"] = cell13_source

with open("voice_rtx2060_complete.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Updated voice_rtx2060_complete.ipynb Cell 13 successfully!")

