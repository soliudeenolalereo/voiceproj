"""
Multilingual Voice Synthesis Engine V2 (Production RTX 2060)
============================================================
Key Improvements over V1:
1. Native Nigerian English (en_ng) First-Class Support.
2. Regional Reference Anchor Routing:
   - Nigerian English: ref_en_ng.wav (Authentic Nigerian English accent)
   - Nigerian Pidgin: ref_pcm.wav (Authentic Lagos Pidgin cadence)
   - Yoruba: yo_000000.wav (Tonal Nigerian Yoruba voice)
   - Igbo: ig_000000.wav (Tonal Nigerian Igbo voice)
   - Hausa: ha_000000.wav (Northern Nigerian Hausa voice)
   - Swahili: ref_sw.wav (East African Swahili voice)
   - isiZulu: zu_000000.wav (South African Nguni voice)
   - Persian: fa_anchor (Deliberate conversational Farsi voice)
3. Precision Decoding Hyperparameters to Eliminate Word Omission:
   - class_temperature: 0.10 (eliminates stochastic word drops & swallowed phonemes)
   - guidance_scale: 1.80 (strong text classifier-free guidance)
   - num_step: 36 (higher acoustic codebook fidelity)
   - speed: 0.95 (natural, clear, customer-service cadence)
   - language: explicit language codes passed to model.generate()
"""

import os
import sys
import json
import numpy as np
import soundfile as sf
import torch

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

try:
    from omnivoice import OmniVoice
except ImportError:
    from omnivoice.models.omnivoice import OmniVoice

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(BASE_DIR, "data", "reference_anchors")
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio_clean")


class TTSEngineV2:
    def __init__(self, model_path: str = "models/checkpoints_rtx2060_foundation/checkpoint-100", device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        print(f"Loading TTS V2 Model from: {self.model_path} onto {self.device}...")
        self.model = OmniVoice.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            attn_implementation="sdpa"
        ).to(self.device)
        self.model.eval()
        print("✓ TTS V2 Engine Ready.")

    def get_regional_anchor(self, lang_code: str):
        """Returns the optimal reference audio and text for the target dialect."""
        anchors = {
            "en_ng": {
                "audio": os.path.join(REF_DIR, "ref_en_ng.wav"),
                "text": "Closing the Google assistant app prevents it from working with your headphones."
            },
            "pcm": {
                "audio": os.path.join(REF_DIR, "ref_pcm.wav"),
                "text": "Hello, I be MTN Zigi. I fit help you check your bundle and airtime mata."
            },
            "yo": {
                "audio": os.path.join(AUDIO_DIR, "yo_000000.wav"),
                "text": "Ẹ lẹ́ẹ̀ o, èmi ni MTN Zigi. Mo lè ràn yín lọ́wọ́ lórí ọ̀rọ̀ bọ́ńdù yín."
            },
            "ig": {
                "audio": os.path.join(AUDIO_DIR, "ig_000000.wav"),
                "text": "Ndéèwó, mụ bụ MTN Zigi. Nwere m ike inyere gị aka maka ajụjụ bọndụ gị."
            },
            "ha": {
                "audio": os.path.join(AUDIO_DIR, "ha_000000.wav"),
                "text": "Barka dai, ni ne MTN Zigi. Zan iya taimaka maka da bayani game da intanet."
            },
            "sw": {
                "audio": os.path.join(REF_DIR, "ref_sw.wav"),
                "text": "Hujambo, mimi ni MTN Zigi. Naweza kukusaidia kuhusu vifurushi na salio lako."
            },
            "zu": {
                "audio": os.path.join(AUDIO_DIR, "zu_000000.wav"),
                "text": "Sawubona, ngingu MTN Zigi. Ngingakusiza ngemibuzo mayelana namabhande e-data."
            },
            "fa": {
                "audio": None,  # Use Voice Design Mode for Persian
                "text": None
            }
        }

        anchor = anchors.get(lang_code, anchors.get("en_ng"))
        if anchor["audio"] and os.path.exists(anchor["audio"]):
            return anchor["audio"], anchor["text"]
        return None, None

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        lang_code: str = "en_ng",
        output_path: str = None,
        speed: float = 0.95,
        num_step: int = 36,
        guidance_scale: float = 1.8,
        class_temperature: float = 0.10
    ) -> np.ndarray:
        """
        Synthesizes speech with full word enunciation and regional accent authenticity.
        """
        ref_audio, ref_text = self.get_regional_anchor(lang_code)

        # Map language code for OmniVoice internal prompt
        omni_lang_map = {
            "en_ng": "en",
            "pcm": "en",
            "yo": "yo",
            "ig": "ig",
            "ha": "ha",
            "sw": "sw",
            "zu": "zu",
            "fa": "fa"
        }
        target_lang = omni_lang_map.get(lang_code, "en")

        gen_kwargs = {
            "text": text,
            "language": target_lang,
            "speed": speed,
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "class_temperature": class_temperature,
            "postprocess_output": False,
        }

        if ref_audio is not None and ref_text is not None:
            gen_kwargs["ref_audio"] = ref_audio
            gen_kwargs["ref_text"] = ref_text
        elif lang_code == "fa":
            gen_kwargs["instruct"] = "female, young adult"

        audios = self.model.generate(**gen_kwargs)
        wav = audios[0]

        # Studio peak normalization to -0.92 dBFS (0.90 amplitude)
        peak = np.abs(wav).max()
        if peak > 1e-6:
            wav = wav / peak * 0.90

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            sf.write(output_path, wav, 24000)

        return wav
