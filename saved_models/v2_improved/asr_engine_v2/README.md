# Dialect-Aware ASR Model (Version 2 - Improved)

## Key Enhancements over V1:
1. **Dialect-Aware Routing**: Direct Whisper language forcing for `en_ng`, `fa`, `sw`, `ha`, `yo`.
2. **Anti-Hallucination & Anti-Repetition Guards**:
   - `temperature = 0.0`
   - `no_repeat_ngram_size = 4`
   - `repetition_penalty = 1.20`
3. **Meta MMS Support**: Acoustic CTC adapters for low-resource African languages missing from Whisper.
