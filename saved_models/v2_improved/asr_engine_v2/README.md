# Multilingual ASR Engine (Version 2 - Dialect-Aware)

## Key Enhancements over V1:
1. **Dialect-Aware Routing**: Routes English, Nigerian English, Nigerian Pidgin, Persian, Swahili, Hausa, and Yoruba through optimized token tables.
2. **Anti-Hallucination & Anti-Repetition Guards**:
   - `repetition_penalty: 1.35` (cures Whisper's infinite repetition loops)
   - `no_repeat_ngram_size: 3` (stops stuttering)
   - `temperature: 0.0` (eliminates random token hallucinations)
3. **MMS Dual Engine Fallback**: Integrates Meta MMS acoustic adapters for low-resource African languages (Igbo `ibo`, isiZulu `zul`, Akan `aka`, Luganda `lug`).
