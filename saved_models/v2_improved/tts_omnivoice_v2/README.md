# Multilingual TTS Model (Version 2 - Improved)

## Key Enhancements over V1:
1. **Nigerian English (`en_ng`) First-Class Citizen**: Added authentic Nigerian English dataset (`benjaminogbonna/nigerian_accented_english_dataset`) with studio DSP cleaning.
2. **Multi-Region Reference Anchor Routing**: Replaced single Xhosa prompt with regional anchors for Nigerian English, Pidgin, Swahili, Yoruba, Igbo, Hausa, Zulu, and Persian.
3. **Anti-Word-Drop Hyperparameters**:
   - `class_temperature: 0.10` (prevents swallowed phonemes and dropped words)
   - `guidance_scale: 1.80` (strong text conditioning)
   - `num_step: 36` (high acoustic resolution)
   - `speed: 0.95` (calm, authoritative customer-service cadence)
