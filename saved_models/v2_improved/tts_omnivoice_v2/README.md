# Multilingual TTS Model (Version 2 - Master Studio Edition)

## Key Enhancements over V1 Baseline:
1. **Unified Brand Persona (MTN Zigi)**:
   - Preserves the authoritative, warm 120-128 Hz baritone persona across ALL languages.
   - Eliminates voice fragmentation (all languages sound like the exact same brand voice).
2. **Native Nigerian English (`en_ng`) First-Class Citizen**:
   - Added clean Nigerian English dataset (`benjaminogbonna/nigerian_accented_english_dataset`).
   - Powered by an authentic 119 Hz Nigerian English male baritone anchor.
3. **Broadcast Studio DSP Mastering**:
   - 65 Hz Butterworth high-pass rumble filter.
   - Adaptive silence trimming at boundaries.
   - Peak normalization to -1.0 dBFS (0.89 amplitude) with soft-clipping headroom.
4. **Decoupled Natural Hyperparameters**:
   - `guidance_scale: 1.05` (natural vocal tract formant resonance without robotic over-compression)
   - `class_temperature: 0.50` (natural human prosody and micro-intonation)
   - `num_step: 28` (optimum diffusion codebook fidelity)
   - `speed: 1.00` (relaxed, authoritative customer-service cadence)
