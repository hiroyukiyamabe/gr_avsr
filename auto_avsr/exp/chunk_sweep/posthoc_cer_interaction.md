# POST HOC: CER interaction, gated vs static, net of the clean offset

POST HOC, exploratory; not pre-registered, not part of any verdict.
I(c) = [gated - static](c) - [gated - static](placebo), CER pt; n = 400 utterances;
paired bootstrap B = 10000, np.random.default_rng(0), one resample shared by all
arms and conditions; 95 % percentile CI. The CI covers test-utterance sampling
only, not training-seed variance.

Clean (placebo) gap: +0.39 pt [+0.09, +0.69]

| condition | gated - static | minus clean gap = I(c) [95 % CI] |
|---|---:|---|
| audio_chunks | +0.88 | +0.49 [-0.09, +1.08] |
| video_chunks | +0.41 | +0.02 [-0.30, +0.33] |
| both_chunks | +0.95 | +0.56 [-0.08, +1.21] |

Command (from auto_avsr/): `python posthoc_cer_interaction.py --dir exp/chunk_sweep`
