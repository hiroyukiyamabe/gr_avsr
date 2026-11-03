# Handover: gr\_avsr changes after the strong-form review

Oct 24, 2026 · @Hiroyuki Yamabe

This is for the local Claude Code agent working in `gr_avsr` at HEAD `6b11b45`. It lists six documentation, analysis and hygiene tasks to do now, and four decisions that Hiro makes before anything else changes. No task trains or evaluates a model, and none changes the §2.3/§2.4 verdicts or the code paths that produced them.

## Ground rules

CLAUDE.md stays binding; nothing in this handover overrides it. In particular:

1. **No training, no evaluation runs, no GPU.** Task 3 reads existing outputs under `auto_avsr/exp/chunk_sweep/` only.
2. **Do not edit the pre-registered code or settings:** `analyze_chunk_sweep.py`, `analyze_gate_partA.py`, `eval_chunk_sweep.py`, `datamodule/chunk_corruption.py`, the `decode.chunk` block of `configs/decode/default.yaml`, `ModalityGate`, `E2E.fuse`. New analysis goes in new files labelled POST HOC.
3. **Mark superseded content instead of deleting it** (CLAUDE.md §8): strike the old wording with `~~…~~` and add the correction with the date 2026-10-24 and a one-line reason. The one exception is the CLAUDE.md instruction block in Task 4, which is a leftover instruction, not a record.
4. **Verify before you edit.** Each task names a file and line from HEAD `6b11b45`. Confirm the text is still there and the claim in this handover matches the code before changing anything. If it doesn't match, stop and report.
5. **One commit per task,** with a message that says what the commit does. Never amend or force-push anything already on GitHub.
6. **Shell is PowerShell 5.1** (CLAUDE.md §9): no `&&`, trailing backtick for continuation, run from `auto_avsr\` with the venv active.

## Task 1 — Correct the training-corruption statements (RESULTS.md)

RESULTS.md says neither arm saw chunk-wise corruption in training, but the inherited `AdaptiveTimeMask` gives both arms a crude form of it.

**Verify first** in `datamodule/transforms.py`:

- The train `AudioTransform` applies `AdaptiveTimeMask(sample_rate * 2 // 5, sample_rate)` *before* `AddNoise`. Masked spans (up to 0.4 s per second) become pure babble at the utterance SNR, or silence when the utterance is clean.
- The train `VideoTransform` applies `AdaptiveTimeMask(10, 25)` before `Normalize`. Up to 10 frames per 25 are zeroed, which become a constant after normalisation.
- The two masks are drawn independently per stream.

**Edits:**

1. **RESULTS.md:566** (§2.4, Part B CER finding 1, third bullet). Strike "neither arm saw chunk-wise corruption in training (babble, one SNR per utterance)". Replace with: neither arm saw *this* corruption (white noise, blur, occlusion) in training. Both did see chunk-wise, per-stream zeroing from the inherited `AdaptiveTimeMask`, plus babble at one SNR per utterance.
2. **RESULTS.md:140** (§2.1 finding 4). Strike "audio always present". Note that time masking replaced up to 0.4 s per second of audio with babble or silence; audio was present in every utterance, not in every frame. Leave the rest of the finding, including its "hypothesis" label.
3. **RESULTS.md §2.4, "Scope of claims".** Add one bullet: the video response (dV) cannot be read as the gate generalising to an unseen corruption family, because blanked video frames are a training analogue. Eval types (white noise, blur, eval-half occluders) remain disjoint from time masking, so CLAUDE.md §4 holds.

Do not touch any number or the verdict.

## Task 2 — Reword the "mechanism confirmed" verdicts

Both places now overclaim under CLAUDE.md §5: the per-frame reliability reading failed in §2.4.

1. **RESULTS.md:56** (§2 verdict). Strike "mechanism confirmed, benefit absent". Replace with: "utterance-level response confirmed (§2.3); per-frame interpretation not validated (§2.4); benefit absent." Keep the July table and the rest of the paragraph.
2. **REPRODUCTION\_PLAN.md:519** ("mechanism CONFIRMED"). Strike it and add the same correction with a pointer to RESULTS §2.4.
3. Search all of RESULTS.md, REPRODUCTION\_PLAN.md and SYSTEM\_STATUS.md for other claims that the gate is "interpretable", "confirmed" or a "reliability estimate" at the frame level. List any hits in your report; correct them the same way.

## Task 3 — Post-hoc CER interaction bootstrap

The question: is the gated arm worse *because of* the chunk corruption, or just by its clean offset (+0.39 pt)? The raw twin gaps can't say.

**New script** `auto_avsr/posthoc_cer_interaction.py`. Docstring opens with "POST HOC, exploratory; not pre-registered, not part of any verdict", like `posthoc_chunk_placebo.py`. Do not modify `analyze_chunk_sweep.py`.

- **Inputs:** `rows.csv` of `rohan_avsr_3448_static` and `rohan_avsr_3448_gated` under placebo, audio\_chunks, video\_chunks and both\_chunks in `exp/chunk_sweep/`. Assert the four row sets have identical `utt_id` order.
- **Estimand,** in CER pt, per condition c ∈ {audio\_chunks, video\_chunks, both\_chunks}: I(c) = \[CER\_gated(c) − CER\_static(c)\] − \[CER\_gated(placebo) − CER\_static(placebo)\]. CER = Σ edit\_distance / Σ ref\_len over the utterances in the sample, × 100.
- **Bootstrap:** B = 10,000 utterance resamples, `np.random.default_rng(0)`, ONE resample shared by all arms and conditions. Report the point estimate on the full set and the 95 % percentile CI.
- **Sanity check:** the full-set point estimates must equal the raw gap minus the clean gap from RESULTS §2.4, about +0.49 (audio), +0.02 (video), +0.56 (both). Abort if they differ by more than 0.01 pt.
- **Output:** print the table and write `exp/chunk_sweep/posthoc_cer_interaction.md` as a new file; refuse to overwrite.

**RESULTS.md:** add a short block after the Part B CER findings, headed POST HOC like the placebo block, with the table and command. Then annotate (do not rewrite) CER finding 1:

- If an interaction is ≥ 0.5 pt with a CI excluding 0, the corruption-specific harm holds for that condition.
- Otherwise add: the level gap holds, but a corruption-specific harm beyond the clean offset is not established (CLAUDE.md §8 0.5 pt rule).

State in the block that the CI covers test-utterance sampling only, not training-seed variance.

## Task 4 — CLAUDE.md cleanup

1. **§5.2 heading:** change "Current task" to "Done 2026-10-23 (RESULTS.md §2.4): FAIL under the pre-registered rule". Leave the body unchanged; it is the pre-registration record.
2. **Delete the "Related edits elsewhere in CLAUDE.md" block** (from the `---` after §5.2 "Scope of claims" up to `## 6. Pseudo-labelling`). Confirm first that both edits it asks for are already in the file (the §5.1 heading and the §8 bootstrap sentence). It is a leftover instruction, not a record, so deleting it is correct.
3. **§2, "Input augmentation" bullet:** add one sentence. The inherited recipe already applies `AdaptiveTimeMask` to both streams independently in training; it stays in every arm, and the AV-RelScore-style augmentation is added on top of it.

No other wording in CLAUDE.md changes. The next task (chunk-wise training augmentation) is not specified here; see Decisions.

## Task 5 — Note on commit 6b11b45

Its message, "Enforce a harsher test", suggests the rule changed after the results. It didn't. Add one line to the strong-form DONE entry in REPRODUCTION\_PLAN.md:

> Commit `6b11b45` ("Enforce a harsher test") only records the results and adds `posthoc_chunk_placebo.py`; no setting, threshold or analysis code changed between `dba2502` and the runs.

Verify with `git diff dba2502 6b11b45 --stat -- auto_avsr/` before writing it: only `lightning_logs/version_70–81` and the post-hoc script should appear. Do not rewrite history.

## Task 6 — Repo hygiene and versioning the analysis outputs

Untrack what shouldn't be public, and track the small outputs the results depend on. Local files stay on disk throughout (`git rm --cached`, never plain `git rm`).

**Untrack and add to `.gitignore`:**

| Path | Why |
| --- | --- |
| `papers/*.pdf` (27 files, \~99 MB) | Published papers; redistribution in a public repo. Keep `papers/paperlist.xlsx`. |
| `avsr/` (\~2.2 MB) | Windows venv, tracked before it was ignored. |
| `whole_claude_code_promt.txt`, `.claude/settings.local.json` | Local agent and editor state. |
| `crash.txt.txt`, `failure_output_second.txt`, `training_failure_output.txt`, `vsr_training_error_260702.txt` | Crash logs. |
| `evaluation2.txt`, `evaluation_results.txt`, `06_27_fine_tuning.txt` | Old logs; untrack only if no doc cites them. |

Before untracking any log, grep RESULTS.md, REPRODUCTION\_PLAN.md and SYSTEM\_STATUS.md for its name. If a doc cites it as provenance, keep it tracked and list it in your report.

**Track the analysis outputs.** `auto_avsr/exp/` is ignored as a whole, and git cannot re-include files under an ignored directory. So change the pattern to `auto_avsr/exp/*`, then re-include:

- `exp/per_utt_sweep/`: the per-run CSVs, `*_meta.json`, `*_traces.json`, `summary.md`/`.json`, `partA_*`, plots, log.
- `exp/chunk_sweep/`: each run's `rows.csv`, `meta.json`, `frames.npz`; `summary.md`/`.json`; plots; log; the Task 3 output.
- `exp/snr_sweep_*.csv` and the sweep plots and logs.

Never `.pth`, `.ckpt` or any training directory. Check the total size first and report it; stop and ask if it exceeds 50 MB.

Untracking does not remove the PDFs from git history. Purging them is Decision 4.

## Decisions Hiro makes first

The agent does not act on any of these until Hiro says so in chat. If asked to, it implements exactly the option chosen.

**Decision 1 — Amend the strong-form readout for the repeat.** Recommended: keep chunk counts, lengths, placement and corruption types unchanged, and make the placebo-corrected per-utterance difference-in-differences the primary measure, with the same CI and ≥ 75 % share criteria. The as-registered rule is still computed and reported beside it. If approved:

- Add the amendment as a new dated sub-entry in CLAUDE.md §5.2 and in REPRODUCTION\_PLAN.md, with the reason (the placebo offset comes from mask placement against the clean gate's speech/pause pattern; RESULTS §2.4).
- Implement it as an *additional* verdict block in a new analysis file or a clearly separate function. The as-registered verdict code stays byte-identical.
- State that the amendment was informed by the current data, on which the corrected measure would pass (82 % / 93 %). The current runs remain FAIL; only the post-augmentation repeat is confirmatory under the amended rule.
- Commit it before any augmented training run starts.

**Decision 2 — Occluder split.** It is split by image, so train and eval halves share COCO categories. Option A: keep it and document the limitation in `configs/decode/default.yaml` and RESULTS §2.4 divergences. Option B: re-split by category, which changes the pre-registered eval corruption and breaks the "repeat unchanged" comparison. Recommended: A. Either way, decide before training uses the train half.

**Decision 3 — Spec for chunk-wise training augmentation.** Not part of this handover. Hiro writes it into CLAUDE.md first: corruption types (train-only list in `decode.chunk`), probabilities, per-stream independence, flags off by default, and all four arms retrained under equal treatment.

**Decision 4 — Purge the PDFs from git history.** Needs a history rewrite and a force push, which changes every commit hash cited in RESULTS and REPRODUCTION\_PLAN. Not recommended unless redistribution is a real concern; if done, every cited hash must be remapped.

## Done when

- [ ] Task 1: RESULTS.md:566 and :140 struck and corrected; the dV scope bullet added; no number or verdict changed.
- [ ] Task 2: both "mechanism confirmed" verdicts struck and corrected; any further hits listed in the report.
- [ ] Task 3: `posthoc_cer_interaction.py` runs on CPU, its point estimates match +0.49 / +0.02 / +0.56 within 0.01 pt, the POST HOC block is in RESULTS §2.4, and CER finding 1 is annotated.
- [ ] Task 4: §5.2 marked done, the leftover block deleted, the time-masking sentence added to §2.
- [ ] Task 5: the one-line note on `6b11b45` added after verifying the diff.
- [ ] Task 6: PDFs, venv, agent state and crash logs untracked and ignored; analysis outputs tracked; total size reported.
- [ ] `git diff dba2502 HEAD -- auto_avsr/analyze_chunk_sweep.py auto_avsr/eval_chunk_sweep.py auto_avsr/datamodule/chunk_corruption.py` is empty.
- [ ] One commit per task; the tree is clean at the end.
- [ ] A short report back to Hiro: commits made, anything that didn't match this handover, and the Task 3 table.
