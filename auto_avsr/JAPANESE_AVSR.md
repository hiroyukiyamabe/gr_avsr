# Japanese AVSR via Cross-Lingual Transfer Learning

Baseline implementation of Kondo & Tamura, *"Audio-Visual Speech Recognition
Based on Cross-Lingual Transfer Learning"* (APSIPA ASC 2025) on top of Auto-AVSR.

> **See also:** [`PIPELINE.md`](PIPELINE.md) — copy-paste command reference for
> the full train/eval pipeline. [`../SYSTEM_STATUS.md`](../SYSTEM_STATUS.md) —
> software environment, code modifications, and the open hardware-stability
> issue (run with `+data.num_workers=0` until that is fixed).

**Idea (framework (c) in the paper):** inherit the **front-end encoders** from
high-performance English checkpoints and **fine-tune** them on a small Japanese
corpus, while training the Conformer back-ends, MLP fusion, CTC projection and
Transformer decoder **from scratch**. The decoder vocabulary is reduced from
5,000 (English subwords) to ~87 (Japanese Katakana characters).

```
                         inherited + fine-tuned        from scratch
 Image seq ─▶ [3D conv + 2D ResNet-18] ─▶ [12x Conformer] ─┐
                                                            ├▶ MLP ─▶ CTC + Transformer decoder
 Audio wav ─▶ [1D ResNet-18]           ─▶ [12x Conformer] ─┘
```

## 1. Extract the English front-ends

The pre-trained checkpoints in `../original_checkpoints/` are full models whose
front-end weights are stored under `frontend.*`. The training code expects a
front-end-only checkpoint keyed `trunk.*` / `frontend3D.*` (wrapped in
`model_state_dict`). Convert them:

```bash
cd ../module_extractor
python extractor.py
# -> frontends/frontend_visual_trlrs3.pth   (from vsr_trlrs3_base.pth)
# -> frontends/frontend_audio_trlrs3.pth    (from asr_trlrs3_base.pth)
```

(Use `--inspect` to preview keys, or `--src/--dst` for other checkpoints such as
`vsr_trlrs3vox2_base.pth`.)

## 2. Tokenizer / vocabulary

Japanese uses a **character-based Katakana** tokenizer (`JapaneseTextTransform`
in `datamodule/transforms.py`) — no SentencePiece model needed. The placeholder
inventory in `spm/ja_char/ja_char_units.txt` yields exactly **87** output classes
(`<blank>` + 84 chars + `<unk>` + `<eos>`), matching the paper.

> **Placeholder:** regenerate this units file from the final corpus once chosen.
> If you prefer SentencePiece over pure character splitting, train an spm model on
> Katakana transcripts and adapt `get_text_transform`.

## 3. Prepare the ROHAN4600 corpus

The corpus lives in `datasets/ROHAN4600/` with three sibling folders:

```
ROHAN4600/
├── ROHAN4600_zumndamon_normal_label/            # HTK phoneme alignment (.lab)
├── ROHAN4600_zumndamon_normal_picture/          # batched lip-ROI videos + landmark csv
│     ROHAN4600_0001-0400_LFROI/LFROI_ROHAN4600_0001.mp4 ...
└── ROHAN4600_zumndamon_normal_synchronized_wav/ # 96 kHz mono audio
```

Because the videos are **already mouth/lip ROIs** (`LFROI` crops),
`preparation/preprocess_rohan.py` does *not* run face detection — it copies the
ROI video through, resamples the audio (96 kHz → 16 kHz by default), and converts
the phoneme `.lab` files to Katakana (`preparation/rohan_phonemes.py`, which
groups consonant+vowel into morae, maps `N`→ン / `cl`→ッ, and drops `sil`/`pau`).

```bash
python preparation/preprocess_rohan.py \
    --rohan-root ../datasets/ROHAN4600 \
    --out-dir    ../datasets/rohan_preprocessed \
    --audio-sample-rate 16000
# (defaults already point at ../datasets/ROHAN4600 and ../datasets/rohan_preprocessed)
```

Produces the layout expected by `datamodule/av_dataset.py`:

```
rohan_preprocessed/
├── rohan/{rohan_video_seg/*.mp4(+.wav), rohan_text_seg/*.txt}
└── labels/rohan_{train,val,test}_transcript_lengths.csv
```

`configs/data/dataset/rohan.yaml` already points `root_dir` at
`datasets/rohan_preprocessed`. The 4,600 utterances are split 3,800 / 400 / 400
(train / val / test) by default — `--val-size` / `--test-size` / `--no-shuffle`
override this (the paper used 3,400 / 400 / 400 on a smaller cut).

> Sanity-check the phoneme→Katakana conversion on the raw labels with
> `python preparation/rohan_phonemes.py datasets/ROHAN4600/ROHAN4600_zumndamon_normal_label`.

## 4. Train

Per-modality experiment configs live in `configs/experiment/` and encode the
paper's hyper-parameters (AdamW, cosine schedule, 5-epoch warm-up, peak LR 1e-4;
ASR/VSR: 60 epochs / 1600 max frames; AVSR: 40 epochs / 1000 max frames).

```bash
# VSR — fine-tune the visual front-end
python train.py +experiment=rohan_vsr exp_dir=exp exp_name=rohan_vsr \
    data.dataset.root_dir=/path/to/rohan_preprocessed \
    pretrained_model_path=../module_extractor/frontends/frontend_visual_trlrs3.pth

# ASR — fine-tune the audio front-end
python train.py +experiment=rohan_asr exp_dir=exp exp_name=rohan_asr \
    data.dataset.root_dir=/path/to/rohan_preprocessed \
    pretrained_model_path=../module_extractor/frontends/frontend_audio_trlrs3.pth

# AVSR — fine-tune BOTH front-ends (visual + audio)
python train.py +experiment=rohan_avsr exp_dir=exp exp_name=rohan_avsr \
    data.dataset.root_dir=/path/to/rohan_preprocessed \
    pretrained_visual_frontend_path=../module_extractor/frontends/frontend_visual_trlrs3.pth \
    pretrained_audio_frontend_path=../module_extractor/frontends/frontend_audio_trlrs3.pth
```

### Competitive strategies from the paper

| Paper scheme | How to run |
|---|---|
| (a) from scratch | omit all `pretrained_*` paths and `transfer_frontend` |
| (b) frozen front-end | (c) flags + freeze `encoder.frontend` / `aux_encoder.frontend` params *(see note)* |
| (c) fine-tune front-end **[proposed]** | the commands above (`transfer_frontend=true`) |

> Note: strategy (b) (freezing) is not wired as a flag yet; add a `freeze_frontend`
> option that sets `requires_grad=False` on the front-end parameters if you want to
> reproduce that ablation.

## 5. Evaluate

The test loop reports **CER** automatically when `data.lang=ja` (character-level
edit distance), as used in the paper.

```bash
python eval.py +experiment=rohan_avsr \
    data.dataset.root_dir=/path/to/rohan_preprocessed \
    pretrained_model_path=exp/rohan_avsr/model_avg_10.pth \
    transfer_frontend=false
```

`transfer_frontend=false` is required so the model constructor loads the full
averaged checkpoint directly rather than expecting a front-end-only file. Swap
`+experiment=` and the checkpoint path for the `rohan_vsr` / `rohan_asr` runs.
See `PIPELINE.md` §2 for the per-modality commands.

## Notes on audio sampling rate

The paper uses **19,200 Hz** for AVSR (vs. 16,000 Hz for ASR) so that, at 25 fps,
each video frame aligns to a whole number of audio samples (19200/25 = 768),
reducing zero-padding during A/V synchronisation. To follow that setting,
preprocess audio at 19,200 Hz and set `data.dataset.rate_ratio=768`. The defaults
here use 16,000 Hz / 640, which matches the model's native stride.

## What is a placeholder vs. done

- **Done & verified:** front-end extraction, Katakana tokenizer (vocab 87),
  dual front-end transfer for AVSR, language-aware CER metric, configs,
  ROHAN4600 preparation (`preprocess_rohan.py` + `rohan_phonemes.py`) and the
  `root_dir` wired in `rohan.yaml`.
- **Note:** the Katakana units file (`spm/ja_char/ja_char_units.txt`) covers every
  phoneme observed in ROHAN4600; regenerate it if you swap in a different corpus.
