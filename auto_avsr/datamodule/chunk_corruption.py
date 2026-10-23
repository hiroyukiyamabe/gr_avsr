"""Seeded, eval-only chunk-wise corruption with a known per-frame mask.

Phase 2c strong form (CLAUDE.md section 5.2, Part B), AV-RelScore-style
(Hong et al., CVPR 2023): random chunks of the audio and/or video stream of each
test utterance are corrupted, and the per-frame boolean mask is returned so the
gate can be read against it. Off unless decode.chunk_condition is set; never
used for training or validation.

Every draw is a function of (noise_seed, utterance ID, stream) only: every arm,
modality and condition sees the same masks, and the same corrupted input for a
corrupted stream. The two streams draw independently, so their chunks overlap in
some utterances and alternate in others. One frame = 40 ms = 640 samples at
16 kHz = one video frame at 25 fps.

Settings are pre-registered in configs/decode/default.yaml (decode.chunk).
"""
import csv
import hashlib
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image

from .transforms import AddNoise


# condition -> (audio corrupted, video corrupted). In "placebo" both masks are
# drawn and logged exactly as in "both_chunks", but nothing is applied.
CONDITIONS = {
    "audio_chunks": (True, False),
    "video_chunks": (False, True),
    "both_chunks": (True, True),
    "placebo": (False, False),
}
# Per-frame video corruption code (bit flags; 3 = overlapping chunks of both types).
VIDEO_TYPE_BITS = {"occlusion": 1, "blur": 2}

AUTO_AVSR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path):
    """Relative chunk-config paths are relative to auto_avsr/, not the cwd."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(AUTO_AVSR_DIR, path))


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_text_file(path):
    """SHA-256 with CRLF normalised to LF, so git's autocrlf cannot change it."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def spans_to_mask(spans, n_frames):
    mask = np.zeros(n_frames, dtype=bool)
    for start, length in spans:
        mask[start:start + length] = True
    return mask


class ChunkCorruption:
    """Draws the chunk plan of an utterance and applies it to either stream.

    Draw order (fixed; changing it changes every mask):
      audio: rng = Random(f"{seed}:{key}:audio"); n = randint(*n_chunks);
             per chunk: length = randint(*chunk_frames), start = randint(0, T - length)
      video: rng = Random(f"{seed}:{key}:video"); n = randint(*n_chunks);
             per chunk: length, start (as audio), then type = choice(types),
             occluder = randrange(n_eval_occluders), dx = randint(-jx, jx),
             dy = randint(-jy, jy) -- all drawn for every chunk, whatever its type.
    Chunks of one stream may overlap; the mask is their union.
    """

    def __init__(self, cfg, condition, seed, sample_rate=16000, rate_ratio=640):
        if condition not in CONDITIONS:
            raise ValueError(f"unknown chunk condition {condition!r}; expected one of {sorted(CONDITIONS)}")
        if seed is None:
            raise ValueError("chunk corruption needs decode.noise_seed")
        if rate_ratio != 640 or sample_rate != 16000:
            raise ValueError("chunk corruption assumes 16 kHz audio, 640 samples per 25 fps frame")
        self.condition = condition
        self.audio_on, self.video_on = CONDITIONS[condition]
        self.seed = seed
        self.rate_ratio = rate_ratio
        self.n_chunks = tuple(cfg.n_chunks)
        self.chunk_frames = tuple(cfg.chunk_frames)

        # Audio: the seeded eval AddNoise at the chunk SNR, i.e. the same crop and
        # level as the per-utterance sweep at that SNR (SNR over the whole clean
        # utterance), kept only inside the chunks.
        self.add_noise = AddNoise(
            noise_filename=resolve(cfg.audio.noise_filename),
            snr_target=cfg.audio.snr,
            sample_rate=sample_rate,
            seed=seed,
        )

        video = cfg.video
        self.video_types = list(video.types)
        if not set(self.video_types) <= set(VIDEO_TYPE_BITS):
            raise ValueError(f"unknown video corruption types {self.video_types}")
        self.blur_kernel = int(video.blur_kernel)
        self.blur_sigma = float(video.blur_sigma)
        self.occluder_size = int(video.occluder_size)
        self.occluder_center = tuple(video.occluder_center)
        self.occluder_jitter = tuple(video.occluder_jitter)
        self.occluder_stems, self.occluders = self._load_occluders(video)

    def _load_occluders(self, video):
        """Eval-half occluders only, verified against the committed split file."""
        split_path = resolve(video.occluder_split)
        digest = sha256_text_file(split_path)
        if digest != video.occluder_split_sha256:
            raise RuntimeError(f"{split_path}: sha256 {digest} != decode.chunk.video.occluder_split_sha256")
        with open(split_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        eval_rows = [r for r in rows if r["split"] == "eval"]
        train_stems = {r["stem"] for r in rows if r["split"] == "train"}
        if not eval_rows or train_stems & {r["stem"] for r in eval_rows}:
            raise RuntimeError("occluder split must have a non-empty eval half disjoint from train")
        root = resolve(video.occluder_dir)
        size = self.occluder_size
        stems, images, alphas = [], [], []
        for r in eval_rows:
            image_path = os.path.join(root, "object_image_sr", r["stem"] + ".jpeg")
            mask_path = os.path.join(root, "object_mask_x4", r["stem"] + ".png")
            if sha256_file(image_path) != r["image_sha256"] or sha256_file(mask_path) != r["mask_sha256"]:
                raise RuntimeError(f"occluder {r['stem']} does not match the split file hashes")
            # As AV-RelScore: zero the image outside the mask, resize image and
            # mask to a size x size square (Lanczos), mask / 255 is a soft alpha.
            image = np.asarray(Image.open(image_path).convert("RGB"))
            mask = np.asarray(Image.open(mask_path).convert("L"))
            image = image * (mask[..., None] > 0)
            image = Image.fromarray(image.astype(np.uint8)).resize((size, size), Image.LANCZOS)
            mask = Image.fromarray(mask).resize((size, size), Image.LANCZOS)
            images.append(torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1))
            alphas.append(torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0).clamp(0.0, 1.0))
            stems.append(r["stem"])
        return stems, list(zip(images, alphas))

    def _spans(self, rng, n_frames):
        if n_frames < self.chunk_frames[1]:
            raise ValueError(f"utterance of {n_frames} frames is shorter than the longest chunk")
        spans = []
        for _ in range(rng.randint(*self.n_chunks)):
            length = rng.randint(*self.chunk_frames)
            start = rng.randint(0, n_frames - length)
            spans.append((start, length))
        return spans

    def audio_plan(self, key, n_frames):
        """[(start, length)] in frames."""
        return self._spans(random.Random(f"{self.seed}:{key}:audio"), n_frames)

    def video_plan(self, key, n_frames):
        """[dict(start, length, type, occluder, dx, dy)] in frames / pixels."""
        rng = random.Random(f"{self.seed}:{key}:video")
        plan = []
        for _ in range(rng.randint(*self.n_chunks)):
            length = rng.randint(*self.chunk_frames)
            start = rng.randint(0, n_frames - length)
            plan.append({
                "start": start, "length": length,
                "type": rng.choice(self.video_types),
                "occluder": rng.randrange(len(self.occluders)),
                "dx": rng.randint(-self.occluder_jitter[0], self.occluder_jitter[0]),
                "dy": rng.randint(-self.occluder_jitter[1], self.occluder_jitter[1]),
            })
        return plan

    def audio_mask(self, key, n_frames):
        return spans_to_mask(self.audio_plan(key, n_frames), n_frames)

    def video_mask(self, key, n_frames):
        """(mask, per-frame type bits)."""
        plan = self.video_plan(key, n_frames)
        vtype = np.zeros(n_frames, dtype=np.int8)
        for c in plan:
            vtype[c["start"]:c["start"] + c["length"]] |= VIDEO_TYPE_BITS[c["type"]]
        return vtype > 0, vtype

    def corrupt_audio(self, speech, key, n_frames):
        """speech: T x 1, the model's audio input before the instance layer norm.

        Returns (speech, mask). Inside the mask the samples are those of the
        seeded AddNoise output at the chunk SNR; outside they are untouched.
        ``n_frames`` is the video frame count, which fixes the mask; for ASR the
        raw audio may be up to one frame shorter or longer than 640 * n_frames.
        """
        spans = self.audio_plan(key, n_frames)
        mask = spans_to_mask(spans, n_frames)
        if not self.audio_on:
            return speech, mask
        noisy = self.add_noise(speech, key=key)
        inside = torch.zeros(speech.shape[0], dtype=torch.bool)
        for start, length in spans:
            inside[start * self.rate_ratio:(start + length) * self.rate_ratio] = True
        return torch.where(inside.unsqueeze(1), noisy, speech), mask

    def corrupt_video(self, frames, key):
        """frames: T x 3 x H x W in [0, 1], the model's mouth crop before grayscale.

        Returns (frames, mask, per-frame type bits). Blur: Gaussian, fixed kernel
        and sigma. Occlusion: one eval occluder per chunk, alpha-blended at a fixed
        position for the whole chunk (the mouth ROI is already stabilised).
        """
        n_frames = frames.shape[0]
        mask, vtype = self.video_mask(key, n_frames)
        if not self.video_on:
            return frames, mask, vtype
        frames = frames.clone()
        height, width = frames.shape[-2:]
        size = self.occluder_size
        for c in self.video_plan(key, n_frames):
            chunk = slice(c["start"], c["start"] + c["length"])
            if c["type"] == "blur":
                frames[chunk] = torchvision.transforms.functional.gaussian_blur(
                    frames[chunk], [self.blur_kernel, self.blur_kernel], [self.blur_sigma, self.blur_sigma]
                )
            else:
                image, alpha = self.occluders[c["occluder"]]
                x0 = self.occluder_center[0] + c["dx"] - size // 2
                y0 = self.occluder_center[1] + c["dy"] - size // 2
                xs, xe = max(x0, 0), min(x0 + size, width)
                ys, ye = max(y0, 0), min(y0 + size, height)
                a = alpha[ys - y0:ye - y0, xs - x0:xe - x0]
                o = image[:, ys - y0:ye - y0, xs - x0:xe - x0]
                region = frames[chunk, :, ys:ye, xs:xe]
                frames[chunk, :, ys:ye, xs:xe] = a * o + (1.0 - a) * region
        return frames, mask, vtype
