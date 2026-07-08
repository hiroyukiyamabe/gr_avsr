#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import random

import numpy as np
import sentencepiece
import soundfile as sf
import torch
import torchaudio
import torchvision


NOISE_FILENAME = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "babble_noise.wav"
)

SP_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "unigram",
    "unigram5000.model",
)

DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "unigram",
    "unigram5000_units.txt",
)

# Character-based Japanese (Katakana) dictionary for cross-lingual transfer.
# NOTE: placeholder vocabulary -- regenerate from the final corpus (see
# spm/ja_char/ja_char_units.txt and JAPANESE_AVSR.md).
JA_DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "spm",
    "ja_char",
    "ja_char_units.txt",
)


class FunctionalModule(torch.nn.Module):
    def __init__(self, functional):
        super().__init__()
        self.functional = functional

    def forward(self, input):
        return self.functional(input)


# Module-level functions (not lambdas) so the transform pipelines stay picklable
# for DataLoader workers under Windows' "spawn" start method.
def _scale_pixels(x):
    return x / 255.0


def _instance_layer_norm(x):
    return torch.nn.functional.layer_norm(x, x.shape, eps=1e-8)


def _identity(x):
    return x


class AdaptiveTimeMask(torch.nn.Module):
    def __init__(self, window, stride):
        super().__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        # x: [T, ...]
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        ts = torch.randint(0, self.window, size=(n_mask, 2))
        for t, t_end in ts:
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - t)
            if t_start == t_start + t:
                continue
            t_end += t_start
            cloned[t_start:t_end] = 0
        return cloned


# Default set of signal-to-noise ratios (dB) sampled during training. A very
# large value (999999) is effectively "clean" so a fraction of utterances are
# left (near-)untouched, giving the model a mix of clean and noisy inputs.
DEFAULT_SNR_LEVELS = [-5, 0, 5, 10, 15, 20, 999999]


class AddNoise(torch.nn.Module):
    def __init__(
        self,
        noise_filename=NOISE_FILENAME,
        snr_target=None,
        snr_levels=None,
        noise_prob=1.0,
    ):
        super().__init__()
        # Precedence: an explicit ``snr_target`` (used at eval time to pin a
        # single SNR) wins; otherwise sample from ``snr_levels`` (configurable
        # per experiment); otherwise fall back to the default mix.
        if snr_target is not None:
            self.snr_levels = [snr_target]
        elif snr_levels is not None:
            self.snr_levels = list(snr_levels)
        else:
            self.snr_levels = list(DEFAULT_SNR_LEVELS)
        # Probability that a given utterance has noise applied at all. With
        # ``noise_prob < 1`` the remaining utterances are passed through clean.
        self.noise_prob = noise_prob
        # torchaudio >= 2.9 routes load() through torchcodec; read via soundfile.
        data, sample_rate = sf.read(noise_filename, dtype="float32", always_2d=True)
        self.noise = torch.from_numpy(np.ascontiguousarray(data.T))  # channels x frames
        assert sample_rate == 16000

    def forward(self, speech):
        # speech: T x 1
        # return: T x 1
        if self.noise_prob < 1.0 and random.random() >= self.noise_prob:
            return speech
        speech = speech.t()
        start_idx = random.randint(0, self.noise.shape[1] - speech.shape[1])
        noise_segment = self.noise[:, start_idx : start_idx + speech.shape[1]]
        snr_level = torch.tensor([random.choice(self.snr_levels)])
        noisy_speech = torchaudio.functional.add_noise(speech, noise_segment, snr_level)
        return noisy_speech.t()


class VideoTransform:
    def __init__(self, subset):
        if subset == "train":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(_scale_pixels),
                torchvision.transforms.RandomCrop(88),
                torchvision.transforms.Grayscale(),
                AdaptiveTimeMask(10, 25),
                torchvision.transforms.Normalize(0.421, 0.165),
            )
        elif subset == "val" or subset == "test":
            self.video_pipeline = torch.nn.Sequential(
                FunctionalModule(_scale_pixels),
                torchvision.transforms.CenterCrop(88),
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Normalize(0.421, 0.165),
            )

    def __call__(self, sample):
        # sample: T x C x H x W
        # rtype: T x 1 x H x W
        return self.video_pipeline(sample)


class AudioTransform:
    def __init__(self, subset, snr_target=None, snr_levels=None, noise_prob=1.0,
                 noise_filename=None):
        # ``noise_filename`` only affects eval-time noise (decode.noise_filename,
        # e.g. white noise for the Kondo & Tamura Table IV replica). Training
        # augmentation keeps the original babble noise regardless.
        if subset == "train":
            self.audio_pipeline = torch.nn.Sequential(
                AdaptiveTimeMask(6400, 16000),
                # Randomly noise each utterance at one of ``snr_levels`` (dB),
                # applied with probability ``noise_prob``.
                AddNoise(snr_levels=snr_levels, noise_prob=noise_prob),
                FunctionalModule(_instance_layer_norm),
            )
        elif subset == "val" or subset == "test":
            self.audio_pipeline = torch.nn.Sequential(
                AddNoise(snr_target=snr_target,
                         noise_filename=noise_filename or NOISE_FILENAME)
                if snr_target is not None
                else FunctionalModule(_identity),
                FunctionalModule(_instance_layer_norm),
            )

    def __call__(self, sample):
        # sample: T x 1
        # rtype: T x 1
        return self.audio_pipeline(sample)


class TextTransform:
    """Mapping Dictionary Class for SentencePiece tokenization."""

    def __init__(
        self,
        sp_model_path=SP_MODEL_PATH,
        dict_path=DICT_PATH,
    ):

        # Load SentencePiece model
        self.spm = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)

        # Load units and create dictionary
        units = open(dict_path, encoding='utf8').read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
        # 0 will be used for "blank" in CTC
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text):
        tokens = self.spm.EncodeAsPieces(text)
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in tokens]
        return torch.tensor(list(map(int, token_ids)))

    def post_process(self, token_ids):
        token_ids = token_ids[token_ids != -1]
        text = self._ids_to_str(token_ids, self.token_list)
        text = text.replace("\u2581", " ").strip()
        return text

    def _ids_to_str(self, token_ids, char_list):
        token_as_list = [char_list[idx] for idx in token_ids]
        return "".join(token_as_list).replace("<space>", " ")


class JapaneseTextTransform:
    """Character-based tokenizer for Japanese (Katakana) transcripts.

    Unlike the English ``TextTransform`` (which relies on a SentencePiece model),
    the Japanese baseline in Kondo & Tamura (2025) uses a character-level
    recognizer over Katakana, yielding a small vocabulary (~87 units). This class
    therefore tokenizes by splitting the string into individual characters and
    mapping each one through the units dictionary -- no SentencePiece model is
    required, which lets us build the model before the final corpus is chosen.
    """

    def __init__(self, dict_path=JA_DICT_PATH):
        units = open(dict_path, encoding="utf8").read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
        # 0 is reserved for the CTC "blank"; <eos> is appended last.
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text):
        # Character-level: drop whitespace (Katakana transcripts are unsegmented).
        tokens = [c for c in text if not c.isspace()]
        token_ids = [self.hashmap.get(t, self.hashmap["<unk>"]) for t in tokens]
        return torch.tensor(list(map(int, token_ids)))

    def post_process(self, token_ids):
        token_ids = token_ids[token_ids != -1]
        text = "".join(self.token_list[idx] for idx in token_ids)
        return text.replace("<eos>", "").strip()


def get_text_transform(lang="en", sp_model_path=None, dict_path=None):
    """Factory returning the tokenizer for the requested language.

    ``lang="en"`` -> SentencePiece-based ``TextTransform`` (LRS3 default).
    ``lang="ja"`` -> character-based ``JapaneseTextTransform`` (Katakana).
    """
    if lang == "ja":
        return JapaneseTextTransform(dict_path or JA_DICT_PATH)
    return TextTransform(
        sp_model_path or SP_MODEL_PATH,
        dict_path or DICT_PATH,
    )
