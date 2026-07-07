#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert ROHAN HTK-style phoneme alignment (`.lab`) files into Katakana.

The ROHAN4600 corpus ships per-utterance forced-alignment labels in the HTK
mono-phone format::

    <start_100ns> <end_100ns> <phoneme>
    0 14675000 sil
    14675000 15504436 n
    15504436 16350000 a
    ...

The Japanese AVSR model is trained on **character-level Katakana** transcripts
(see ``datamodule/transforms.py::JapaneseTextTransform`` and
``spm/ja_char/ja_char_units.txt``). This module reconstructs the spoken Katakana
string from the phoneme sequence: it groups (consonant + vowel) into morae and
maps each mora to Katakana, drops the silence markers (``sil``/``pau``/``sp``),
maps the moraic nasal ``N`` -> ``ン`` and the geminate marker ``cl`` -> ``ッ``.

The alignment files contain a small amount of editing noise -- phonemes with
stray underscores or repeated characters (e.g. ``s_h``, ``pa__u``,
``t________s``). ``normalize_phoneme`` strips that noise before mapping.

Run ``python rohan_phonemes.py <dir-or-file> ...`` to self-test the conversion
against real ``.lab`` files (prints a few transcripts and any unmapped tokens).
"""

import os
import sys

# Phonemes that map to a single kana on their own.
STANDALONE = {
    "N": "ン",   # moraic nasal
    "cl": "ッ",  # geminate consonant (sokuon)
    "q": "ッ",   # alternative geminate symbol
}

VOWELS = ("a", "i", "u", "e", "o")

# Small kana used to build palatalised (-y-) and glide (-w-) morae.
_SMALL_Y = {"a": "ャ", "u": "ュ", "o": "ョ", "i": "ィ", "e": "ェ"}
_SMALL_V = {"a": "ァ", "i": "ィ", "u": "ゥ", "e": "ェ", "o": "ォ"}

# Base ("i-column") kana for palatalised consonants: base + small-y vowel.
_PALATAL_BASE = {
    "ky": "キ", "gy": "ギ", "ny": "ニ", "hy": "ヒ", "by": "ビ",
    "py": "ピ", "my": "ミ", "ry": "リ", "dy": "ヂ", "ty": "テ", "fy": "フ",
}
# Glide consonants (kw/gw): base + small vowel (ァィゥェォ).
_GLIDE_BASE = {"kw": "ク", "gw": "グ"}

# Plain (consonant -> per-vowel) morae, ordered a/i/u/e/o.
_PLAIN = {
    "":   ("ア", "イ", "ウ", "エ", "オ"),
    "k":  ("カ", "キ", "ク", "ケ", "コ"),
    "g":  ("ガ", "ギ", "グ", "ゲ", "ゴ"),
    "s":  ("サ", "スィ", "ス", "セ", "ソ"),
    "sh": ("シャ", "シ", "シュ", "シェ", "ショ"),
    "z":  ("ザ", "ズィ", "ズ", "ゼ", "ゾ"),
    "j":  ("ジャ", "ジ", "ジュ", "ジェ", "ジョ"),
    "t":  ("タ", "ティ", "トゥ", "テ", "ト"),
    "ch": ("チャ", "チ", "チュ", "チェ", "チョ"),
    "ts": ("ツァ", "ツィ", "ツ", "ツェ", "ツォ"),
    "d":  ("ダ", "ディ", "ドゥ", "デ", "ド"),
    "n":  ("ナ", "ニ", "ヌ", "ネ", "ノ"),
    "h":  ("ハ", "ヒ", "フ", "ヘ", "ホ"),
    "f":  ("ファ", "フィ", "フ", "フェ", "フォ"),
    "b":  ("バ", "ビ", "ブ", "ベ", "ボ"),
    "p":  ("パ", "ピ", "プ", "ペ", "ポ"),
    "m":  ("マ", "ミ", "ム", "メ", "モ"),
    "y":  ("ヤ", "イ", "ユ", "イェ", "ヨ"),
    "r":  ("ラ", "リ", "ル", "レ", "ロ"),
    "w":  ("ワ", "ウィ", "ウ", "ウェ", "ヲ"),
    "v":  ("ヴァ", "ヴィ", "ヴ", "ヴェ", "ヴォ"),
}

# All recognised consonant onsets (longest first so digraphs win over singles).
_CONSONANTS = sorted(
    set(_PLAIN) - {""} | set(_PALATAL_BASE) | set(_GLIDE_BASE),
    key=len,
    reverse=True,
)


def normalize_phoneme(token):
    """Canonicalise a raw phoneme token from a ``.lab`` file.

    Strips the editing noise seen in ROHAN (stray ``_``) and collapses obvious
    silence variants. Returns ``""`` for tokens that should be ignored.
    """
    t = token.strip().replace("_", "")
    if not t:
        return ""
    # Silence / pause variants (sil, silB, silE, pau, sp, ...).
    if t.startswith("sil") or t.startswith("pau") or t == "sp":
        return "sil"
    return t


def _mora_to_kana(consonant, vowel):
    """Map a (consonant, vowel) mora to Katakana, or ``None`` if unknown."""
    if vowel not in VOWELS:
        return None
    vi = VOWELS.index(vowel)
    if consonant in _PLAIN:
        return _PLAIN[consonant][vi]
    if consonant in _PALATAL_BASE:
        return _PALATAL_BASE[consonant] + _SMALL_Y[vowel]
    if consonant in _GLIDE_BASE:
        return _GLIDE_BASE[consonant] + _SMALL_V[vowel]
    return None


def phonemes_to_katakana(phonemes, unknown=None):
    """Convert a sequence of (raw) phoneme tokens into a Katakana string.

    ``unknown`` (if a list) collects phonemes that could not be mapped, for
    diagnostics. Unknown/silence tokens never raise; they are skipped.
    """
    out = []
    pending = ""  # buffered consonant awaiting its vowel
    for raw in phonemes:
        p = normalize_phoneme(raw)
        if not p or p == "sil":
            pending = ""
            continue
        if p in STANDALONE:
            out.append(STANDALONE[p])
            pending = ""
            continue
        if p in VOWELS:
            kana = _mora_to_kana(pending, p)
            if kana is None:
                if unknown is not None:
                    unknown.append(pending + p)
            else:
                out.append(kana)
            pending = ""
            continue
        if p in _CONSONANTS:
            # Two consonants in a row -> the previous one had no vowel; the
            # aligner occasionally emits that. Drop the orphan and keep going.
            pending = p
            continue
        # Anything else is genuinely unknown (e.g. corrupt "labl"/"lab2").
        if unknown is not None:
            unknown.append(p)
        pending = ""
    return "".join(out)


def read_lab(lab_path):
    """Read the phoneme column from an HTK ``.lab`` alignment file."""
    phonemes = []
    with open(lab_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            # "<start> <end> <phoneme>" or just "<phoneme>".
            phonemes.append(parts[-1])
    return phonemes


def lab_to_katakana(lab_path, unknown=None):
    """Convert a ``.lab`` alignment file directly to a Katakana transcript."""
    return phonemes_to_katakana(read_lab(lab_path), unknown=unknown)


def _selftest(paths):
    import glob

    lab_files = []
    for p in paths:
        if os.path.isdir(p):
            lab_files += sorted(glob.glob(os.path.join(p, "**", "*.lab"), recursive=True))
        else:
            lab_files.append(p)
    if not lab_files:
        print("no .lab files found", file=sys.stderr)
        return 1

    unknown = []
    shown = 0
    for lab in lab_files:
        kana = lab_to_katakana(lab, unknown=unknown)
        if shown < 8:
            print(f"{os.path.basename(lab)}: {kana}")
            shown += 1
    from collections import Counter

    print(f"\nconverted {len(lab_files)} files")
    if unknown:
        c = Counter(unknown)
        print(f"unmapped tokens ({len(unknown)} total, {len(c)} distinct):")
        for tok, n in c.most_common(30):
            print(f"  {tok!r}: {n}")
    else:
        print("no unmapped tokens")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:] or [os.getcwd()]
    raise SystemExit(_selftest(args))
