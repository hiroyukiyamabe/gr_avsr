"""Front-end extractor for cross-lingual transfer learning.

The pre-trained Auto-AVSR checkpoints in ``original_checkpoints/`` store the
whole end-to-end model as a *flat* state dict whose top-level modules are::

    frontend.*        # visual (Conv3dResNet) or audio (Conv1dResNet) front-end
    proj_encoder.*    # linear projection into the Conformer
    encoder.*         # Conformer back-end
    decoder.*         # Transformer decoder
    ctc.*             # CTC projection

For Japanese cross-lingual transfer (framework (c) in Kondo & Tamura, 2025) we
only want to inherit the **front-end encoders** and train everything else from
scratch.  The training code (``auto_avsr/lightning.py`` / ``lightning_av.py``)
loads a front-end with::

    ckpt = torch.load(path)
    tmp = {k: v for k, v in ckpt["model_state_dict"].items()
           if k.startswith("trunk.") or k.startswith("frontend3D.")}
    model.encoder.frontend.load_state_dict(tmp)   # strict=True

i.e. it expects the weights to be (a) wrapped in a ``model_state_dict`` entry and
(b) keyed *relative to the front-end module* (``trunk.*`` / ``frontend3D.*``),
without the leading ``frontend.`` that the full checkpoints carry.

This script performs that conversion: it reads a full checkpoint, keeps only the
``frontend.*`` tensors, strips the prefix, and writes a small front-end-only
checkpoint that ``transfer_frontend`` can load as-is.

Newer single-modality checkpoints (the 3,448 h ``*_trlrwlrs2lrs3vox2avsp``
files) nest the front-end at ``encoder.frontend.*`` instead of the flat
``frontend.*`` shown above; ``extract_frontend`` handles both layouts.

Audio-visual checkpoints (``--av``): the AV model stores the visual front-end at
``encoder.frontend.*`` and the audio front-end at ``aux_encoder.frontend.*``.
``--av --src avsr_trlrwlrs2lrs3vox2avsp_base.pth`` extracts both in one pass and
writes ``frontend_visual_3448.pth`` + ``frontend_audio_3448.pth``.
"""

import argparse
from pathlib import Path

import torch

HERE = Path(__file__).parent
DEFAULT_CKPT_DIR = HERE.parent / "original_checkpoints"
DEFAULT_OUT_DIR = HERE / "frontends"

# Source checkpoint -> output front-end file.
#   VSR checkpoint  -> visual front-end (Conv3dResNet: frontend3D.* + trunk.*)
#   ASR checkpoint  -> audio  front-end (Conv1dResNet: trunk.*)
DEFAULT_JOBS = {
    "vsr_trlrs3_base.pth": "frontend_visual_trlrs3.pth",
    "asr_trlrs3_base.pth": "frontend_audio_trlrs3.pth",
}


def _load_state(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Unwrap if the checkpoint is wrapped (some are saved as {"model_state_dict": ...}).
    return ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt


def _strip_prefix(state, prefix, ckpt_name):
    sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not sub:
        raise RuntimeError(
            f"No '{prefix}*' keys found in {ckpt_name}. "
            f"Top-level prefixes present: "
            f"{sorted({k.split('.')[0] for k in state})}"
        )
    return sub


def extract_frontend(ckpt_path: Path):
    """Return a front-end-only state dict keyed as ``trunk.*`` / ``frontend3D.*``.

    Two single-modality layouts exist in the model zoo: the older
    trlrs3-generation checkpoints keep the front-end flat at the top level
    (``frontend.*``), while the newer 3,448 h ones nest it inside the encoder
    module (``encoder.frontend.*``), like the AV checkpoints. Try both.
    """
    state = _load_state(ckpt_path)
    for prefix in ("frontend.", "encoder.frontend."):
        sub = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        if sub:
            return sub
    raise RuntimeError(
        f"No 'frontend.*' or 'encoder.frontend.*' keys found in {ckpt_path.name}. "
        f"Top-level prefixes present: {sorted({k.split('.')[0] for k in state})}"
    )


def extract_av_frontends(ckpt_path: Path):
    """Extract BOTH front-ends from an audio-visual (avsr_*) checkpoint.

    The AV model (``e2e_asr_conformer_av.py``) keeps the visual front-end at
    ``encoder.frontend.*`` and the audio front-end at ``aux_encoder.frontend.*``.
    Returns ``(visual_frontend, audio_frontend)``, each keyed relative to the
    front-end module (``trunk.*`` / ``frontend3D.*``) like the single-modality
    extraction, so ``transfer_frontend`` loads them unchanged.
    """
    state = _load_state(ckpt_path)
    visual = _strip_prefix(state, "encoder.frontend.", ckpt_path.name)
    audio = _strip_prefix(state, "aux_encoder.frontend.", ckpt_path.name)
    return visual, audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR,
                        help="Directory holding the full pre-trained checkpoints.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Where to write the extracted front-end checkpoints.")
    parser.add_argument("--src", type=str, default=None,
                        help="Single source checkpoint filename (overrides defaults).")
    parser.add_argument("--dst", type=str, default=None,
                        help="Output filename for --src.")
    parser.add_argument("--av", action="store_true",
                        help="Treat --src as an audio-visual (avsr_*) checkpoint and "
                             "extract BOTH front-ends: encoder.frontend.* -> visual, "
                             "aux_encoder.frontend.* -> audio.")
    parser.add_argument("--inspect", action="store_true",
                        help="Only print the front-end keys/shapes, do not write.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def report_and_save(frontend, dst):
        print(f"  {len(frontend)} front-end tensors")
        for k in list(frontend)[:6]:
            print(f"    {k}  {tuple(frontend[k].shape)}")
        if len(frontend) > 6:
            print("    ...")
        if not args.inspect:
            out_path = args.out_dir / dst
            torch.save({"model_state_dict": frontend}, out_path)
            print(f"    saved -> {out_path}")

    if args.av:
        if not args.src:
            parser.error("--av requires --src")
        src_path = args.ckpt_dir / args.src
        visual, audio = extract_av_frontends(src_path)
        print(f"\n{args.src} [visual: encoder.frontend.*]")
        report_and_save(visual, "frontend_visual_3448.pth")
        print(f"\n{args.src} [audio: aux_encoder.frontend.*]")
        report_and_save(audio, "frontend_audio_3448.pth")
        return

    if args.src:
        jobs = {args.src: args.dst or ("frontend_" + args.src)}
    else:
        jobs = DEFAULT_JOBS

    for src, dst in jobs.items():
        src_path = args.ckpt_dir / src
        if not src_path.exists():
            print(f"[skip] {src_path} not found")
            continue
        frontend = extract_frontend(src_path)
        print(f"\n{src}")
        report_and_save(frontend, dst)


if __name__ == "__main__":
    main()
