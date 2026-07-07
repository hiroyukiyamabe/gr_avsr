"""Pick the lowest-loss best-val Lightning checkpoint and export it as a bare
model state_dict (.pth) that eval.py can load directly.

The training run saves `best-{epoch}-{loss_val}.ckpt` files (one per process, so
several may survive across resumes). eval.py does
`model.load_state_dict(torch.load(pretrained_model_path))`, which needs the plain
model weights -- not the nested Lightning dict. This mirrors avg_ckpts.py's
prefix stripping (state_dict -> drop the "model." prefix).

Usage:
    python preparation/export_best_val.py exp/rohan_vsr_fixed
Prints the chosen checkpoint and writes <exp_dir>/best_val_model.pth.
"""
import glob
import os
import re
import sys

import torch


def main(exp_dir):
    ckpts = glob.glob(os.path.join(exp_dir, "best-*.ckpt"))
    if not ckpts:
        raise SystemExit(f"no best-*.ckpt found in {exp_dir}")

    def loss_of(path):
        m = re.search(r"loss_val=([0-9]+\.[0-9]+)", os.path.basename(path))
        return float(m.group(1)) if m else float("inf")

    best = min(ckpts, key=loss_of)
    print("candidates:")
    for c in sorted(ckpts, key=loss_of):
        print(f"  {os.path.basename(c)}  (loss_val={loss_of(c)})")
    print(f"selected: {os.path.basename(best)}")

    state = torch.load(best, map_location="cpu", weights_only=False)["state_dict"]
    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}

    out = os.path.join(exp_dir, "best_val_model.pth")
    torch.save(model_state, out)
    print(f"wrote {out}  ({len(model_state)} tensors)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "exp/rohan_vsr_fixed")
