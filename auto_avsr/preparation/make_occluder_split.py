"""Split the NatOcc COCO occluder objects into disjoint eval and train halves.

CLAUDE.md sections 2 and 4: occluder images used for evaluation must never be
used in training. The eval half is used by the seeded chunk corruption of the
Phase 2c strong-form test (decode.chunk, CLAUDE.md section 5.2 Part B); the train
half is reserved for future visual training augmentation.

Source: coco_object.7z (object_image_sr/*.jpeg + object_mask_x4/*.png, 128
objects), the occluder set used by AV-RelScore (Hong et al., CVPR 2023), from
https://github.com/kennyvoo/face-occlusion-generation (Google Drive file
1xPqPQCpIF6RmuehGxS8WRA992O5Pdj2z, archive sha256 d19b3328...307d).

Rule (fixed 2026-10-23, before any Part B run): sort the object stems by the
SHA-256 of the stem (UTF-8); the first half is "eval", the rest "train".

Usage (from auto_avsr/):
  python preparation/make_occluder_split.py \
      --src ../datasets/occluders/natocc_coco_object/coco_object \
      --out datamodule/natocc_split.csv
"""
import argparse
import csv
import hashlib
import os


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_text_file(path):
    """SHA-256 with CRLF normalised to LF, so git's autocrlf cannot change it."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(args.src, "object_image_sr")))
    stems.sort(key=lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    half = len(stems) // 2
    rows = []
    for i, stem in enumerate(stems):
        image = os.path.join(args.src, "object_image_sr", stem + ".jpeg")
        mask = os.path.join(args.src, "object_mask_x4", stem + ".png")
        rows.append({
            "stem": stem,
            "split": "eval" if i < half else "train",
            "image_sha256": sha256_file(image),
            "mask_sha256": sha256_file(mask),
        })
    if os.path.exists(args.out):
        raise FileExistsError(f"refusing to overwrite {args.out}")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "split", "image_sha256", "mask_sha256"],
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} objects: {half} eval, {len(rows) - half} train -> {args.out}")
    print("split file sha256 (LF-normalised):", sha256_text_file(args.out))


if __name__ == "__main__":
    main()
