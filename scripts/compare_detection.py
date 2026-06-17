#!/usr/bin/env python3
"""Side-by-side detection comparison: baseline vs final model.

Runs two checkpoints on the same image/radar pairs, draws detection boxes with
each, and saves a horizontal "baseline | final" comparison image per sample —
the qualitative figures typically shown in a paper.

Single GPU (or CPU via ASY_CUDA=0) is enough; no multi-GPU / training needed.

Example:
  python3 scripts/compare_detection.py \
      --baseline weights/baseline_best.pth \
      --final    weights/final_greedy_soup.pth \
      --num 8 --confidence 0.4 \
      --out_dir presentation/comparison
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Match the paper eval protocol for radar preprocessing BEFORE importing YOLO.
os.environ.setdefault("ASY_RADAR_LEGACY_PREPROCESS", "1")
os.environ.setdefault("ASY_RADAR_PRESERVE_POINTS", "0")
os.environ.setdefault("ASY_RADAR_CHANNELS", "4")
os.environ.setdefault("ASY_RADAR_SOURCE_ORDER", "range,doppler,elevation,power")
os.environ.setdefault("ASY_RADAR_TARGET_ORDER", "range,doppler,elevation,power")

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def sample_ids(val_txt: Path, num: int, seed: int):
    ids = []
    with open(val_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split()[0]
            ids.append(Path(token).stem)
    random.Random(seed).shuffle(ids)
    return ids[:num]


def label_bar(img: Image.Image, text: str) -> Image.Image:
    """Add a title bar above the image."""
    bar_h = max(24, img.height // 18)
    canvas = Image.new("RGB", (img.width, img.height + bar_h), (30, 30, 30))
    canvas.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(str(ROOT / "model_data" / "simhei.ttf"), bar_h - 8)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 2), text, fill=(255, 255, 255), font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="weights/baseline_best.pth")
    ap.add_argument("--final", default="weights/final_greedy_soup.pth")
    ap.add_argument("--classes_path", default="model_data/waterscenes.txt")
    ap.add_argument("--radar_root", default="dataset/VOCradar_5_frames")
    ap.add_argument("--images_dir", default="dataset/VOCdevkit/VOC2007/JPEGImages")
    ap.add_argument("--val_txt", default="2007_val.txt")
    ap.add_argument("--ids", nargs="*", default=None, help="explicit image ids; overrides sampling")
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phi", default="l")
    ap.add_argument("--confidence", type=float, default=0.4)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--out_dir", default="presentation/comparison")
    args = ap.parse_args()

    from yolo import YOLO

    common = dict(classes_path=args.classes_path, radar_root=args.radar_root,
                  input_shape=[320, 320], phi=args.phi,
                  confidence=args.confidence, nms_iou=args.nms_iou)
    print("[compare] loading baseline ...")
    yolo_base = YOLO(model_path=args.baseline, **common)
    print("[compare] loading final ...")
    yolo_final = YOLO(model_path=args.final, **common)

    ids = args.ids if args.ids else sample_ids(ROOT / args.val_txt, args.num, args.seed)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_id in ids:
        img_path = ROOT / args.images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"[skip] missing image {img_path}")
            continue
        base_img = label_bar(yolo_base.detect_image(Image.open(img_path).convert("RGB"), img_id),
                              f"Baseline  (mAP 42.6)")
        final_img = label_bar(yolo_final.detect_image(Image.open(img_path).convert("RGB"), img_id),
                              f"Ours  (mAP 52.0)")
        h = max(base_img.height, final_img.height)
        combo = Image.new("RGB", (base_img.width + final_img.width + 8, h), (255, 255, 255))
        combo.paste(base_img, (0, 0))
        combo.paste(final_img, (base_img.width + 8, 0))
        out = out_dir / f"compare_{img_id}.jpg"
        combo.save(out, quality=92)
        print(f"[saved] {out}")

    print(f"[compare] done -> {out_dir}")


if __name__ == "__main__":
    main()
