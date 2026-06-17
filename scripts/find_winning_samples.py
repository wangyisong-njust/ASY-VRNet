#!/usr/bin/env python3
"""Find qualitative samples where the final model clearly beats the baseline.

For each candidate image it runs both checkpoints, matches detections to the
ground-truth boxes (IoU>=0.5, class-aware), and scores the image by
   win = (#GT objects ours detects but baseline misses)
        - (#GT objects baseline detects but ours misses).
Images with the largest positive win are the most convincing paper figures
(ours recovers objects the baseline drops). The top-K are saved as
"baseline | ours" comparisons with the missed GT boxes highlighted.

Single GPU / CPU is enough.

Example:
  python3 scripts/find_winning_samples.py --scan 600 --topk 10 --confidence 0.3 \
      --out_dir presentation/comparison_wins
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("ASY_RADAR_LEGACY_PREPROCESS", "1")
os.environ.setdefault("ASY_RADAR_PRESERVE_POINTS", "0")
os.environ.setdefault("ASY_RADAR_CHANNELS", "4")
os.environ.setdefault("ASY_RADAR_SOURCE_ORDER", "range,doppler,elevation,power")
os.environ.setdefault("ASY_RADAR_TARGET_ORDER", "range,doppler,elevation,power")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from utils.utils import cvtColor, preprocess_input, resize_image  # noqa: E402
from utils.utils_bbox import decode_outputs, non_max_suppression  # noqa: E402


def parse_gt(val_txt: Path):
    gt = {}
    with open(val_txt, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            img_id = Path(parts[0]).stem
            boxes = []
            for tok in parts[1:]:
                xs = tok.split(",")
                if len(xs) >= 5:
                    x1, y1, x2, y2, c = map(int, xs[:5])
                    boxes.append((x1, y1, x2, y2, c))
            gt[img_id] = boxes
    return gt


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def predict_boxes(yolo, image, image_id):
    """Return list of (x1,y1,x2,y2,score,label) in original-image pixels."""
    image_shape = np.array(np.shape(image)[0:2])
    image = cvtColor(image)
    image_data = resize_image(image, (yolo.input_shape[1], yolo.input_shape[0]), yolo.letterbox_image)
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype="float32")), (2, 0, 1)), 0)
    radar_data = yolo._prepare_radar(image_id, image)
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        if yolo.cuda:
            images = images.cuda()
        outputs, _ = yolo.net(images, radar_data)
        outputs = decode_outputs(outputs, yolo.input_shape)
        results = non_max_suppression(outputs, yolo.num_classes, yolo.input_shape, image_shape,
                                      yolo.letterbox_image, conf_thres=yolo.confidence, nms_thres=yolo.nms_iou)
    if results[0] is None:
        return []
    out = []
    for r in results[0]:
        top, left, bottom, right = r[:4]
        out.append((float(left), float(top), float(right), float(bottom), float(r[4] * r[5]), int(r[6])))
    return out


def matched_gt(gt_boxes, dets, iou_thr=0.5):
    """Return set of GT indices that have a class-matching det with IoU>=thr."""
    hit = set()
    for gi, g in enumerate(gt_boxes):
        for d in dets:
            if d[5] == g[4] and iou_xyxy(g[:4], d[:4]) >= iou_thr:
                hit.add(gi)
                break
    return hit


def draw_gt(img, gt_boxes, missed_idx, color_hit=(0, 220, 0), color_miss=(255, 215, 0)):
    d = ImageDraw.Draw(img)
    for gi, g in enumerate(gt_boxes):
        c = color_miss if gi in missed_idx else None
        if c is None:
            continue
        for k in range(3):
            d.rectangle([g[0] - k, g[1] - k, g[2] + k, g[3] + k], outline=c)
    return img


def label_bar(img, text):
    bar = max(26, img.height // 18)
    canvas = Image.new("RGB", (img.width, img.height + bar), (30, 30, 30))
    canvas.paste(img, (0, bar))
    dd = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(str(ROOT / "model_data" / "simhei.ttf"), bar - 8)
    except Exception:
        font = ImageFont.load_default()
    dd.text((6, 2), text, fill=(255, 255, 255), font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="weights/baseline_best.pth")
    ap.add_argument("--final", default="weights/final_greedy_soup.pth")
    ap.add_argument("--classes_path", default="model_data/waterscenes.txt")
    ap.add_argument("--radar_root", default="dataset/VOCradar_5_frames")
    ap.add_argument("--images_dir", default="dataset/VOCdevkit/VOC2007/JPEGImages")
    ap.add_argument("--val_txt", default="2007_val.txt")
    ap.add_argument("--scan", type=int, default=600, help="how many val images to scan")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--confidence", type=float, default=0.3)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="presentation/comparison_wins")
    args = ap.parse_args()

    from yolo import YOLO
    common = dict(classes_path=args.classes_path, radar_root=args.radar_root, input_shape=[320, 320],
                  phi="l", confidence=args.confidence, nms_iou=args.nms_iou)
    print("[win] loading models ...")
    yb = YOLO(model_path=args.baseline, **common)
    yf = YOLO(model_path=args.final, **common)

    gt = parse_gt(ROOT / args.val_txt)
    ids = list(gt.keys())
    import random
    random.Random(args.seed).shuffle(ids)
    ids = ids[:args.scan]

    scored = []
    for n, img_id in enumerate(ids):
        p = ROOT / args.images_dir / f"{img_id}.jpg"
        if not p.exists() or not gt[img_id]:
            continue
        img = Image.open(p).convert("RGB")
        db = predict_boxes(yb, img, img_id)
        df = predict_boxes(yf, img, img_id)
        hb = matched_gt(gt[img_id], db)
        hf = matched_gt(gt[img_id], df)
        win = len(hf - hb)      # ours catches, baseline misses
        loss = len(hb - hf)     # baseline catches, ours misses
        score = win - loss
        if win >= 1 and score >= 1:
            scored.append((score, win, loss, img_id, hb, hf))
        if (n + 1) % 50 == 0:
            print(f"  scanned {n+1}/{len(ids)}, candidates={len(scored)}")

    scored.sort(reverse=True)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[win] {len(scored)} winning candidates; saving top {args.topk}")

    for rank, (score, win, loss, img_id, hb, hf) in enumerate(scored[:args.topk]):
        p = ROOT / args.images_dir / f"{img_id}.jpg"
        ng = len(gt[img_id])
        # baseline panel: its preds + GT it MISSED (yellow)
        base_panel = yb.detect_image(Image.open(p).convert("RGB"), img_id)
        base_panel = draw_gt(base_panel, gt[img_id], set(range(ng)) - hb)
        base_panel = label_bar(base_panel, f"Baseline  hit {len(hb)}/{ng} GT")
        # ours panel
        final_panel = yf.detect_image(Image.open(p).convert("RGB"), img_id)
        final_panel = draw_gt(final_panel, gt[img_id], set(range(ng)) - hf)
        final_panel = label_bar(final_panel, f"Ours  hit {len(hf)}/{ng} GT  (+{win})")
        h = max(base_panel.height, final_panel.height)
        combo = Image.new("RGB", (base_panel.width + final_panel.width + 8, h), (255, 255, 255))
        combo.paste(base_panel, (0, 0))
        combo.paste(final_panel, (base_panel.width + 8, 0))
        out = out_dir / f"win{rank:02d}_{img_id}_plus{win}.jpg"
        combo.save(out, quality=92)
        print(f"  [saved] {out.name}  (baseline {len(hb)}/{ng}, ours {len(hf)}/{ng})")

    print(f"[win] done -> {out_dir}  (yellow boxes = GT objects that model missed)")


if __name__ == "__main__":
    main()
