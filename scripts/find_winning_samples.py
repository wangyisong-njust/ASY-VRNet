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


def draw_boxes_only(img, dets, hl_boxes=None, w=3):
    """Draw detection rectangles only (no text). hl_boxes (xyxy) drawn in lime."""
    d = ImageDraw.Draw(img)
    hl = hl_boxes or []
    for det in dets:
        x1, y1, x2, y2 = det[:4]
        color = (50, 255, 50) if any(iou_xyxy(det[:4], h) >= 0.5 for h in hl) else (255, 40, 40)
        for k in range(w):
            d.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)
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
    ap.add_argument("--filter", default="none", choices=["none", "night", "small", "dim"],
                    help="restrict candidates to a hard subset")
    ap.add_argument("--info_csv", default="dataset/WaterScenes_Full/information_list.csv")
    ap.add_argument("--small_area", type=float, default=4096.0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--crop_to_diff", action="store_true", default=True,
                    help="zoom both panels into the region ours detects but baseline misses")
    ap.add_argument("--no_crop", action="store_false", dest="crop_to_diff")
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

    # ---- optional hard-subset filter ----
    if args.filter in ("night", "dim"):
        import csv as _csv
        keep = set()
        csv_path = ROOT / args.info_csv
        if not csv_path.exists():
            raise SystemExit(f"info csv not found: {csv_path}")
        with open(csv_path, encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if args.filter == "night" and (row.get("time") or "").strip().lower() == "night":
                    keep.add(row.get("id", ""))
                elif args.filter == "dim" and (row.get("lighting") or "").strip().lower() == "dim" \
                        and (row.get("weather") or "").strip().lower() in ("overcast", "rainy"):
                    keep.add(row.get("id", ""))
        ids = [i for i in ids if i in keep]
        print(f"[win] filter={args.filter}: {len(ids)} candidate images")
    elif args.filter == "small":
        def has_small(boxes):
            return any((b[2] - b[0]) * (b[3] - b[1]) <= args.small_area for b in boxes)
        ids = [i for i in ids if has_small(gt[i])]
        print(f"[win] filter=small (GT area<= {args.small_area}): {len(ids)} candidate images")

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
        W, H = Image.open(p).size

        # Region of interest = the GT objects ours catches but baseline misses.
        diff = [gt[img_id][gi] for gi in (hf - hb)]
        if args.crop_to_diff and diff:
            x1 = min(b[0] for b in diff); y1 = min(b[1] for b in diff)
            x2 = max(b[2] for b in diff); y2 = max(b[3] for b in diff)
            mw = max(40, (x2 - x1) * 0.8); mh = max(40, (y2 - y1) * 0.8)
            crop = (max(0, int(x1 - mw)), max(0, int(y1 - mh)),
                    min(W, int(x2 + mw)), min(H, int(y2 + mh)))
        else:
            crop = (0, 0, W, H)

        # Boxes only (no text). On ours, highlight in green the GT objects it
        # recovers that baseline missed; everything else red.
        db = predict_boxes(yb, Image.open(p).convert("RGB"), img_id)
        df = predict_boxes(yf, Image.open(p).convert("RGB"), img_id)
        base_panel = draw_boxes_only(Image.open(p).convert("RGB"), db).crop(crop)
        final_panel = draw_boxes_only(Image.open(p).convert("RGB"), df, hl_boxes=diff).crop(crop)
        # upscale small crops so the difference is visible
        if base_panel.width < 520:
            s = 520 / base_panel.width
            sz = (int(base_panel.width * s), int(base_panel.height * s))
            base_panel = base_panel.resize(sz); final_panel = final_panel.resize(sz)
        base_panel = label_bar(base_panel, f"Baseline  detect {len(hb)}/{ng}")
        final_panel = label_bar(final_panel, f"Ours  detect {len(hf)}/{ng}  (green = recovered)")
        h = max(base_panel.height, final_panel.height)
        combo = Image.new("RGB", (base_panel.width + final_panel.width + 8, h), (255, 255, 255))
        combo.paste(base_panel, (0, 0))
        combo.paste(final_panel, (base_panel.width + 8, 0))
        out = out_dir / f"win{rank:02d}_{img_id}_plus{win}.jpg"
        combo.save(out, quality=92)
        print(f"  [saved] {out.name}  (baseline {len(hb)}/{ng}, ours {len(hf)}/{ng})")

    print(f"[win] done -> {out_dir}  (cropped to the region ours detects but baseline misses)")


if __name__ == "__main__":
    main()
