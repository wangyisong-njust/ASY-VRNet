"""Create baseline-vs-ours detection comparison figures with confidence scores.

The script reads prediction txt files in the common mAP format:

    class score left top right bottom

Ground-truth txt files are optional. When provided, the title reports both
prediction count and matched true positives, e.g. "pred 15, TP 8/17".
In the Ours panel, boxes that match a GT missed by baseline are drawn green.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw baseline-vs-ours detection comparisons with scores."
    )
    parser.add_argument("--image-dir", required=True, help="Directory containing source images.")
    parser.add_argument("--baseline-dir", required=True, help="Baseline detection-results directory.")
    parser.add_argument("--ours-dir", required=True, help="Ours detection-results directory.")
    parser.add_argument("--out-dir", required=True, help="Output directory for comparison jpgs.")
    parser.add_argument("--gt-dir", default="", help="Optional ground-truth directory.")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional image ids to render.")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for GT matching.")
    parser.add_argument("--score-digits", type=int, default=2, help="Digits shown after decimal point.")
    parser.add_argument("--max-boxes", type=int, default=100, help="Maximum predictions drawn per panel.")
    parser.add_argument("--font", default="model_data/simhei.ttf", help="Font path.")
    parser.add_argument("--hide-class", action="store_true", help="Draw score only, without class name.")
    return parser.parse_args()


def read_pred(path: Path) -> list[dict]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        cls, score, left, top, right, bottom = parts
        boxes.append(
            {
                "cls": cls,
                "score": float(score),
                "box": np.asarray([left, top, right, bottom], dtype=np.float32),
            }
        )
    boxes.sort(key=lambda item: item["score"], reverse=True)
    return boxes


def read_gt(path: Path) -> list[dict]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, left, top, right, bottom = parts
        boxes.append({"cls": cls, "box": np.asarray([left, top, right, bottom], dtype=np.float32)})
    return boxes


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def match_gt(preds: list[dict], gts: list[dict], iou_thr: float) -> set[int]:
    matched: set[int] = set()
    used_pred: set[int] = set()
    for gt_idx, gt in enumerate(gts):
        best_iou = 0.0
        best_pred = None
        for pred_idx, pred in enumerate(preds):
            if pred_idx in used_pred or pred["cls"] != gt["cls"]:
                continue
            overlap = iou(pred["box"], gt["box"])
            if overlap > best_iou:
                best_iou = overlap
                best_pred = pred_idx
        if best_pred is not None and best_iou >= iou_thr:
            matched.add(gt_idx)
            used_pred.add(best_pred)
    return matched


def find_image(image_dir: Path, image_id: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = image_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_font(path: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_detections(
    image: Image.Image,
    preds: list[dict],
    font: ImageFont.ImageFont,
    score_digits: int,
    hide_class: bool,
    recovered_gt: set[int] | None = None,
    gts: list[dict] | None = None,
    iou_thr: float = 0.5,
) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    thickness = max(2, int(round((canvas.size[0] + canvas.size[1]) / 700)))

    for pred in preds:
        color = (239, 68, 68)
        if recovered_gt and gts:
            for gt_idx in recovered_gt:
                if pred["cls"] == gts[gt_idx]["cls"] and iou(pred["box"], gts[gt_idx]["box"]) >= iou_thr:
                    color = (34, 197, 94)
                    break

        left, top, right, bottom = [int(round(v)) for v in pred["box"]]
        left = max(0, min(left, canvas.size[0] - 1))
        right = max(0, min(right, canvas.size[0] - 1))
        top = max(0, min(top, canvas.size[1] - 1))
        bottom = max(0, min(bottom, canvas.size[1] - 1))
        if right <= left or bottom <= top:
            continue

        box_thickness = min(thickness, max(1, (right - left + 1) // 2), max(1, (bottom - top + 1) // 2))
        for t in range(box_thickness):
            draw.rectangle([left + t, top + t, right - t, bottom - t], outline=color)

        score = f"{pred['score']:.{score_digits}f}"
        label = score if hide_class else f"{pred['cls']} {score}"
        tw, th = text_size(draw, label, font)
        label_top = max(0, top - th - 4)
        label_w = min(tw + 6, canvas.size[0])
        label_left = min(max(0, left), max(0, canvas.size[0] - label_w))
        draw.rectangle([label_left, label_top, label_left + label_w, label_top + th + 4], fill=color)
        draw.text((label_left + 3, label_top + 2), label, fill=(255, 255, 255), font=font)

    return canvas


def add_title(image: Image.Image, title: str, font: ImageFont.ImageFont) -> Image.Image:
    draw_probe = ImageDraw.Draw(image)
    _, th = text_size(draw_probe, title, font)
    bar_h = max(28, th + 10)
    canvas = Image.new("RGB", (image.size[0], image.size[1] + bar_h), (25, 25, 25))
    canvas.paste(image, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(245, 245, 245), font=font)
    return canvas


def default_ids(baseline_dir: Path, ours_dir: Path) -> list[str]:
    baseline = {p.stem for p in baseline_dir.glob("*.txt")}
    ours = {p.stem for p in ours_dir.glob("*.txt")}
    return sorted(baseline & ours)


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    baseline_dir = Path(args.baseline_dir)
    ours_dir = Path(args.ours_dir)
    gt_dir = Path(args.gt_dir) if args.gt_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ids = args.ids if args.ids else default_ids(baseline_dir, ours_dir)
    if not ids:
        raise SystemExit("No image ids found. Pass --ids or check prediction directories.")

    for image_id in ids:
        image_path = find_image(image_dir, image_id)
        if image_path is None:
            print(f"skip {image_id}: image not found")
            continue

        image = Image.open(image_path).convert("RGB")
        font = load_font(args.font, max(12, int(image.size[1] * 0.032)))
        title_font = load_font(args.font, max(16, int(image.size[1] * 0.04)))

        baseline = read_pred(baseline_dir / f"{image_id}.txt")[: args.max_boxes]
        ours = read_pred(ours_dir / f"{image_id}.txt")[: args.max_boxes]
        gts = read_gt(gt_dir / f"{image_id}.txt") if gt_dir else []

        if gts:
            base_match = match_gt(baseline, gts, args.iou_thr)
            ours_match = match_gt(ours, gts, args.iou_thr)
            recovered = ours_match - base_match
            base_title = f"Baseline pred {len(baseline)}, TP {len(base_match)}/{len(gts)}"
            ours_title = f"Ours pred {len(ours)}, TP {len(ours_match)}/{len(gts)} (green = recovered)"
        else:
            recovered = set()
            base_title = f"Baseline pred {len(baseline)}"
            ours_title = f"Ours pred {len(ours)}"

        left = draw_detections(
            image, baseline, font, args.score_digits, args.hide_class,
            recovered_gt=None, gts=gts, iou_thr=args.iou_thr,
        )
        right = draw_detections(
            image, ours, font, args.score_digits, args.hide_class,
            recovered_gt=recovered, gts=gts, iou_thr=args.iou_thr,
        )
        left = add_title(left, base_title, title_font)
        right = add_title(right, ours_title, title_font)

        panel = Image.new("RGB", (left.size[0] + right.size[0] + 4, left.size[1]), (255, 255, 255))
        panel.paste(left, (0, 0))
        panel.paste(right, (left.size[0] + 4, 0))
        panel.save(out_dir / f"compare_{image_id}_scores.jpg", quality=95)
        print(out_dir / f"compare_{image_id}_scores.jpg")


if __name__ == "__main__":
    main()
