"""Generate clearer Baseline-vs-Ours evidence figures.

The default side-by-side prediction view can look cluttered when both models
produce many boxes. This script keeps the evaluation honest by matching
predictions to GT with the same class-aware IoU rule, then highlights GT objects
that Ours recovers but Baseline misses. It also crops around the object cluster
so small targets are visible in a paper/PPT figure.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_detection import (  # noqa: E402
    box_iou,
    parse_shape,
    parse_tta_scales,
    read_detection_txt,
    read_voc_xml,
)
from yolo import YOLO  # noqa: E402


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make clear visual evidence figures.")
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument("--ids-file", default="")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--radar-root", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--baseline", default="weights/baseline_best.pth")
    parser.add_argument("--ours", default="weights/final_greedy_soup.pth")
    parser.add_argument("--out-dir", default="presentation/clear_visual_evidence_512")
    parser.add_argument("--classes-path", default="model_data/waterscenes.txt")
    parser.add_argument("--input-shape", type=parse_shape, default=[512, 512])
    parser.add_argument("--confidence", type=float, default=0.2)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--phi", default="l")
    parser.add_argument("--font", default="model_data/simhei.ttf")
    parser.add_argument("--radar-legacy-preprocess", action="store_true")
    parser.add_argument("--radar-preserve-points", action="store_true", default=True)
    parser.add_argument("--no-radar-preserve-points", action="store_false", dest="radar_preserve_points")
    parser.add_argument("--radar-source-order", default="range,doppler,elevation,power")
    parser.add_argument("--radar-target-order", default="range,elevation,velocity,power")
    parser.add_argument("--baseline-fusion-mode", default="baseline")
    parser.add_argument("--ours-fusion-mode", default="baseline")
    parser.add_argument("--ours-tta", action="store_true")
    parser.add_argument("--tta-scales", default="320,384")
    parser.add_argument("--tta-fusion", default="softnms", choices=["nms", "wbf", "softnms"])
    parser.add_argument("--tta-radar-alpha", type=float, default=0.0)
    parser.add_argument("--tta-wbf-iou", type=float, default=0.55)
    parser.add_argument("--tta-radar-tau", type=float, default=3.0)
    parser.add_argument("--tta-flip", action="store_true")
    return parser.parse_args()


def load_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if args.ids_file:
        for line in Path(args.ids_file).read_text(encoding="utf-8", errors="ignore").splitlines():
            item = line.strip().split()[0] if line.strip() else ""
            if item:
                ids.append(Path(item).stem)
    if args.ids:
        ids.extend(Path(item).stem for item in args.ids)
    return sorted(dict.fromkeys(ids))


def find_image(image_dir: Path, image_id: str) -> Path | None:
    for ext in IMAGE_EXTS:
        path = image_dir / f"{image_id}{ext}"
        if path.exists():
            return path
    return None


def load_font(path: str, size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def make_model(args: argparse.Namespace, model_path: str, fusion_mode: str) -> YOLO:
    return YOLO(
        model_path=model_path,
        radar_root=args.radar_root,
        classes_path=args.classes_path,
        input_shape=args.input_shape,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        phi=args.phi,
        fusion_mode=fusion_mode,
        radar_legacy_preprocess=args.radar_legacy_preprocess,
        radar_preserve_points=args.radar_preserve_points,
        radar_source_order=args.radar_source_order,
        radar_target_order=args.radar_target_order,
    )


def predict_txt(model: YOLO, image_id: str, image: Image.Image, tmp_root: Path, use_tta: bool, args: argparse.Namespace):
    map_out = tmp_root / image_id
    (map_out / "detection-results").mkdir(parents=True, exist_ok=True)
    if use_tta:
        model.get_map_txt_tta(
            image_id,
            image,
            model.class_names,
            str(map_out),
            scales=parse_tta_scales(args.tta_scales),
            flip=args.tta_flip,
            radar_alpha=args.tta_radar_alpha,
            wbf_iou=args.tta_wbf_iou,
            radar_tau=args.tta_radar_tau,
            fusion=args.tta_fusion,
        )
    else:
        model.get_map_txt(image_id, image, model.class_names, str(map_out))
    return read_detection_txt(map_out / "detection-results" / f"{image_id}.txt")


def match_gt(preds, gts, iou_thr: float) -> tuple[set[int], dict[int, int]]:
    used_preds: set[int] = set()
    matched_gt: set[int] = set()
    gt_to_pred: dict[int, int] = {}
    for gt_idx, (gt_cls, _, gt_box) in enumerate(gts):
        best_iou = 0.0
        best_idx = None
        for pred_idx, (pred_cls, _, pred_box) in enumerate(preds):
            if pred_idx in used_preds or pred_cls != gt_cls:
                continue
            overlap = box_iou(pred_box, gt_box)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = pred_idx
        if best_idx is not None and best_iou >= iou_thr:
            used_preds.add(best_idx)
            matched_gt.add(gt_idx)
            gt_to_pred[gt_idx] = best_idx
    return matched_gt, gt_to_pred


def crop_around_gts(image: Image.Image, gts, pad_ratio: float = 0.12) -> tuple[Image.Image, tuple[int, int, int, int]]:
    if not gts:
        return image.copy(), (0, 0, image.size[0], image.size[1])
    boxes = np.stack([gt[2] for gt in gts])
    left = float(np.min(boxes[:, 0]))
    top = float(np.min(boxes[:, 1]))
    right = float(np.max(boxes[:, 2]))
    bottom = float(np.max(boxes[:, 3]))
    width = right - left
    height = bottom - top
    pad = max(width, height, image.size[0] * 0.02) * pad_ratio
    left = max(0, int(left - pad))
    top = max(0, int(top - pad))
    right = min(image.size[0], int(right + pad))
    bottom = min(image.size[1], int(bottom + pad))
    min_h = int(image.size[1] * 0.18)
    if bottom - top < min_h:
        center = (top + bottom) // 2
        top = max(0, center - min_h // 2)
        bottom = min(image.size[1], top + min_h)
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def shift_box(box: np.ndarray, crop: tuple[int, int, int, int]) -> np.ndarray:
    left, top, _, _ = crop
    shifted = box.astype(np.float32).copy()
    shifted[[0, 2]] -= left
    shifted[[1, 3]] -= top
    return shifted


def draw_box(draw: ImageDraw.ImageDraw, box, color, width: int = 3, dash: bool = False) -> None:
    left, top, right, bottom = [int(round(float(v))) for v in box]
    if right <= left or bottom <= top:
        return
    if not dash:
        for t in range(width):
            draw.rectangle([left + t, top + t, right - t, bottom - t], outline=color)
        return
    step = 8
    for x in range(left, right, step * 2):
        draw.line([(x, top), (min(x + step, right), top)], fill=color, width=width)
        draw.line([(x, bottom), (min(x + step, right), bottom)], fill=color, width=width)
    for y in range(top, bottom, step * 2):
        draw.line([(left, y), (left, min(y + step, bottom))], fill=color, width=width)
        draw.line([(right, y), (right, min(y + step, bottom))], fill=color, width=width)


def render_panel(image: Image.Image, crop, preds, gts, matched_gt, gt_to_pred, recovered, title: str, font_path: str):
    crop_img, crop_box = crop
    panel = crop_img.copy().convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = load_font(font_path, max(12, int(panel.size[1] * 0.055)))
    small_font = load_font(font_path, max(10, int(panel.size[1] * 0.045)))

    for gt_idx, (_, _, gt_box) in enumerate(gts):
        shifted = shift_box(gt_box, crop_box)
        if gt_idx in recovered:
            draw_box(draw, shifted, (34, 197, 94), width=5)
        elif gt_idx not in matched_gt:
            draw_box(draw, shifted, (245, 158, 11), width=3, dash=True)

    for gt_idx, pred_idx in gt_to_pred.items():
        cls, score, pred_box = preds[pred_idx]
        shifted = shift_box(pred_box, crop_box)
        color = (34, 197, 94) if gt_idx in recovered else (59, 130, 246)
        draw_box(draw, shifted, color, width=3)
        label = f"{cls} {score:.2f}"
        lx, ly = int(max(0, shifted[0])), int(max(0, shifted[1] - 16))
        draw.text((lx, ly), label, fill=color, font=small_font)

    bar_h = max(30, int(panel.size[1] * 0.12))
    canvas = Image.new("RGB", (panel.size[0], panel.size[1] + bar_h), (25, 25, 25))
    canvas.paste(panel, (0, bar_h))
    ImageDraw.Draw(canvas).text((8, 7), title, fill=(245, 245, 245), font=font)
    return canvas


def fit_width(image: Image.Image, width: int) -> Image.Image:
    scale = width / image.size[0]
    return image.resize((width, max(1, int(image.size[1] * scale))))


def main() -> None:
    args = parse_args()
    ids = load_ids(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = make_model(args, args.baseline, args.baseline_fusion_mode)
    ours = make_model(args, args.ours, args.ours_fusion_mode)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for image_id in ids:
            image_path = find_image(Path(args.image_dir), image_id)
            gt_path = Path(args.gt_dir) / f"{image_id}.xml"
            if image_path is None or not gt_path.exists():
                continue
            image = Image.open(image_path).convert("RGB")
            gts = read_voc_xml(gt_path)
            base_boxes = predict_txt(baseline, image_id, image, tmp_root / "baseline", False, args)
            ours_boxes = predict_txt(ours, image_id, image, tmp_root / "ours", args.ours_tta, args)
            base_matched, base_gt_to_pred = match_gt(base_boxes, gts, args.match_iou)
            ours_matched, ours_gt_to_pred = match_gt(ours_boxes, gts, args.match_iou)
            recovered = ours_matched - base_matched

            crop = crop_around_gts(image, gts)
            base_title = f"Baseline: pred {len(base_boxes)}, TP {len(base_matched)}/{len(gts)}, missed {len(gts)-len(base_matched)}"
            ours_title = f"Ours: pred {len(ours_boxes)}, TP {len(ours_matched)}/{len(gts)}, recovered +{len(recovered)}"
            base_panel = render_panel(
                image, crop, base_boxes, gts, base_matched, base_gt_to_pred, set(), base_title, args.font
            )
            ours_panel = render_panel(
                image, crop, ours_boxes, gts, ours_matched, ours_gt_to_pred, recovered, ours_title, args.font
            )
            base_panel = fit_width(base_panel, 960)
            ours_panel = fit_width(ours_panel, 960)
            height = max(base_panel.size[1], ours_panel.size[1])
            canvas = Image.new("RGB", (base_panel.size[0] + ours_panel.size[0] + 6, height), (255, 255, 255))
            canvas.paste(base_panel, (0, 0))
            canvas.paste(ours_panel, (base_panel.size[0] + 6, 0))
            out_path = out_dir / f"clear_{image_id}_b{len(base_matched)}_o{len(ours_matched)}_gt{len(gts)}.jpg"
            canvas.save(out_path, quality=95)
            rows.append((image_id, len(base_boxes), len(base_matched), len(ours_boxes), len(ours_matched), len(gts), len(recovered), out_path))
            print(f"{image_id}: recovered +{len(recovered)} -> {out_path}")

    with (out_dir / "clear_evidence_summary.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("image_id,baseline_pred,baseline_tp,ours_pred,ours_tp,gt_count,recovered,output\n")
        for row in rows:
            f.write(",".join(str(x) for x in row) + "\n")


if __name__ == "__main__":
    main()
