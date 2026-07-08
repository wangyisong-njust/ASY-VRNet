"""Run baseline and improved ASY-VRNet checkpoints on one image.

Example:
    python compare_detection.py \
        --image image/15239.jpg \
        --radar-root dataset/VOCradar_5_frames \
        --baseline weights/baseline_best.pth \
        --ours weights/final_greedy_soup.pth \
        --out outputs/compare_15239_scores.jpg

The output keeps the confidence text drawn by YOLO.detect_image, so the visual
comparison can be checked against confidence thresholds instead of bare boxes.
"""

from __future__ import annotations

import argparse
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from yolo import YOLO


def parse_shape(value: str) -> list[int]:
    parts = [int(part) for part in value.replace(",", " ").split()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--input-shape needs two integers, e.g. 320,320")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and ours detections with confidence scores.")
    parser.add_argument("--image", required=True, help="Input image path, e.g. image/15239.jpg.")
    parser.add_argument("--radar-root", required=True, help="Directory containing <image_id>.npz radar files.")
    parser.add_argument("--baseline", required=True, help="Baseline checkpoint path.")
    parser.add_argument("--ours", required=True, help="Improved checkpoint path.")
    parser.add_argument("--out", default="", help="Output side-by-side image path.")
    parser.add_argument("--gt-xml", default="", help="Optional VOC/WaterScenes detection XML for a GT panel.")
    parser.add_argument("--classes-path", default="model_data/waterscenes.txt")
    parser.add_argument("--input-shape", type=parse_shape, default=[320, 320])
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--phi", default="l")
    parser.add_argument("--radar-legacy-preprocess", action="store_true")
    parser.add_argument("--radar-preserve-points", action="store_true", default=True)
    parser.add_argument("--no-radar-preserve-points", action="store_false", dest="radar_preserve_points")
    parser.add_argument("--radar-source-order", default="range,doppler,elevation,power")
    parser.add_argument("--radar-target-order", default="range,elevation,velocity,power")
    parser.add_argument("--baseline-fusion-mode", default="baseline")
    parser.add_argument("--ours-fusion-mode", default="baseline")
    parser.add_argument("--font", default="model_data/simhei.ttf")
    parser.add_argument("--baseline-tta", action="store_true", help="Use TTA rendering for baseline.")
    parser.add_argument("--ours-tta", action="store_true", help="Use TTA rendering for ours.")
    parser.add_argument("--tta-scales", default="320,384", help="Comma-separated TTA square sizes.")
    parser.add_argument("--tta-fusion", default="softnms", choices=["nms", "wbf", "softnms"])
    parser.add_argument("--tta-radar-alpha", type=float, default=0.0)
    parser.add_argument("--tta-wbf-iou", type=float, default=0.55)
    parser.add_argument("--tta-radar-tau", type=float, default=3.0)
    parser.add_argument("--tta-flip", action="store_true", help="Enable horizontal flip TTA.")
    return parser.parse_args()


def load_title_font(path: str, height: int) -> ImageFont.ImageFont:
    size = max(18, int(height * 0.045))
    try:
        return ImageFont.truetype(path, size=size)
    except OSError:
        return ImageFont.load_default()


def add_title(image: Image.Image, title: str, font_path: str) -> Image.Image:
    font = load_title_font(font_path, image.size[1])
    probe = ImageDraw.Draw(image)
    bbox = probe.textbbox((0, 0), title, font=font)
    bar_h = max(34, bbox[3] - bbox[1] + 12)
    canvas = Image.new("RGB", (image.size[0], image.size[1] + bar_h), (25, 25, 25))
    canvas.paste(image.convert("RGB"), (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(245, 245, 245), font=font)
    return canvas


def read_detection_txt(path: Path) -> list[tuple[str, float, np.ndarray]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) != 6:
            continue
        cls, score, left, top, right, bottom = parts
        boxes.append((cls, float(score), np.asarray([left, top, right, bottom], dtype=np.float32)))
    boxes.sort(key=lambda item: item[1], reverse=True)
    return boxes


def draw_txt_detections(
    image: Image.Image,
    boxes: list[tuple[str, float | None, np.ndarray]],
    color: tuple[int, int, int],
    font_path: str,
) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = load_title_font(font_path, canvas.size[1] * 0.7)
    thickness = max(2, int(round((canvas.size[0] + canvas.size[1]) / 700)))
    for cls, score, box in boxes:
        left, top, right, bottom = [int(round(float(v))) for v in box]
        left = max(0, min(left, canvas.size[0] - 1))
        right = max(0, min(right, canvas.size[0] - 1))
        top = max(0, min(top, canvas.size[1] - 1))
        bottom = max(0, min(bottom, canvas.size[1] - 1))
        if right <= left or bottom <= top:
            continue
        box_thickness = min(thickness, max(1, (right - left + 1) // 2), max(1, (bottom - top + 1) // 2))
        for t in range(box_thickness):
            draw.rectangle([left + t, top + t, right - t, bottom - t], outline=color)
        text = cls if score is None else f"{cls} {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_top = max(0, top - th - 4)
        label_w = min(tw + 6, canvas.size[0])
        label_left = min(max(0, left), max(0, canvas.size[0] - label_w))
        draw.rectangle([label_left, label_top, label_left + label_w, label_top + th + 4], fill=color)
        draw.text((label_left + 3, label_top + 2), text, fill=(0, 0, 0), font=font)
    return canvas


def read_voc_xml(path: Path) -> list[tuple[str, None, np.ndarray]]:
    boxes = []
    if not path.exists():
        return boxes
    root = ET.parse(path).getroot()
    for obj in root.findall("object"):
        name = obj.findtext("name", default="object")
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            left = float(bnd.findtext("xmin"))
            top = float(bnd.findtext("ymin"))
            right = float(bnd.findtext("xmax"))
            bottom = float(bnd.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        boxes.append((name, None, np.asarray([left, top, right, bottom], dtype=np.float32)))
    return boxes


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0.0 else 0.0


def count_matches(
    preds: list[tuple[str, float | None, np.ndarray]],
    gts: list[tuple[str, float | None, np.ndarray]],
    iou_thr: float = 0.5,
) -> int:
    used_preds: set[int] = set()
    matched = 0
    for gt_cls, _, gt_box in gts:
        best_iou = 0.0
        best_idx = None
        for idx, (pred_cls, _, pred_box) in enumerate(preds):
            if idx in used_preds or pred_cls != gt_cls:
                continue
            overlap = box_iou(pred_box, gt_box)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx is not None and best_iou >= iou_thr:
            used_preds.add(best_idx)
            matched += 1
    return matched


def parse_tta_scales(value: str) -> list[list[int]]:
    scales = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        size = int(item)
        scales.append([size, size])
    return scales


def run_one(
    args: argparse.Namespace,
    label: str,
    model_path: str,
    fusion_mode: str,
    use_tta: bool,
    gts: list[tuple[str, float | None, np.ndarray]],
) -> Image.Image:
    image_path = Path(args.image)
    image_id = image_path.stem
    radar_path = Path(args.radar_root) / f"{image_id}.npz"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")
    if not radar_path.exists():
        raise FileNotFoundError(f"Missing radar npz for {image_id}: {radar_path}")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Missing checkpoint: {model_path}")

    yolo = YOLO(
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
    image = Image.open(image_path).convert("RGB")
    with tempfile.TemporaryDirectory() as tmp:
        map_out = Path(tmp)
        (map_out / "detection-results").mkdir(parents=True, exist_ok=True)
        if use_tta:
            yolo.get_map_txt_tta(
                image_id,
                image,
                yolo.class_names,
                str(map_out),
                scales=parse_tta_scales(args.tta_scales),
                flip=args.tta_flip,
                radar_alpha=args.tta_radar_alpha,
                wbf_iou=args.tta_wbf_iou,
                radar_tau=args.tta_radar_tau,
                fusion=args.tta_fusion,
            )
        else:
            yolo.get_map_txt(image_id, image, yolo.class_names, str(map_out))
        boxes = read_detection_txt(map_out / "detection-results" / f"{image_id}.txt")

    color = (34, 197, 94) if label.lower().startswith("ours") else (239, 68, 68)
    result = draw_txt_detections(image, boxes, color, args.font)
    mode = f" TTA-{args.tta_fusion}" if use_tta else ""
    title = f"{label}{mode} conf>{args.confidence:g}, pred {len(boxes)}"
    if gts:
        title += f", TP {count_matches(boxes, gts)}/{len(gts)}"
    return add_title(result, title, args.font)


def main() -> None:
    args = parse_args()
    out = Path(args.out) if args.out else Path("outputs") / f"compare_{Path(args.image).stem}_scores.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)

    gts = read_voc_xml(Path(args.gt_xml)) if args.gt_xml else []
    baseline = run_one(args, "Baseline", args.baseline, args.baseline_fusion_mode, args.baseline_tta, gts)
    ours = run_one(args, "Ours", args.ours, args.ours_fusion_mode, args.ours_tta, gts)

    panels = [baseline, ours]
    if gts:
        image = Image.open(args.image).convert("RGB")
        gt = draw_txt_detections(image, gts, (59, 130, 246), args.font)
        panels.append(add_title(gt, "Ground Truth", args.font))

    h = max(panel.size[1] for panel in panels)
    total_w = sum(panel.size[0] for panel in panels) + 4 * (len(panels) - 1)
    panel = Image.new("RGB", (total_w, h), (255, 255, 255))
    x = 0
    for item in panels:
        panel.paste(item, (x, 0))
        x += item.size[0] + 4
    panel.save(out, quality=95)
    print(out)


if __name__ == "__main__":
    main()
