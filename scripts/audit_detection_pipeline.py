#!/usr/bin/env python3
"""Audit the ASY-VRNet detection data and decode path.

This script does not train or evaluate metrics. It checks the assumptions that
must be correct before a reproduction run is meaningful:

* WaterScenes class order.
* VOC/YOLO label conversion and box bounds.
* Letterbox box transform.
* Radar map sparsity and image/radar geometric alignment.
* YOLOX-style anchor-free head strides, decode and class-aware NMS.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nets.efficient_vrnet import EfficientVRNet
from utils.radar_utils import align_radar_map, load_radar_npz, normalize_radar
from utils.utils import get_classes
from utils.utils_bbox import decode_outputs, non_max_suppression


EXPECTED_WATERSCENES_CLASSES = [
    "pier",
    "buoy",
    "sailor",
    "ship",
    "boat",
    "vessel",
    "kayak",
]


@dataclass
class AuditIssue:
    severity: str
    area: str
    message: str


def add_issue(issues: list[AuditIssue], severity: str, area: str, message: str) -> None:
    issues.append(AuditIssue(severity=severity, area=area, message=message))


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def parse_annotation_line(line: str) -> tuple[Path, list[tuple[int, int, int, int, int]]]:
    parts = line.split()
    image_path = Path(parts[0])
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    boxes = []
    for box_text in parts[1:]:
        vals = box_text.split(",")
        if len(vals) != 5:
            raise ValueError(f"bad box token: {box_text}")
        boxes.append(tuple(int(float(v)) for v in vals))
    return image_path, boxes


def read_xml_boxes(xml_path: Path, classes: list[str]) -> list[tuple[int, int, int, int, int]]:
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.iter("object"):
        name_node = obj.find("name")
        bnd = obj.find("bndbox")
        if name_node is None or bnd is None:
            continue
        name = name_node.text
        if name not in classes:
            continue
        cls_id = classes.index(name)
        boxes.append(
            (
                int(float(bnd.findtext("xmin", "0"))),
                int(float(bnd.findtext("ymin", "0"))),
                int(float(bnd.findtext("xmax", "0"))),
                int(float(bnd.findtext("ymax", "0"))),
                cls_id,
            )
        )
    return boxes


def read_yolo_boxes(yolo_path: Path, image_size: tuple[int, int]) -> list[tuple[int, int, int, int, int]]:
    img_w, img_h = image_size
    boxes = []
    if not yolo_path.exists():
        return boxes
    for raw in read_lines(yolo_path):
        parts = raw.split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        xmin = max(0, int((cx - bw / 2.0) * img_w))
        ymin = max(0, int((cy - bh / 2.0) * img_h))
        xmax = min(img_w, int((cx + bw / 2.0) * img_w))
        ymax = min(img_h, int((cy + bh / 2.0) * img_h))
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax, cls_id))
    return boxes


def letterbox_boxes(
    boxes: list[tuple[int, int, int, int, int]],
    image_size: tuple[int, int],
    input_shape: tuple[int, int],
) -> list[tuple[float, float, float, float, int]]:
    iw, ih = image_size
    h, w = input_shape
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    out = []
    for xmin, ymin, xmax, ymax, cls_id in boxes:
        out.append(
            (
                xmin * nw / iw + dx,
                ymin * nh / ih + dy,
                xmax * nw / iw + dx,
                ymax * nh / ih + dy,
                cls_id,
            )
        )
    return out


def support_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def audit_classes(classes_path: Path, issues: list[AuditIssue]) -> dict:
    classes, num_classes = get_classes(str(classes_path))
    ok = classes == EXPECTED_WATERSCENES_CLASSES
    if not ok:
        add_issue(
            issues,
            "error",
            "classes",
            f"class order is {classes}, expected {EXPECTED_WATERSCENES_CLASSES}",
        )
    return {"classes": classes, "num_classes": num_classes, "matches_waterscenes": ok}


def audit_annotations(
    lines: list[str],
    classes: list[str],
    vocdevkit: Path,
    yolo_dir: Path,
    input_shape: tuple[int, int],
    sample_limit: int,
    issues: list[AuditIssue],
) -> dict:
    total_boxes = 0
    empty_images = 0
    xml_mismatch = 0
    yolo_mismatch = 0
    out_of_bounds = 0
    transformed_out_of_bounds = 0
    checked = min(len(lines), sample_limit)

    class_counts = {name: 0 for name in classes}
    samples = lines[:checked]
    for line in samples:
        try:
            image_path, boxes = parse_annotation_line(line)
        except Exception as exc:
            add_issue(issues, "error", "labels", f"failed to parse line: {exc}")
            continue

        if not image_path.exists():
            add_issue(issues, "error", "labels", f"missing image: {image_path}")
            continue

        with Image.open(image_path) as image:
            image_size = image.size
        img_w, img_h = image_size

        if not boxes:
            empty_images += 1

        total_boxes += len(boxes)
        for xmin, ymin, xmax, ymax, cls_id in boxes:
            if cls_id < 0 or cls_id >= len(classes):
                add_issue(issues, "error", "labels", f"class id {cls_id} out of range in {image_path.name}")
                continue
            class_counts[classes[cls_id]] += 1
            if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h or xmax <= xmin or ymax <= ymin:
                out_of_bounds += 1

        stem = image_path.stem
        xml_path = vocdevkit / "VOC2007" / "Annotations" / f"{stem}.xml"
        if xml_path.exists():
            xml_boxes = read_xml_boxes(xml_path, classes)
            if sorted(xml_boxes) != sorted(boxes):
                xml_mismatch += 1
        else:
            add_issue(issues, "error", "labels", f"missing xml: {xml_path}")

        yolo_path = yolo_dir / f"{stem}.txt"
        if yolo_path.exists():
            yolo_boxes = read_yolo_boxes(yolo_path, image_size)
            if sorted(yolo_boxes) != sorted(boxes):
                yolo_mismatch += 1

        transformed = letterbox_boxes(boxes, image_size, input_shape)
        h, w = input_shape
        for xmin, ymin, xmax, ymax, _ in transformed:
            if xmin < -1e-4 or ymin < -1e-4 or xmax > w + 1e-4 or ymax > h + 1e-4:
                transformed_out_of_bounds += 1

    if out_of_bounds:
        add_issue(issues, "error", "labels", f"{out_of_bounds} sampled boxes are out of image bounds")
    if transformed_out_of_bounds:
        add_issue(
            issues,
            "error",
            "labels",
            f"{transformed_out_of_bounds} sampled letterboxed boxes are out of input bounds",
        )
    if xml_mismatch:
        add_issue(issues, "error", "labels", f"{xml_mismatch} sampled lines differ from VOC XML")
    if yolo_mismatch:
        add_issue(issues, "warning", "labels", f"{yolo_mismatch} sampled lines differ from source YOLO txt")

    return {
        "checked_images": checked,
        "total_boxes": total_boxes,
        "empty_images": empty_images,
        "class_counts": class_counts,
        "out_of_bounds": out_of_bounds,
        "transformed_out_of_bounds": transformed_out_of_bounds,
        "xml_mismatch": xml_mismatch,
        "yolo_mismatch": yolo_mismatch,
    }


def audit_radar(
    lines: list[str],
    radar_root: Path,
    input_shape: tuple[int, int],
    sample_limit: int,
    issues: list[AuditIssue],
) -> dict:
    checked = 0
    missing = 0
    zero_after_load = 0
    dense_background_after_normalize = 0
    stats = []

    for line in lines[:sample_limit]:
        image_path, _ = parse_annotation_line(line)
        stem = image_path.stem
        radar_path = radar_root / f"{stem}.npz"
        if not radar_path.exists():
            missing += 1
            continue

        with Image.open(image_path) as image:
            image_size = image.size
        raw = np.load(radar_path)["arr_0"]
        if raw.ndim != 3 or raw.shape[0] != 4:
            add_issue(issues, "error", "radar", f"{radar_path.name} shape is {raw.shape}, expected [4,H,W]")
            continue

        letterbox = align_radar_map(raw, image_size, input_shape, align_mode="letterbox")
        direct = align_radar_map(raw, image_size, input_shape, align_mode="direct")
        loaded = load_radar_npz(str(radar_root), stem, image_size, input_shape, normalize=False, align_mode="letterbox")
        norm = normalize_radar(letterbox)

        raw_support = support_bbox(np.any(raw != 0, axis=0))
        lb_support = support_bbox(np.any(letterbox != 0, axis=0))
        direct_support = support_bbox(np.any(direct != 0, axis=0))

        if np.count_nonzero(loaded) == 0:
            zero_after_load += 1

        zero_mask = letterbox == 0
        if np.any(norm[zero_mask] != 0):
            dense_background_after_normalize += 1

        checked += 1
        if len(stats) < 5:
            stats.append(
                {
                    "id": stem,
                    "raw_shape": list(raw.shape),
                    "raw_nonzero": int(np.count_nonzero(raw)),
                    "letterbox_nonzero": int(np.count_nonzero(letterbox)),
                    "direct_nonzero": int(np.count_nonzero(direct)),
                    "raw_support_xyxy": raw_support,
                    "letterbox_support_xyxy": lb_support,
                    "direct_support_xyxy": direct_support,
                    "raw_min": float(np.nanmin(raw)),
                    "raw_max": float(np.nanmax(raw)),
                }
            )

    if missing:
        add_issue(issues, "error", "radar", f"{missing} sampled radar npz files are missing")
    if zero_after_load:
        add_issue(issues, "warning", "radar", f"{zero_after_load} sampled loaded radar maps are all zero")
    if dense_background_after_normalize:
        add_issue(issues, "error", "radar", "normalization made zero background nonzero")

    return {
        "checked": checked,
        "missing": missing,
        "zero_after_load": zero_after_load,
        "dense_background_after_normalize": dense_background_after_normalize,
        "sample_stats": stats,
        "interpretation": (
            "The current NPZ maps are generated by scaling u to width and v to height in a square map. "
            "Applying image letterbox at load time maps them back to the 320x320 image letterbox geometry."
        ),
    }


def audit_decode(
    num_classes: int,
    input_shape: tuple[int, int],
    phi: str,
    issues: list[AuditIssue],
    skip_model: bool,
) -> dict:
    h, w = input_shape
    if skip_model:
        head_shapes = [[1, 5 + num_classes, h // 8, w // 8], [1, 5 + num_classes, h // 16, w // 16], [1, 5 + num_classes, h // 32, w // 32]]
    else:
        model = EfficientVRNet(num_classes=num_classes, num_seg_classes=9, phi=phi)
        model.eval()
        with torch.no_grad():
            det_outputs, seg_output = model(
                torch.zeros(1, 3, h, w, dtype=torch.float32),
                torch.zeros(1, 4, h, w, dtype=torch.float32),
            )
        head_shapes = [list(t.shape) for t in det_outputs]
        seg_shape = list(seg_output.shape)
        expected_seg_shape = [1, 9, h, w]
        if seg_shape != expected_seg_shape:
            add_issue(issues, "error", "model", f"seg shape {seg_shape}, expected {expected_seg_shape}")

    inferred_strides = []
    fake_outputs = []
    for idx, shape in enumerate(head_shapes):
        _, channels, oh, ow = shape
        if channels != 5 + num_classes:
            add_issue(issues, "error", "decode", f"head {idx} channels={channels}, expected {5 + num_classes}")
        stride_y = h / oh
        stride_x = w / ow
        inferred_strides.append([stride_y, stride_x])
        if not math.isclose(stride_y, stride_x):
            add_issue(issues, "error", "decode", f"head {idx} non-square stride {stride_y} vs {stride_x}")
        fake = torch.full((1, 5 + num_classes, oh, ow), -20.0, dtype=torch.float32)
        cy, cx = oh // 2, ow // 2
        fake[0, 0:4, cy, cx] = 0.0
        fake[0, 4, cy, cx] = 10.0
        fake[0, 5, cy, cx] = 10.0
        fake_outputs.append(fake)

    if [round(s[0]) for s in inferred_strides] != [8, 16, 32]:
        add_issue(issues, "error", "decode", f"inferred strides are {inferred_strides}, expected [8,16,32]")

    decoded = decode_outputs(fake_outputs, list(input_shape))
    nms_result = non_max_suppression(
        decoded,
        num_classes,
        list(input_shape),
        np.array(input_shape),
        letterbox_image=True,
        conf_thres=0.5,
        nms_thres=0.5,
    )[0]
    nms_count = 0 if nms_result is None else int(len(nms_result))
    if nms_count == 0:
        add_issue(issues, "error", "decode", "synthetic high-confidence prediction was removed by decode/NMS")

    return {
        "head_shapes": head_shapes,
        "inferred_strides": inferred_strides,
        "synthetic_nms_count": nms_count,
    }


def write_reports(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"detection_pipeline_audit_{stamp}.json"
    md_path = out_dir / f"detection_pipeline_audit_{stamp}.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    issues = result["issues"]
    lines = [
        "# Detection Pipeline Audit",
        "",
        f"- Timestamp: {result['timestamp']}",
        f"- Train lines: {result['train_size']}",
        f"- Val lines: {result['val_size']}",
        f"- Input shape: {result['input_shape']}",
        "",
        "## Summary",
        "",
        f"- Errors: {sum(1 for i in issues if i['severity'] == 'error')}",
        f"- Warnings: {sum(1 for i in issues if i['severity'] == 'warning')}",
        "",
        "## Issues",
        "",
    ]
    if issues:
        for item in issues:
            lines.append(f"- {item['severity'].upper()} [{item['area']}]: {item['message']}")
    else:
        lines.append("- No blocking issues found in sampled checks.")
    lines.extend(
        [
            "",
            "## Radar Interpretation",
            "",
            result["radar"]["interpretation"],
            "",
            "## Decode",
            "",
            f"- Head shapes: {result['decode']['head_shapes']}",
            f"- Inferred strides: {result['decode']['inferred_strides']}",
            f"- Synthetic NMS detections: {result['decode']['synthetic_nms_count']}",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", default=str(PROJECT_ROOT / "2007_train.txt"))
    parser.add_argument("--val_txt", default=str(PROJECT_ROOT / "2007_val.txt"))
    parser.add_argument("--classes_path", default=str(PROJECT_ROOT / "model_data" / "waterscenes.txt"))
    parser.add_argument("--vocdevkit", default=os.environ.get("ASY_VOCDEVKIT", str(PROJECT_ROOT / "dataset" / "VOCdevkit")))
    parser.add_argument("--radar_root", default=os.environ.get("ASY_RADAR_ROOT", str(PROJECT_ROOT / "dataset" / "VOCradar_5_frames")))
    parser.add_argument("--waterscenes_full", default=str(PROJECT_ROOT / "dataset" / "WaterScenes_Full"))
    parser.add_argument("--input_shape", nargs=2, type=int, default=[320, 320])
    parser.add_argument("--phi", default=os.environ.get("ASY_PHI", "nano"))
    parser.add_argument("--sample_limit", type=int, default=512)
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "reproduction_reports"))
    parser.add_argument("--skip_model", action="store_true")
    args = parser.parse_args()

    issues: list[AuditIssue] = []
    classes_info = audit_classes(Path(args.classes_path), issues)
    classes = classes_info["classes"]
    train_lines = read_lines(Path(args.train_txt))
    val_lines = read_lines(Path(args.val_txt))
    input_shape = (int(args.input_shape[0]), int(args.input_shape[1]))
    yolo_dir = Path(args.waterscenes_full) / "detection" / "yolo"

    train_report = audit_annotations(
        train_lines,
        classes,
        Path(args.vocdevkit),
        yolo_dir,
        input_shape,
        args.sample_limit,
        issues,
    )
    val_report = audit_annotations(
        val_lines,
        classes,
        Path(args.vocdevkit),
        yolo_dir,
        input_shape,
        args.sample_limit,
        issues,
    )
    radar_report = audit_radar(
        val_lines,
        Path(args.radar_root),
        input_shape,
        args.sample_limit,
        issues,
    )
    decode_report = audit_decode(
        len(classes),
        input_shape,
        args.phi,
        issues,
        skip_model=args.skip_model,
    )

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_shape": list(input_shape),
        "train_size": len(train_lines),
        "val_size": len(val_lines),
        "classes": classes_info,
        "train_annotations": train_report,
        "val_annotations": val_report,
        "radar": radar_report,
        "decode": decode_report,
        "issues": [asdict(issue) for issue in issues],
    }

    json_path, md_path = write_reports(result, Path(args.out_dir))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        print(f"Audit finished with {len(errors)} error(s).")
        return 1
    print("Audit passed without blocking errors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
