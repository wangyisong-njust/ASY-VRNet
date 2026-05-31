#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPECTED_CLASSES = ["pier", "buoy", "sailor", "ship", "boat", "vessel", "kayak"]
PAPER_PARAMS_M = 4.12


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_lines(path):
    return [line.strip() for line in resolve_path(path).read_text().splitlines() if line.strip()]


def class_names(path):
    return read_lines(path)


def detection_distribution(lines, names):
    counts = Counter()
    empty = 0
    areas = []
    for line in lines:
        parts = line.split()
        if len(parts) == 1:
            empty += 1
            continue
        for item in parts[1:]:
            x1, y1, x2, y2, cls_id = map(int, item.split(","))
            counts[names[cls_id]] += 1
            areas.append(max(x2 - x1, 0) * max(y2 - y1, 0))
    return counts, empty, np.asarray(areas, dtype=np.float64)


def file_missing(lines, vocdevkit, radar_root, limit):
    missing = Counter()
    for line in lines[:limit]:
        image_path = resolve_path(line.split()[0])
        stem = image_path.stem
        if not image_path.exists():
            missing["image"] += 1
        if not (Path(vocdevkit) / "VOC2007" / "SegmentationClass" / f"{stem}.png").exists():
            missing["segmentation"] += 1
        if not (Path(radar_root) / f"{stem}.npz").exists():
            missing["radar"] += 1
    return missing


def radar_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def audit_radar_alignment(image_id, vocdevkit, radar_root, radar_csv_root):
    image_path = Path(vocdevkit) / "VOC2007" / "JPEGImages" / f"{image_id}.jpg"
    npz_path = Path(radar_root) / f"{image_id}.npz"
    csv_path = Path(radar_csv_root) / f"{image_id}.csv"
    if not image_path.exists() or not npz_path.exists() or not csv_path.exists():
        return {"image_id": image_id, "status": "missing_input"}

    image = Image.open(image_path)
    raw = np.load(npz_path)["arr_0"]
    raw_bbox = radar_bbox(np.any(np.abs(raw) > 1e-6, axis=0))

    df = pd.read_csv(csv_path)
    direct_u = (df["u"] * 512 / image.size[0]).astype(int).clip(0, 511)
    direct_v = (df["v"] * 512 / image.size[1]).astype(int).clip(0, 511)

    scale = min(512 / image.size[0], 512 / image.size[1])
    nw = int(image.size[0] * scale)
    nh = int(image.size[1] * scale)
    dx = (512 - nw) // 2
    dy = (512 - nh) // 2
    letter_u = (df["u"] * scale + dx).astype(int).clip(0, 511)
    letter_v = (df["v"] * scale + dy).astype(int).clip(0, 511)

    direct_bbox = [int(direct_u.min()), int(direct_v.min()), int(direct_u.max()), int(direct_v.max())]
    letter_bbox = [int(letter_u.min()), int(letter_v.min()), int(letter_u.max()), int(letter_v.max())]
    return {
        "image_id": image_id,
        "image_size": list(image.size),
        "npz_shape": list(raw.shape),
        "npz_nonzero_bbox": raw_bbox,
        "csv_direct_512_bbox": direct_bbox,
        "csv_letterbox_512_bbox": letter_bbox,
        "interpretation": "npz_matches_direct_512" if raw_bbox == direct_bbox else "check_needed",
    }


def model_param_audit():
    try:
        from nets.efficient_vrnet import EfficientVRNet
    except Exception as exc:
        return {"error": repr(exc)}

    result = {}
    for phi in ["nano", "tiny", "s", "m", "l"]:
        model = EfficientVRNet(7, 9, phi, radar_in_channels=4, task_loss_mode="sum")
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        result[phi] = round(params_m, 4)
    closest = min(result, key=lambda key: abs(result[key] - PAPER_PARAMS_M))
    result["paper_params_m"] = PAPER_PARAMS_M
    result["closest_to_paper"] = closest
    return result


def script_defaults(path):
    text = Path(path).read_text()
    keys = [
        "EXP_NAME", "ASY_PHI", "ASY_BATCH_SIZE", "ASY_UNFREEZE_EPOCH",
        "ASY_INPUT_SHAPE", "ASY_RADAR_ROOT", "ASY_RADAR_ALIGN_MODE",
        "ASY_TASK_LOSS", "ASY_INIT_LR",
    ]
    values = {}
    for key in keys:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):]
            match = re.match(rf"{key}=\$\{{{key}:-(.*)\}}$", line)
            if match:
                values[key] = match.group(1)
                break
    return values


def main():
    parser = argparse.ArgumentParser(description="Audit ASY-VRNet reproduction-critical settings.")
    parser.add_argument("--train_txt", default="2007_train.txt")
    parser.add_argument("--val_txt", default="2007_val.txt")
    parser.add_argument("--classes_path", default="model_data/waterscenes.txt")
    parser.add_argument("--vocdevkit", default=os.environ.get("ASY_VOCDEVKIT", "dataset/VOCdevkit"))
    parser.add_argument("--radar_root", default=os.environ.get("ASY_RADAR_ROOT", "dataset/VOCradar_5_frames"))
    parser.add_argument("--radar_csv_root", default="dataset/WaterScenes_Full/radar_5_frames")
    parser.add_argument("--check_limit", type=int, default=200)
    parser.add_argument("--out", default="reproduction_reports/audit_reproduction.json")
    args = parser.parse_args()

    names = class_names(args.classes_path)
    train_lines = read_lines(args.train_txt)
    val_lines = read_lines(args.val_txt)
    train_counts, train_empty, train_areas = detection_distribution(train_lines, names)
    val_counts, val_empty, val_areas = detection_distribution(val_lines, names)

    sample_ids = [Path(line.split()[0]).stem for line in val_lines[:3]]
    report = {
        "classes": {
            "current": names,
            "expected_waterscenes": EXPECTED_CLASSES,
            "matches_expected": names == EXPECTED_CLASSES,
        },
        "splits": {
            "train_images": len(train_lines),
            "val_images": len(val_lines),
            "train_empty_box_lines": train_empty,
            "val_empty_box_lines": val_empty,
            "train_objects": dict(train_counts),
            "val_objects": dict(val_counts),
            "train_area_percentiles": np.percentile(train_areas, [0, 25, 50, 75, 100]).round(2).tolist(),
            "val_area_percentiles": np.percentile(val_areas, [0, 25, 50, 75, 100]).round(2).tolist(),
        },
        "missing_files": {
            "train": dict(file_missing(train_lines, args.vocdevkit, args.radar_root, args.check_limit)),
            "val": dict(file_missing(val_lines, args.vocdevkit, args.radar_root, args.check_limit)),
            "checked_per_split": args.check_limit,
        },
        "model_params_m": model_param_audit(),
        "radar_alignment_samples": [
            audit_radar_alignment(image_id, args.vocdevkit, args.radar_root, args.radar_csv_root)
            for image_id in sample_ids
        ],
        "paper_baseline_script_defaults": script_defaults("scripts/run_train_paper_baseline_4gpu.sh"),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved audit to {out}")


if __name__ == "__main__":
    main()
