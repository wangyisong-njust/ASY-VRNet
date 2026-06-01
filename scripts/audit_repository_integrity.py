#!/usr/bin/env python3
"""Comprehensive ASY-VRNet reproduction audit.

The script checks dataset splits, labels, masks, radar maps, high-risk script
defaults, model contracts, postprocessing and metric helper invariants.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_paper_metrics import load_adverse_subsets
from nets.efficient_vrnet import EfficientVRNet
from scripts.audit_end_to_end_pipeline import (
    audit_coco_identity,
    audit_postprocess,
    audit_sample,
)
from utils.utils import get_classes


EXPECTED_DET_CLASSES = ["pier", "buoy", "sailor", "ship", "boat", "vessel", "kayak"]
SEG_CLASSES = ["background", "ship", "buoy", "sailor", "pier", "boat", "vessel", "kayak", "water"]
EXCLUDED_DIRS = {
    ".git",
    ".idea",
    "__pycache__",
    "venv",
    "logs",
    "logs_seg",
    "dataset",
    "downloads",
    "results_archives",
}


@dataclass
class Issue:
    level: str
    section: str
    message: str


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_lines(path: str | Path) -> list[str]:
    path = resolve_path(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stem_from_line(line: str) -> str:
    return Path(line.split()[0]).stem


def sample_items(items: list, limit: int) -> list:
    if limit <= 0 or len(items) <= limit:
        return items
    indices = np.linspace(0, len(items) - 1, limit, dtype=np.int64)
    return [items[int(i)] for i in indices]


def add_issue(issues: list[Issue], level: str, section: str, message: str):
    issues.append(Issue(level=level, section=section, message=message))


def parse_xml_size(xml_path: Path) -> tuple[int, int] | None:
    try:
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        if size is None:
            return None
        width = int(float(size.findtext("width", "0")))
        height = int(float(size.findtext("height", "0")))
        return width, height
    except Exception:
        return None


def audit_split(
    name: str,
    lines: list[str],
    classes: list[str],
    vocdevkit: Path,
    radar_root: Path,
    issues: list[Issue],
) -> dict:
    ids = []
    duplicate_ids = []
    seen_ids = set()
    missing = Counter()
    malformed = Counter()
    class_counts = Counter()
    area_values = []
    empty_lines = 0

    for idx, line in enumerate(lines, start=1):
        parts = line.split()
        if not parts:
            malformed["empty_line"] += 1
            continue
        image_path = resolve_path(parts[0])
        stem = image_path.stem
        ids.append(stem)
        if stem in seen_ids:
            duplicate_ids.append(stem)
        seen_ids.add(stem)

        xml_path = vocdevkit / "VOC2007" / "Annotations" / f"{stem}.xml"
        seg_path = vocdevkit / "VOC2007" / "SegmentationClass" / f"{stem}.png"
        radar_path = radar_root / f"{stem}.npz"
        if not image_path.exists():
            missing["image"] += 1
        if not xml_path.exists():
            missing["xml"] += 1
        if not seg_path.exists():
            missing["segmentation"] += 1
        if not radar_path.exists():
            missing["radar"] += 1

        image_size = parse_xml_size(xml_path) if xml_path.exists() else None
        if len(parts) == 1:
            empty_lines += 1
        for token in parts[1:]:
            fields = token.split(",")
            if len(fields) != 5:
                malformed["box_field_count"] += 1
                continue
            try:
                left, top, right, bottom = [float(v) for v in fields[:4]]
                cls_id = int(float(fields[4]))
            except ValueError:
                malformed["box_parse"] += 1
                continue
            if cls_id < 0 or cls_id >= len(classes):
                malformed["class_range"] += 1
                continue
            if right <= left or bottom <= top:
                malformed["non_positive_box"] += 1
            if image_size is not None:
                width, height = image_size
                if left < 0 or top < 0 or right > width or bottom > height:
                    malformed["box_out_of_bounds"] += 1
            class_counts[classes[cls_id]] += 1
            area_values.append(max(right - left, 0.0) * max(bottom - top, 0.0))

    for key, value in missing.items():
        if value:
            add_issue(issues, "error", f"{name}_files", f"{value} {key} files are missing")
    for key, value in malformed.items():
        if value:
            add_issue(issues, "error", f"{name}_labels", f"{value} malformed entries: {key}")
    if duplicate_ids:
        add_issue(issues, "error", f"{name}_split", f"{len(duplicate_ids)} duplicate image ids")

    area_array = np.asarray(area_values, dtype=np.float64)
    return {
        "images": len(lines),
        "unique_ids": len(seen_ids),
        "empty_box_lines": empty_lines,
        "objects": int(sum(class_counts.values())),
        "class_counts": dict(class_counts),
        "missing": dict(missing),
        "malformed": dict(malformed),
        "area_percentiles": np.percentile(area_array, [0, 25, 50, 75, 100]).round(2).tolist()
        if len(area_array)
        else [],
        "ids": ids,
    }


def audit_official_splits(train_ids: set[str], val_ids: set[str], dataset_root: Path, issues: list[Issue]) -> dict:
    result = {"available": False}
    train_path = dataset_root / "train.txt"
    val_path = dataset_root / "val.txt"
    if not train_path.exists() or not val_path.exists():
        add_issue(issues, "warning", "official_split", "WaterScenes official train/val txt files were not found")
        return result

    official_train = {Path(line.split()[0]).stem for line in read_lines(train_path)}
    official_val = {Path(line.split()[0]).stem for line in read_lines(val_path)}
    result = {
        "available": True,
        "train_matches": train_ids == official_train,
        "val_matches": val_ids == official_val,
        "train_missing_from_current": len(official_train - train_ids),
        "train_extra_in_current": len(train_ids - official_train),
        "val_missing_from_current": len(official_val - val_ids),
        "val_extra_in_current": len(val_ids - official_val),
    }
    if not result["train_matches"]:
        add_issue(issues, "error", "official_split", "Current train split does not match WaterScenes train.txt")
    if not result["val_matches"]:
        add_issue(issues, "error", "official_split", "Current val split does not match WaterScenes val.txt")
    return result


def audit_masks(lines: list[str], vocdevkit: Path, num_seg_classes: int, limit: int, issues: list[Issue]) -> dict:
    pixel_counts = Counter()
    invalid_masks = []
    size_mismatches = []
    checked = 0
    for line in sample_items(lines, limit):
        stem = stem_from_line(line)
        image_path = resolve_path(line.split()[0])
        seg_path = vocdevkit / "VOC2007" / "SegmentationClass" / f"{stem}.png"
        if not seg_path.exists() or not image_path.exists():
            continue
        with Image.open(seg_path) as mask:
            mask_array = np.asarray(mask)
            mask_size = mask.size
        with Image.open(image_path) as image:
            if image.size != mask_size:
                size_mismatches.append(stem)
        unique, counts = np.unique(mask_array, return_counts=True)
        for value, count in zip(unique.tolist(), counts.tolist()):
            pixel_counts[int(value)] += int(count)
        bad_values = [int(v) for v in unique if int(v) >= num_seg_classes and int(v) != 255]
        if bad_values:
            invalid_masks.append({"image_id": stem, "values": bad_values})
        checked += 1

    if invalid_masks:
        add_issue(issues, "error", "segmentation_masks", f"{len(invalid_masks)} sampled masks contain invalid class ids")
    if size_mismatches:
        add_issue(issues, "error", "segmentation_masks", f"{len(size_mismatches)} sampled masks do not match image size")
    return {
        "checked": checked,
        "pixel_values": dict(sorted(pixel_counts.items())),
        "invalid_masks": invalid_masks[:20],
        "size_mismatches": size_mismatches[:20],
    }


def audit_radar_dirs(dataset_root: Path, radar_root: Path, all_ids: set[str], issues: list[Issue]) -> dict:
    dirs = {
        "single_frame_csv": dataset_root / "radar",
        "three_frame_csv": dataset_root / "radar_3_frames",
        "five_frame_csv": dataset_root / "radar_5_frames",
        "npz_root": radar_root,
    }
    result = {}
    for name, path in dirs.items():
        suffix = "*.npz" if name == "npz_root" else "*.csv"
        count = len(list(path.glob(suffix))) if path.exists() else 0
        result[name] = {"path": str(path), "exists": path.exists(), "files": count}
        if not path.exists():
            add_issue(issues, "error", "radar_dirs", f"Missing {name}: {path}")

    npz_ids = {path.stem for path in radar_root.glob("*.npz")} if radar_root.exists() else set()
    missing_npz = all_ids - npz_ids
    result["npz_coverage"] = {
        "expected_ids": len(all_ids),
        "npz_ids": len(npz_ids),
        "missing_for_train_val": len(missing_npz),
    }
    if missing_npz:
        add_issue(issues, "error", "radar_dirs", f"{len(missing_npz)} train/val ids have no radar npz")
    return result


def audit_end_to_end_samples(lines: list[str], args: argparse.Namespace, class_count: int, issues: list[Issue]) -> dict:
    namespace = argparse.Namespace(
        radar_root=str(resolve_path(args.radar_root)),
        radar_csv_root=str(resolve_path(args.radar_csv_root)),
        vocdevkit=str(resolve_path(args.vocdevkit)),
        input_shape=args.input_shape,
        num_classes=class_count,
        num_seg_classes=args.num_seg_classes,
        radar_channels=args.radar_channels,
        radar_align_mode=args.radar_align_mode,
        radar_normalize=args.radar_normalize,
        radar_preserve_points=args.radar_preserve_points,
        radar_source_order=args.radar_source_order,
        radar_target_order=args.radar_target_order,
        channel_tol=args.channel_tol,
    )
    results = []
    for line in sample_items(lines, args.e2e_samples):
        item = audit_sample(line, namespace)
        results.append(item)
    failures = [item.image_id for item in results if not item.ok]
    if failures:
        add_issue(issues, "error", "end_to_end", f"{len(failures)} sampled end-to-end checks failed")
    return {
        "checked": len(results),
        "failed_ids": failures[:20],
        "max_raw_channel_abs_error": float(max((item.raw_channel_max_abs_error for item in results), default=0.0)),
        "max_dataloader_radar_abs_diff": float(max((item.dataloader_radar_max_abs_diff for item in results), default=0.0)),
        "max_dataloader_box_abs_diff": float(max((item.dataloader_box_max_abs_diff for item in results), default=0.0)),
    }


def audit_source_syntax(issues: list[Issue]) -> dict:
    checked = 0
    failures = []
    for path in PROJECT_ROOT.rglob("*.py"):
        rel_parts = set(path.relative_to(PROJECT_ROOT).parts)
        if rel_parts & EXCLUDED_DIRS:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{path.relative_to(PROJECT_ROOT)}: {exc}")
        checked += 1
    if failures:
        add_issue(issues, "error", "python_syntax", f"{len(failures)} Python files failed AST parsing")
    return {"checked": checked, "failures": failures[:20]}


def audit_high_risk_defaults(issues: list[Issue]) -> dict:
    checks = {
        "yolo.py": ["VOCradar_5_frames", "[320, 320]", "uncertainty"],
        "train.py": ["VOCradar_5_frames", "[320, 320]", "uncertainty"],
        "eval_paper_metrics.py": ["VOCradar_5_frames", "small_area_space", "nightfall", "uncertainty"],
        "scripts/check_dataset.py": ["VOCradar_5_frames"],
        "run_predict.py": ["VOCradar_5_frames"],
        "prepare_dataset.py": ["VOCradar_5_frames", "radar_5_frames"],
        "prepare_dataset_full.py": ["VOCradar_5_frames", "radar_5_frames", "ASY_MAX_SAMPLES", '"0"'],
        "deeplab.py": ["VOCradar_5_frames", "[320, 320]"],
        "MANUAL.md": ["VOCradar_5_frames", "radar_5_frames", "320,320"],
        "scripts/run_train_4gpu.sh": ["VOCradar_5_frames", "ASY_RADAR_PRESERVE_POINTS"],
    }
    result = {}
    for rel_path, required in checks.items():
        path = PROJECT_ROOT / rel_path
        text = path.read_text(encoding="utf-8")
        missing = [token for token in required if token not in text]
        result[rel_path] = {"missing_required_tokens": missing}
        if missing:
            add_issue(issues, "error", "script_defaults", f"{rel_path} is missing required defaults: {missing}")

    for rel_path in ["predict.py", "run_predict.py"]:
        text = (PROJECT_ROOT / rel_path).read_text(encoding="utf-8")
        if re.search(r"YOLO\(\s*\)", text):
            add_issue(issues, "error", "script_defaults", f"{rel_path} still constructs YOLO() without explicit config")

    legacy_deeplab = []
    for rel_path in ["predict_seg.py", "deeplab.py"]:
        path = PROJECT_ROOT / rel_path
        if path.exists() and "DeeplabV3" in path.read_text(encoding="utf-8"):
            legacy_deeplab.append(rel_path)
    if legacy_deeplab:
        add_issue(
            issues,
            "warning",
            "legacy_scripts",
            f"Legacy Deeplab-only scripts remain for reference, not paper ASY-VRNet evaluation: {legacy_deeplab}",
        )
    result["legacy_deeplab_scripts"] = legacy_deeplab
    return result


def audit_model_and_pretrain(args: argparse.Namespace, class_count: int, issues: list[Issue]) -> dict:
    model = EfficientVRNet(
        num_classes=class_count,
        num_seg_classes=args.num_seg_classes,
        phi=args.phi,
        radar_in_channels=args.radar_channels,
        fusion_mode=args.fusion_mode,
        radar_dropout=0.0,
        task_loss_mode=args.task_loss,
    )
    radar_initial = model.backbone.backbone.radar_initial.proj
    head_channels = [int(layer.out_channels) for layer in model.head.cls_preds]
    params_m = sum(param.numel() for param in model.parameters()) / 1e6
    result = {
        "phi": args.phi,
        "params_m": round(params_m, 4),
        "radar_initial_in_channels": int(radar_initial.in_channels),
        "radar_initial_out_channels": int(radar_initial.out_channels),
        "head_cls_out_channels": head_channels,
    }
    if int(radar_initial.in_channels) != args.radar_channels:
        add_issue(issues, "error", "model_contract", "Radar input channel count does not match configuration")
    if any(channel != class_count for channel in head_channels):
        add_issue(issues, "error", "model_contract", "Detection head class count does not match classes file")

    pretrain_path = resolve_path(args.coc_pretrained)
    result["coc_pretrained"] = {"path": str(pretrain_path), "exists": pretrain_path.exists()}
    if args.phi == "l" and not pretrain_path.exists():
        add_issue(issues, "warning", "pretrain", f"CoC pretrained checkpoint missing: {pretrain_path}")
    elif pretrain_path.exists():
        try:
            ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
            result["coc_pretrained"]["keys"] = len(state)
        except Exception as exc:
            add_issue(issues, "warning", "pretrain", f"Could not inspect CoC checkpoint: {exc}")
    return result


def audit_metrics_subsets(args: argparse.Namespace, val_lines: list[str], issues: list[Issue]) -> dict:
    image_ids = [stem_from_line(line) for line in val_lines]
    annotation_by_id = {stem_from_line(line): line for line in val_lines}
    info_csv = resolve_path(args.info_csv)
    dark_times = tuple(item.strip() for item in args.dark_times.split(",") if item.strip())
    subsets = load_adverse_subsets(
        str(info_csv),
        image_ids,
        annotation_by_id,
        args.input_shape,
        args.small_area,
        dark_times=dark_times,
        small_area_space=args.small_area_space,
    )
    counts = {name: len(ids) for name, ids in subsets.items()}

    night_only = 0
    nightfall = 0
    if info_csv.exists():
        val_set = set(image_ids)
        with open(info_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("id") not in val_set:
                    continue
                night_only += int(row.get("time") == "night")
                nightfall += int(row.get("time") == "nightfall")
    if "nightfall" in dark_times and counts["dark"] <= night_only and nightfall:
        add_issue(issues, "error", "metric_subsets", "Dark subset did not include nightfall samples")

    old_scaled_small = load_adverse_subsets(
        str(info_csv),
        image_ids,
        annotation_by_id,
        args.input_shape,
        args.small_area,
        dark_times=dark_times,
        small_area_space="input",
    )
    return {
        "dark_times": dark_times,
        "small_area": args.small_area,
        "small_area_space": args.small_area_space,
        "counts": counts,
        "night_only_count": night_only,
        "nightfall_count": nightfall,
        "input_space_small_count_for_comparison": len(old_scaled_small["small"]),
    }


def write_report(result: dict, issues: list[Issue], out_dir: Path) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"repository_integrity_audit_{stamp}.json"
    md_path = out_dir / f"repository_integrity_audit_{stamp}.md"
    serializable = {
        **result,
        "issues": [issue.__dict__ for issue in issues],
    }
    json_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")

    errors = [issue for issue in issues if issue.level == "error"]
    warnings = [issue for issue in issues if issue.level == "warning"]
    lines = [
        "# Repository Integrity Audit",
        "",
        f"- Timestamp: {result['timestamp']}",
        f"- Status: {'PASS' if not errors else 'FAIL'}",
        f"- Errors: {len(errors)}",
        f"- Warnings: {len(warnings)}",
        f"- Train images: {result['splits']['train']['images']}",
        f"- Val images: {result['splits']['val']['images']}",
        f"- E2E samples: {result['end_to_end']['checked']}",
        f"- Max dataloader box diff: {result['end_to_end']['max_dataloader_box_abs_diff']:.6f}",
        f"- Max radar channel error: {result['end_to_end']['max_raw_channel_abs_error']:.6f}",
        f"- COCO identity mAP50-95: {result['postprocess']['coco_identity']['mAP50_95']:.3f}",
        "",
        "## Issues",
    ]
    if issues:
        for issue in issues:
            lines.append(f"- {issue.level.upper()} [{issue.section}] {issue.message}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Metric Subsets",
            f"- Dark count: {result['metric_subsets']['counts']['dark']}",
            f"- Dim count: {result['metric_subsets']['counts']['dim']}",
            f"- Small count: {result['metric_subsets']['counts']['small']}",
            f"- Input-space small count, for comparison: {result['metric_subsets']['input_space_small_count_for_comparison']}",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def parse_args():
    parser = argparse.ArgumentParser(description="Audit ASY-VRNet repository integrity.")
    parser.add_argument("--train_txt", default=os.environ.get("ASY_TRAIN_TXT", "2007_train.txt"))
    parser.add_argument("--val_txt", default=os.environ.get("ASY_VAL_TXT", "2007_val.txt"))
    parser.add_argument("--classes_path", default=os.environ.get("ASY_CLASSES_PATH", "model_data/waterscenes.txt"))
    parser.add_argument("--vocdevkit", default=os.environ.get("ASY_VOCDEVKIT", str(PROJECT_ROOT / "dataset" / "VOCdevkit")))
    parser.add_argument("--dataset_root", default=str(PROJECT_ROOT / "dataset" / "WaterScenes_Full"))
    parser.add_argument("--radar_root", default=os.environ.get("ASY_RADAR_ROOT", str(PROJECT_ROOT / "dataset" / "VOCradar_5_frames")))
    parser.add_argument("--radar_csv_root", default=str(PROJECT_ROOT / "dataset" / "WaterScenes_Full" / "radar_5_frames"))
    parser.add_argument("--info_csv", default=os.environ.get("ASY_INFO_CSV", str(PROJECT_ROOT / "dataset" / "WaterScenes_Full" / "information_list.csv")))
    parser.add_argument("--input_shape", type=int, nargs=2, default=[320, 320])
    parser.add_argument("--num_seg_classes", type=int, default=9)
    parser.add_argument("--radar_channels", type=int, default=int(os.environ.get("ASY_RADAR_CHANNELS", "4")))
    parser.add_argument("--radar_align_mode", default=os.environ.get("ASY_RADAR_ALIGN_MODE", "letterbox"))
    parser.add_argument("--radar_normalize", action="store_true", default=os.environ.get("ASY_RADAR_NORMALIZE", "0").lower() in {"1", "true", "yes", "on"})
    parser.add_argument("--radar_preserve_points", action="store_true", default=os.environ.get("ASY_RADAR_PRESERVE_POINTS", "1").lower() not in {"0", "false", "no", "off"})
    parser.add_argument("--radar_source_order", default=os.environ.get("ASY_RADAR_SOURCE_ORDER", "range,doppler,elevation,power"))
    parser.add_argument("--radar_target_order", default=os.environ.get("ASY_RADAR_TARGET_ORDER", "range,elevation,velocity,power"))
    parser.add_argument("--phi", default=os.environ.get("ASY_PHI", "l"))
    parser.add_argument("--fusion_mode", default=os.environ.get("ASY_FUSION_MODE", "baseline"))
    parser.add_argument("--task_loss", default=os.environ.get("ASY_TASK_LOSS", "uncertainty"))
    parser.add_argument("--coc_pretrained", default="model_data/coc_small-bs128-lr0.001-wd0.05-dp0.0-distillnone-224/model_best.pth.tar")
    parser.add_argument("--mask_samples", type=int, default=1024)
    parser.add_argument("--e2e_samples", type=int, default=32)
    parser.add_argument("--channel_tol", type=float, default=1e-4)
    parser.add_argument("--small_area", type=float, default=32 * 32)
    parser.add_argument("--small_area_space", choices=["original", "input"], default=os.environ.get("ASY_SMALL_AREA_SPACE", "original"))
    parser.add_argument("--dark_times", default=os.environ.get("ASY_DARK_TIMES", "night,nightfall"))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "reproduction_reports"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    issues: list[Issue] = []

    classes, class_count = get_classes(str(resolve_path(args.classes_path)))
    if classes != EXPECTED_DET_CLASSES:
        add_issue(issues, "error", "classes", f"Detection class order differs from WaterScenes: {classes}")

    train_lines = read_lines(args.train_txt)
    val_lines = read_lines(args.val_txt)
    vocdevkit = resolve_path(args.vocdevkit)
    dataset_root = resolve_path(args.dataset_root)
    radar_root = resolve_path(args.radar_root)

    syntax = audit_source_syntax(issues)
    defaults = audit_high_risk_defaults(issues)
    train_split = audit_split("train", train_lines, classes, vocdevkit, radar_root, issues)
    val_split = audit_split("val", val_lines, classes, vocdevkit, radar_root, issues)
    train_ids = set(train_split.pop("ids"))
    val_ids = set(val_split.pop("ids"))
    overlap = train_ids & val_ids
    if overlap:
        add_issue(issues, "error", "splits", f"Train/val overlap contains {len(overlap)} ids")

    official = audit_official_splits(train_ids, val_ids, dataset_root, issues)
    masks = audit_masks(train_lines + val_lines, vocdevkit, args.num_seg_classes, args.mask_samples, issues)
    radar_dirs = audit_radar_dirs(dataset_root, radar_root, train_ids | val_ids, issues)
    model = audit_model_and_pretrain(args, class_count, issues)
    metric_subsets = audit_metrics_subsets(args, val_lines, issues)
    e2e = audit_end_to_end_samples(val_lines, args, class_count, issues)
    postprocess = audit_postprocess(argparse.Namespace(input_shape=args.input_shape, num_classes=class_count))
    coco_identity = audit_coco_identity(classes)
    if not postprocess["ok"]:
        add_issue(issues, "error", "postprocess", "Letterbox/NMS/segmentation postprocess invariant failed")
    if not coco_identity["ok"]:
        add_issue(issues, "error", "metrics", "COCO identity mAP is not 100%")

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "input_shape": args.input_shape,
            "radar_root": str(radar_root),
            "radar_csv_root": str(resolve_path(args.radar_csv_root)),
            "radar_source_order": args.radar_source_order,
            "radar_target_order": args.radar_target_order,
            "radar_preserve_points": args.radar_preserve_points,
            "small_area_space": args.small_area_space,
            "dark_times": args.dark_times,
        },
        "classes": {"detection": classes, "segmentation": SEG_CLASSES},
        "syntax": syntax,
        "script_defaults": defaults,
        "splits": {"train": train_split, "val": val_split, "overlap": len(overlap), "official": official},
        "segmentation_masks": masks,
        "radar_dirs": radar_dirs,
        "model": model,
        "metric_subsets": metric_subsets,
        "end_to_end": e2e,
        "postprocess": {"invariants": postprocess, "coco_identity": coco_identity},
    }

    json_path, md_path = write_report(result, issues, resolve_path(args.out_dir))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    errors = [issue for issue in issues if issue.level == "error"]
    if errors:
        print("Repository integrity audit failed.")
        return 1
    print("Repository integrity audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
