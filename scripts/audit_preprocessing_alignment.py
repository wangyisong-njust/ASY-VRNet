#!/usr/bin/env python3
"""Check image/box/segmentation/radar preprocessing alignment.

The goal is to verify the exact tensors that enter training. For each sampled
frame this script:

* applies the same image and box letterbox transform as the training dataset;
* loads the NPZ radar map through `load_radar_npz`;
* projects the official WaterScenes CSV `u/v` points through the same letterbox
  geometry and compares them with the nonzero radar pixels;
* writes overlay images with GT boxes and radar points for manual inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.radar_utils import load_radar_npz, reorder_radar_channels
from utils_seg.utils import preprocess_input as preprocess_image


@dataclass
class SampleAudit:
    image_id: str
    image_size: tuple[int, int]
    boxes: int
    segmentation_unique_count: int
    radar_npz_nonzero: int
    radar_csv_points: int
    csv_points_in_canvas: int
    matched_csv_points: int
    mean_nearest_distance_px: float
    max_nearest_distance_px: float
    raw_channel_checked_pixels: int
    raw_channel_mean_abs_error: float
    raw_channel_max_abs_error: float
    image_tensor_min: float
    image_tensor_max: float
    image_tensor_mean: float
    overlay_path: str | None


def parse_annotation_line(line: str) -> tuple[Path, list[tuple[int, int, int, int, int]]]:
    parts = line.split()
    image_path = Path(parts[0])
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    boxes = []
    for token in parts[1:]:
        boxes.append(tuple(int(float(v)) for v in token.split(",")))
    return image_path, boxes


def letterbox_params(image_size: tuple[int, int], input_shape: tuple[int, int]) -> tuple[float, int, int, int, int]:
    iw, ih = image_size
    h, w = input_shape
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    return scale, nw, nh, dx, dy


def letterbox_image(image: Image.Image, input_shape: tuple[int, int]) -> Image.Image:
    h, w = input_shape
    _, nw, nh, dx, dy = letterbox_params(image.size, input_shape)
    resized = image.resize((nw, nh), Image.BICUBIC)
    canvas = Image.new("RGB", (w, h), (128, 128, 128))
    canvas.paste(resized, (dx, dy))
    return canvas


def transform_boxes(
    boxes: list[tuple[int, int, int, int, int]],
    image_size: tuple[int, int],
    input_shape: tuple[int, int],
) -> list[tuple[float, float, float, float, int]]:
    iw, ih = image_size
    _, nw, nh, dx, dy = letterbox_params(image_size, input_shape)
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


def csv_expected_pixels(csv_path: Path, image_size: tuple[int, int], input_shape: tuple[int, int]) -> np.ndarray:
    if not csv_path.exists():
        return np.zeros((0, 2), dtype=np.int32)
    df = pd.read_csv(csv_path)
    if df.empty or "u" not in df.columns or "v" not in df.columns:
        return np.zeros((0, 2), dtype=np.int32)
    iw, ih = image_size
    _, nw, nh, dx, dy = letterbox_params(image_size, input_shape)
    x = np.floor(df["u"].to_numpy(np.float32) * nw / iw + dx).astype(np.int32)
    y = np.floor(df["v"].to_numpy(np.float32) * nh / ih + dy).astype(np.int32)
    h, w = input_shape
    keep = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    return np.stack([x[keep], y[keep]], axis=1)


def raw_channel_errors(csv_path: Path, radar_npz_path: Path, image_size: tuple[int, int]) -> tuple[int, float, float]:
    if not csv_path.exists() or not radar_npz_path.exists():
        return 0, 0.0, 0.0

    df = pd.read_csv(csv_path)
    required = {"u", "v", "range", "doppler", "elevation", "power"}
    if df.empty or not required.issubset(df.columns):
        return 0, 0.0, 0.0

    raw = np.load(radar_npz_path)["arr_0"]
    raw = reorder_radar_channels(raw)
    src_h, src_w = raw.shape[-2:]
    iw, ih = image_size
    df = df.copy()
    df["x"] = np.floor(df["u"].to_numpy(np.float32) * src_w / iw).astype(np.int32)
    df["y"] = np.floor(df["v"].to_numpy(np.float32) * src_h / ih).astype(np.int32)
    keep = (df["x"] >= 0) & (df["x"] < src_w) & (df["y"] >= 0) & (df["y"] < src_h)
    df = df[keep]
    if df.empty:
        return 0, 0.0, 0.0

    # NPZ generation keeps the strongest return when several points hit one pixel.
    errors = []
    for (_, _), group in df.groupby(["x", "y"], sort=False):
        max_power = group["power"].max()
        candidates = group[np.isclose(group["power"], max_power, rtol=0.0, atol=1e-6)]
        expected = np.stack(
            [
                candidates["range"].to_numpy(np.float32),
                candidates["elevation"].to_numpy(np.float32),
                candidates["doppler"].to_numpy(np.float32),
                candidates["power"].to_numpy(np.float32),
            ],
            axis=1,
        )
        x = int(group["x"].iloc[0])
        y = int(group["y"].iloc[0])
        observed = raw[:, y, x]
        candidate_errors = np.abs(expected - observed[None, :])
        errors.append(candidate_errors[np.argmin(candidate_errors.max(axis=1))])

    if not errors:
        return 0, 0.0, 0.0
    errors = np.asarray(errors, dtype=np.float32)
    return int(errors.shape[0]), float(np.mean(errors)), float(np.max(errors))


def nearest_distances(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if len(points) == 0 or len(reference) == 0:
        return np.zeros((0,), dtype=np.float32)
    dists = []
    ref = reference.astype(np.float32)
    for point in points.astype(np.float32):
        diff = ref - point
        dists.append(float(np.sqrt(np.min(np.sum(diff * diff, axis=1)))))
    return np.asarray(dists, dtype=np.float32)


def draw_overlay(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float, int]],
    radar_points: np.ndarray,
    out_path: Path,
) -> None:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for x, y in radar_points:
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(255, 32, 32, 160))
    for xmin, ymin, xmax, ymax, cls_id in boxes:
        draw.rectangle((xmin, ymin, xmax, ymax), outline=(0, 255, 80, 230), width=2)
        draw.text((xmin + 2, max(0, ymin - 10)), str(cls_id), fill=(0, 255, 80, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)


def write_montage(overlay_paths: list[Path], out_path: Path, thumb_size: tuple[int, int] = (320, 320)) -> None:
    if not overlay_paths:
        return
    cols = min(4, len(overlay_paths))
    rows = int(np.ceil(len(overlay_paths) / cols))
    pad = 12
    tw, th = thumb_size
    canvas = Image.new("RGB", (cols * tw + (cols + 1) * pad, rows * th + (rows + 1) * pad), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, path in enumerate(overlay_paths):
        image = Image.open(path).convert("RGB").resize((tw, th))
        x = pad + (idx % cols) * (tw + pad)
        y = pad + (idx // cols) * (th + pad)
        canvas.paste(image, (x, y))
        draw.text((x + 6, y + 6), path.stem.replace("_preprocess_overlay", ""), fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def audit_sample(
    line: str,
    vocdevkit: Path,
    radar_root: Path,
    radar_csv_root: Path,
    input_shape: tuple[int, int],
    overlay_dir: Path | None,
) -> SampleAudit:
    image_path, boxes = parse_annotation_line(line)
    image_id = image_path.stem
    with Image.open(image_path) as im:
        image = im.convert("RGB")
    image_size = image.size
    image_lb = letterbox_image(image, input_shape)
    image_tensor = preprocess_image(np.asarray(image_lb, dtype=np.float64))

    seg_path = vocdevkit / "VOC2007" / "SegmentationClass" / f"{image_id}.png"
    with Image.open(seg_path) as seg:
        seg_lb = seg.resize((letterbox_params(image_size, input_shape)[1], letterbox_params(image_size, input_shape)[2]), Image.NEAREST)
        canvas = Image.new("L", (input_shape[1], input_shape[0]), 0)
        canvas.paste(seg_lb, (letterbox_params(image_size, input_shape)[3], letterbox_params(image_size, input_shape)[4]))
        seg_unique = np.unique(np.asarray(canvas))

    radar = load_radar_npz(str(radar_root), image_id, image_size, input_shape, normalize=False, align_mode="letterbox")
    radar_mask = np.any(radar != 0, axis=0)
    radar_y, radar_x = np.where(radar_mask)
    radar_points = np.stack([radar_x, radar_y], axis=1) if len(radar_x) else np.zeros((0, 2), dtype=np.int32)

    csv_points = csv_expected_pixels(radar_csv_root / f"{image_id}.csv", image_size, input_shape)
    raw_checked, raw_mean_error, raw_max_error = raw_channel_errors(
        radar_csv_root / f"{image_id}.csv",
        radar_root / f"{image_id}.npz",
        image_size,
    )
    dists = nearest_distances(csv_points, radar_points)
    matched = int(np.sum(dists <= 1.5)) if len(dists) else 0

    overlay_path = None
    if overlay_dir is not None:
        overlay_path = overlay_dir / f"{image_id}_preprocess_overlay.jpg"
        draw_overlay(image_lb, transform_boxes(boxes, image_size, input_shape), radar_points, overlay_path)

    return SampleAudit(
        image_id=image_id,
        image_size=image_size,
        boxes=len(boxes),
        segmentation_unique_count=int(len(seg_unique)),
        radar_npz_nonzero=int(np.count_nonzero(radar)),
        radar_csv_points=int(len(pd.read_csv(radar_csv_root / f"{image_id}.csv"))) if (radar_csv_root / f"{image_id}.csv").exists() else 0,
        csv_points_in_canvas=int(len(csv_points)),
        matched_csv_points=matched,
        mean_nearest_distance_px=float(np.mean(dists)) if len(dists) else 0.0,
        max_nearest_distance_px=float(np.max(dists)) if len(dists) else 0.0,
        raw_channel_checked_pixels=raw_checked,
        raw_channel_mean_abs_error=raw_mean_error,
        raw_channel_max_abs_error=raw_max_error,
        image_tensor_min=float(np.min(image_tensor)),
        image_tensor_max=float(np.max(image_tensor)),
        image_tensor_mean=float(np.mean(image_tensor)),
        overlay_path=str(overlay_path.relative_to(PROJECT_ROOT)) if overlay_path is not None else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_txt", default=str(PROJECT_ROOT / "2007_val.txt"))
    parser.add_argument("--vocdevkit", default=os.environ.get("ASY_VOCDEVKIT", str(PROJECT_ROOT / "dataset" / "VOCdevkit")))
    parser.add_argument("--radar_root", default=os.environ.get("ASY_RADAR_ROOT", str(PROJECT_ROOT / "dataset" / "VOCradar_5_frames")))
    parser.add_argument("--radar_csv_root", default=str(PROJECT_ROOT / "dataset" / "WaterScenes_Full" / "radar_5_frames"))
    parser.add_argument("--input_shape", type=int, nargs=2, default=[320, 320])
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--visuals", type=int, default=8)
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "reproduction_reports"))
    args = parser.parse_args()

    lines = [line.strip() for line in Path(args.val_txt).read_text().splitlines() if line.strip()]
    selected = lines[: args.samples]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "preprocess_visuals" / stamp

    audits = []
    for idx, line in enumerate(selected):
        audits.append(
            audit_sample(
                line,
                Path(args.vocdevkit),
                Path(args.radar_root),
                Path(args.radar_csv_root),
                tuple(args.input_shape),
                overlay_dir if idx < args.visuals else None,
            )
        )

    match_ratios = [
        item.matched_csv_points / item.csv_points_in_canvas
        for item in audits
        if item.csv_points_in_canvas > 0
    ]
    overlay_paths = sorted(overlay_dir.glob("*_preprocess_overlay.jpg"))
    montage_path = overlay_dir / "preprocess_alignment_montage.jpg"
    write_montage(overlay_paths, montage_path)
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_shape": args.input_shape,
        "radar_source_order": os.environ.get("ASY_RADAR_SOURCE_ORDER", "range,doppler,elevation,power"),
        "radar_target_order": os.environ.get("ASY_RADAR_TARGET_ORDER", "range,elevation,velocity,power"),
        "samples": [asdict(item) for item in audits],
        "summary": {
            "sample_count": len(audits),
            "mean_csv_to_npz_match_ratio": float(np.mean(match_ratios)) if match_ratios else 0.0,
            "min_csv_to_npz_match_ratio": float(np.min(match_ratios)) if match_ratios else 0.0,
            "mean_nearest_distance_px": float(np.mean([item.mean_nearest_distance_px for item in audits])),
            "max_nearest_distance_px": float(np.max([item.max_nearest_distance_px for item in audits])),
            "mean_raw_channel_abs_error": float(np.mean([item.raw_channel_mean_abs_error for item in audits])),
            "max_raw_channel_abs_error": float(np.max([item.raw_channel_max_abs_error for item in audits])),
            "overlays_dir": str(overlay_dir.relative_to(PROJECT_ROOT)),
            "montage_path": str(montage_path.relative_to(PROJECT_ROOT)) if montage_path.exists() else "",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"preprocessing_alignment_audit_{stamp}.json"
    md_path = out_dir / f"preprocessing_alignment_audit_{stamp}.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    lines_md = [
        "# Preprocessing Alignment Audit",
        "",
        f"- Timestamp: {result['timestamp']}",
        f"- Input shape: {args.input_shape}",
        f"- Radar source order: {result['radar_source_order']}",
        f"- Radar target order: {result['radar_target_order']}",
        f"- Samples: {result['summary']['sample_count']}",
        f"- Mean CSV-to-NPZ match ratio: {result['summary']['mean_csv_to_npz_match_ratio']:.4f}",
        f"- Min CSV-to-NPZ match ratio: {result['summary']['min_csv_to_npz_match_ratio']:.4f}",
        f"- Mean nearest distance px: {result['summary']['mean_nearest_distance_px']:.4f}",
        f"- Max nearest distance px: {result['summary']['max_nearest_distance_px']:.4f}",
        f"- Mean raw channel abs error: {result['summary']['mean_raw_channel_abs_error']:.6f}",
        f"- Max raw channel abs error: {result['summary']['max_raw_channel_abs_error']:.6f}",
        f"- Overlay directory: {result['summary']['overlays_dir']}",
        f"- Montage: {result['summary']['montage_path']}",
        "",
        "A match ratio close to 1.0, nearest distances near 0, and raw channel errors near 0 indicate that the radar NPZ pixels loaded by training align with the official WaterScenes CSV `u/v` coordinates and keep the expected REVP channel semantics after the same image letterbox transform.",
    ]
    md_path.write_text("\n".join(lines_md) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    if result["summary"]["min_csv_to_npz_match_ratio"] < 0.95:
        print("Preprocessing audit found radar alignment samples below 0.95 match ratio.")
        return 1
    if result["summary"]["max_raw_channel_abs_error"] > 1e-3:
        print("Preprocessing audit found radar channel values that do not match the source CSV.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
