#!/usr/bin/env python3
"""Audit radar/image/label/postprocess/evaluation consistency end to end."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval_paper_metrics import summarize_segmentation
from nets.efficient_vrnet import EfficientVRNet
from utils.dataloader import YoloDataset
from utils.radar_utils import load_radar_npz, reorder_radar_channels
from utils.utils import get_classes
from utils.utils_bbox import non_max_suppression, yolo_correct_boxes
from utils.utils_map import get_coco_map
from utils_seg.utils_metrics import per_class_iu


def parse_annotation_line(line: str) -> tuple[Path, list[tuple[int, int, int, int, int]]]:
    parts = line.split()
    image_path = Path(parts[0])
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    boxes = [tuple(int(float(v)) for v in token.split(",")) for token in parts[1:]]
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


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 2:4] = out[:, 2:4] - out[:, 0:2]
    out[:, 0:2] = out[:, 0:2] + out[:, 2:4] / 2
    return out


def sort_boxes(boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return boxes.reshape(0, 5)
    order = np.lexsort((boxes[:, 4], boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]))
    return boxes[order]


def manual_transformed_boxes(
    boxes: list[tuple[int, int, int, int, int]],
    image_size: tuple[int, int],
    input_shape: tuple[int, int],
) -> np.ndarray:
    if not boxes:
        return np.zeros((0, 5), dtype=np.float64)
    iw, ih = image_size
    _, nw, nh, dx, dy = letterbox_params(image_size, input_shape)
    out = []
    for left, top, right, bottom, cls_id in boxes:
        out.append(
            [
                left * nw / iw + dx,
                top * nh / ih + dy,
                right * nw / iw + dx,
                bottom * nh / ih + dy,
                cls_id,
            ]
        )
    out = np.asarray(out, dtype=np.float64)
    out[:, 0:2][out[:, 0:2] < 0] = 0
    h, w = input_shape
    out[:, 2][out[:, 2] > w] = w
    out[:, 3][out[:, 3] > h] = h
    box_w = out[:, 2] - out[:, 0]
    box_h = out[:, 3] - out[:, 1]
    out = out[np.logical_and(box_w > 1, box_h > 1)]
    return xyxy_to_cxcywh(out)


def raw_channel_errors(csv_path: Path, radar_npz_path: Path, image_size: tuple[int, int]) -> tuple[int, float, float]:
    if not csv_path.exists() or not radar_npz_path.exists():
        return 0, 0.0, 0.0
    df = pd.read_csv(csv_path)
    required = {"u", "v", "range", "doppler", "elevation", "power"}
    if df.empty or not required.issubset(df.columns):
        return 0, 0.0, 0.0

    raw = reorder_radar_channels(np.load(radar_npz_path)["arr_0"])
    src_h, src_w = raw.shape[-2:]
    iw, ih = image_size
    df = df.copy()
    df["x"] = np.floor(df["u"].to_numpy(np.float32) * src_w / iw).astype(np.int32)
    df["y"] = np.floor(df["v"].to_numpy(np.float32) * src_h / ih).astype(np.int32)
    keep = (df["x"] >= 0) & (df["x"] < src_w) & (df["y"] >= 0) & (df["y"] < src_h)
    df = df[keep]
    if df.empty:
        return 0, 0.0, 0.0

    df = df.sort_values("power", ascending=True).drop_duplicates(["x", "y"], keep="last")
    x = df["x"].to_numpy(np.int32)
    y = df["y"].to_numpy(np.int32)
    expected = np.stack(
        [
            df["range"].to_numpy(np.float32),
            df["elevation"].to_numpy(np.float32),
            df["doppler"].to_numpy(np.float32),
            df["power"].to_numpy(np.float32),
        ],
        axis=0,
    )
    observed = raw[:, y, x]
    errors = np.abs(observed - expected)
    return int(errors.shape[1]), float(np.mean(errors)), float(np.max(errors))


@dataclass
class SampleResult:
    image_id: str
    raw_shape: tuple[int, int, int]
    loaded_shape: tuple[int, int, int]
    dataloader_radar_shape: tuple[int, int, int]
    raw_channel_checked_pixels: int
    raw_channel_mean_abs_error: float
    raw_channel_max_abs_error: float
    dataloader_radar_max_abs_diff: float
    dataloader_box_max_abs_diff: float
    seg_label_min: int
    seg_label_max: int
    ok: bool


def audit_sample(line: str, args: argparse.Namespace) -> SampleResult:
    image_path, boxes = parse_annotation_line(line)
    image_id = image_path.stem
    with Image.open(image_path) as image:
        image_size = image.size

    raw_npz = np.load(Path(args.radar_root) / f"{image_id}.npz")["arr_0"]
    loaded = load_radar_npz(
        args.radar_root,
        image_id,
        image_size,
        tuple(args.input_shape),
        normalize=args.radar_normalize,
        align_mode=args.radar_align_mode,
        source_order=args.radar_source_order,
        target_order=args.radar_target_order,
        preserve_points=args.radar_preserve_points,
    )
    checked, mean_error, max_error = raw_channel_errors(
        Path(args.radar_csv_root) / f"{image_id}.csv",
        Path(args.radar_root) / f"{image_id}.npz",
        image_size,
    )

    dataset = YoloDataset(
        [line],
        input_shape=args.input_shape,
        num_classes=args.num_classes,
        num_classes_seg=args.num_seg_classes,
        epoch_length=1,
        radar_root=args.radar_root,
        mosaic=False,
        mixup=False,
        mosaic_prob=0,
        mixup_prob=0,
        seg_dataset_path=args.vocdevkit,
        train=False,
        special_aug_ratio=0,
        radar_align_mode=args.radar_align_mode,
        radar_normalize=args.radar_normalize,
        radar_preserve_points=args.radar_preserve_points,
        radar_source_order=args.radar_source_order,
        radar_target_order=args.radar_target_order,
    )
    _, dataset_boxes, dataset_radar, png, _ = dataset[0]
    radar_diff = float(np.max(np.abs(dataset_radar.astype(np.float32) - loaded.astype(np.float32))))
    expected_boxes = manual_transformed_boxes(boxes, image_size, tuple(args.input_shape))
    if len(expected_boxes) == 0 and len(dataset_boxes) == 0:
        box_diff = 0.0
    else:
        box_diff = float(np.max(np.abs(sort_boxes(dataset_boxes) - sort_boxes(expected_boxes))))

    ok = (
        raw_npz.shape[0] == args.radar_channels
        and tuple(loaded.shape) == (args.radar_channels, args.input_shape[0], args.input_shape[1])
        and max_error <= args.channel_tol
        and radar_diff <= args.channel_tol
        and box_diff <= 1e-4
        and int(np.max(png)) <= args.num_seg_classes
    )
    return SampleResult(
        image_id=image_id,
        raw_shape=tuple(int(x) for x in raw_npz.shape),
        loaded_shape=tuple(int(x) for x in loaded.shape),
        dataloader_radar_shape=tuple(int(x) for x in dataset_radar.shape),
        raw_channel_checked_pixels=checked,
        raw_channel_mean_abs_error=mean_error,
        raw_channel_max_abs_error=max_error,
        dataloader_radar_max_abs_diff=radar_diff,
        dataloader_box_max_abs_diff=box_diff,
        seg_label_min=int(np.min(png)),
        seg_label_max=int(np.max(png)),
        ok=ok,
    )


def audit_model_contract(args: argparse.Namespace) -> dict:
    model = EfficientVRNet(
        num_classes=args.num_classes,
        num_seg_classes=args.num_seg_classes,
        phi=args.phi,
        radar_in_channels=args.radar_channels,
        fusion_mode=args.fusion_mode,
        radar_dropout=0.0,
        task_loss_mode=args.task_loss,
    )
    radar_initial = model.backbone.backbone.radar_initial.proj
    return {
        "radar_initial_in_channels": int(radar_initial.in_channels),
        "radar_initial_out_channels": int(radar_initial.out_channels),
        "expected_input_channels": int(args.radar_channels),
        "ok": int(radar_initial.in_channels) == int(args.radar_channels),
    }


def audit_postprocess(args: argparse.Namespace) -> dict:
    image_shape = np.array([1080, 1920], dtype=np.float32)
    input_shape = np.array(args.input_shape, dtype=np.float32)
    box = np.array([[812.0, 339.0, 1469.0, 529.0]], dtype=np.float32)
    scale = min(input_shape[1] / image_shape[1], input_shape[0] / image_shape[0])
    nw = int(image_shape[1] * scale)
    nh = int(image_shape[0] * scale)
    dx = (input_shape[1] - nw) // 2
    dy = (input_shape[0] - nh) // 2
    box_lb = box.copy()
    box_lb[:, [0, 2]] = box_lb[:, [0, 2]] * nw / image_shape[1] + dx
    box_lb[:, [1, 3]] = box_lb[:, [1, 3]] * nh / image_shape[0] + dy
    xy = np.stack([(box_lb[:, 0] + box_lb[:, 2]) / 2 / input_shape[1], (box_lb[:, 1] + box_lb[:, 3]) / 2 / input_shape[0]], axis=-1)
    wh = np.stack([(box_lb[:, 2] - box_lb[:, 0]) / input_shape[1], (box_lb[:, 3] - box_lb[:, 1]) / input_shape[0]], axis=-1)
    corrected = yolo_correct_boxes(xy, wh, input_shape, image_shape, True)
    corrected_xyxy = np.stack([corrected[:, 1], corrected[:, 0], corrected[:, 3], corrected[:, 2]], axis=-1)
    box_roundtrip_error = float(np.max(np.abs(corrected_xyxy - box)))

    empty_pred = torch.zeros((1, 3, 5 + args.num_classes), dtype=torch.float32)
    empty_nms = non_max_suppression(
        empty_pred,
        args.num_classes,
        args.input_shape,
        image_shape,
        True,
        conf_thres=0.5,
        nms_thres=0.5,
    )

    hist = np.array([[4, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.float64)
    ious = per_class_iu(hist)
    seg_summary = summarize_segmentation(
        np.array([0.9, 0.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.95]),
        np.array([1.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0]),
        np.array([1.0, 1.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.0]),
    )
    return {
        "letterbox_box_roundtrip_error": box_roundtrip_error,
        "empty_nms_is_none": empty_nms[0] is None,
        "absent_class_iou_is_nan": bool(np.isnan(ious[2])),
        "segmentation_nanmean_object_miou": float(seg_summary["mIoU_o"]),
        "ok": box_roundtrip_error <= 1e-3 and empty_nms[0] is None and bool(np.isnan(ious[2])),
    }


def audit_coco_identity(class_names: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        gt_dir = root / "ground-truth"
        dr_dir = root / "detection-results"
        gt_dir.mkdir()
        dr_dir.mkdir()
        boxes = [
            (class_names[0], 10, 10, 80, 80),
            (class_names[min(1, len(class_names) - 1)], 100, 90, 180, 170),
            (class_names[min(2, len(class_names) - 1)], 250, 120, 390, 260),
        ]
        with open(gt_dir / "00001.txt", "w", encoding="utf-8") as f_gt, open(dr_dir / "00001.txt", "w", encoding="utf-8") as f_dr:
            for cls, left, top, right, bottom in boxes:
                f_gt.write(f"{cls} {left} {top} {right} {bottom}\n")
                f_dr.write(f"{cls} 0.99 {left} {top} {right} {bottom}\n")
        stats = get_coco_map(class_names, str(root))
        stats = [float(x) for x in stats]
        return {
            "mAP50_95": stats[0] * 100,
            "AP50": stats[1] * 100,
            "AR100": stats[8] * 100,
            "ok": stats[0] > 0.999 and stats[1] > 0.999 and stats[8] > 0.999,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_txt", default=str(PROJECT_ROOT / "2007_val.txt"))
    parser.add_argument("--classes_path", default=str(PROJECT_ROOT / "model_data" / "waterscenes.txt"))
    parser.add_argument("--vocdevkit", default=os.environ.get("ASY_VOCDEVKIT", str(PROJECT_ROOT / "dataset" / "VOCdevkit")))
    parser.add_argument("--radar_root", default=os.environ.get("ASY_RADAR_ROOT", str(PROJECT_ROOT / "dataset" / "VOCradar_5_frames")))
    parser.add_argument("--radar_csv_root", default=str(PROJECT_ROOT / "dataset" / "WaterScenes_Full" / "radar_5_frames"))
    parser.add_argument("--input_shape", type=int, nargs=2, default=[320, 320])
    parser.add_argument("--samples", type=int, default=16)
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
    parser.add_argument("--channel_tol", type=float, default=1e-4)
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "reproduction_reports"))
    args = parser.parse_args()

    os.environ["ASY_RADAR_SOURCE_ORDER"] = args.radar_source_order
    os.environ["ASY_RADAR_TARGET_ORDER"] = args.radar_target_order
    os.environ["ASY_RADAR_PRESERVE_POINTS"] = "1" if args.radar_preserve_points else "0"
    class_names, num_classes = get_classes(args.classes_path)
    args.num_classes = num_classes

    lines = [line.strip() for line in Path(args.val_txt).read_text().splitlines() if line.strip()]
    sample_results = [audit_sample(line, args) for line in lines[: args.samples]]
    model_contract = audit_model_contract(args)
    postprocess = audit_postprocess(args)
    coco_identity = audit_coco_identity(class_names)

    ok = all(item.ok for item in sample_results) and model_contract["ok"] and postprocess["ok"] and coco_identity["ok"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "ok": ok,
        "config": {
            "radar_source_order": args.radar_source_order,
            "radar_target_order": args.radar_target_order,
            "radar_align_mode": args.radar_align_mode,
            "radar_normalize": args.radar_normalize,
            "radar_preserve_points": args.radar_preserve_points,
            "input_shape": args.input_shape,
            "num_classes": args.num_classes,
            "num_seg_classes": args.num_seg_classes,
        },
        "samples": [asdict(item) for item in sample_results],
        "summary": {
            "sample_count": len(sample_results),
            "max_raw_channel_abs_error": float(max(item.raw_channel_max_abs_error for item in sample_results)),
            "max_dataloader_radar_abs_diff": float(max(item.dataloader_radar_max_abs_diff for item in sample_results)),
            "max_dataloader_box_abs_diff": float(max(item.dataloader_box_max_abs_diff for item in sample_results)),
        },
        "model_contract": model_contract,
        "postprocess": postprocess,
        "coco_identity": coco_identity,
    }

    json_path = out_dir / f"end_to_end_pipeline_audit_{stamp}.json"
    md_path = out_dir / f"end_to_end_pipeline_audit_{stamp}.md"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    md_lines = [
        "# End-to-End Pipeline Audit",
        "",
        f"- Timestamp: {result['timestamp']}",
        f"- Status: {'PASS' if ok else 'FAIL'}",
        f"- Radar source order: {args.radar_source_order}",
        f"- Radar target order: {args.radar_target_order}",
        f"- Samples: {result['summary']['sample_count']}",
        f"- Max raw channel abs error: {result['summary']['max_raw_channel_abs_error']:.6f}",
        f"- Max dataloader radar abs diff: {result['summary']['max_dataloader_radar_abs_diff']:.6f}",
        f"- Max dataloader box abs diff: {result['summary']['max_dataloader_box_abs_diff']:.6f}",
        f"- Model radar input channels: {model_contract['radar_initial_in_channels']}",
        f"- Model radar initial output channels: {model_contract['radar_initial_out_channels']}",
        f"- Letterbox box roundtrip error: {postprocess['letterbox_box_roundtrip_error']:.6f}",
        f"- COCO identity mAP50-95: {coco_identity['mAP50_95']:.3f}",
        "",
        "This audit verifies that stored radar maps, loader channel reordering, dataloader outputs, model input contracts, detection postprocessing, and metric helpers use the same geometry and channel semantics.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    if not ok:
        print("End-to-end audit failed.")
        return 1
    print("End-to-end audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
