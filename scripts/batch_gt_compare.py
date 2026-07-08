"""Batch-generate Baseline/Ours/Ground-Truth comparison figures.

This script is meant for evidence building: it keeps confidence labels on boxes,
counts true-positive GT matches with the same IoU rule for both models, writes a
summary CSV, and separates visual cases into best/tie/failure folders.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compare_detection import (
    add_title,
    count_matches,
    draw_txt_detections,
    parse_shape,
    parse_tta_scales,
    read_detection_txt,
    read_voc_xml,
)
from yolo import YOLO


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch GT comparison for ASY-VRNet.")
    parser.add_argument("--ids", nargs="*", default=None, help="Image ids to process.")
    parser.add_argument("--ids-file", default="", help="Optional txt file containing one image id per line.")
    parser.add_argument("--image-dir", default="image")
    parser.add_argument("--radar-root", default="dataset/VOCradar_5_frames")
    parser.add_argument("--gt-dir", required=True, help="Directory containing VOC XML labels.")
    parser.add_argument("--baseline", default="weights/baseline_best.pth")
    parser.add_argument("--ours", default="weights/final_greedy_soup.pth")
    parser.add_argument("--out-dir", default="outputs/batch_gt_compare")
    parser.add_argument("--classes-path", default="model_data/waterscenes.txt")
    parser.add_argument("--input-shape", type=parse_shape, default=[512, 512])
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--phi", default="l")
    parser.add_argument("--radar-legacy-preprocess", action="store_true")
    parser.add_argument("--radar-preserve-points", action="store_true", default=True)
    parser.add_argument("--no-radar-preserve-points", action="store_false", dest="radar_preserve_points")
    parser.add_argument("--radar-source-order", default="range,doppler,elevation,power")
    parser.add_argument("--radar-target-order", default="range,elevation,velocity,power")
    parser.add_argument("--baseline-fusion-mode", default="baseline")
    parser.add_argument("--ours-fusion-mode", default="baseline")
    parser.add_argument("--font", default="model_data/simhei.ttf")
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


def predict_txt(
    model: YOLO,
    image_id: str,
    image: Image.Image,
    tmp_root: Path,
    use_tta: bool,
    args: argparse.Namespace,
) -> list:
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


def save_panel(
    image: Image.Image,
    image_id: str,
    base_boxes: list,
    ours_boxes: list,
    gts: list,
    base_hit: int,
    ours_hit: int,
    args: argparse.Namespace,
    out_path: Path,
) -> None:
    baseline_panel = draw_txt_detections(image, base_boxes, (239, 68, 68), args.font)
    ours_panel = draw_txt_detections(image, ours_boxes, (34, 197, 94), args.font)
    gt_panel = draw_txt_detections(image, gts, (59, 130, 246), args.font)
    baseline_panel = add_title(
        baseline_panel,
        f"Baseline conf>{args.confidence:g}, pred {len(base_boxes)}, TP {base_hit}/{len(gts)}",
        args.font,
    )
    ours_mode = f" TTA-{args.tta_fusion}" if args.ours_tta else ""
    ours_panel = add_title(
        ours_panel,
        f"Ours{ours_mode} conf>{args.confidence:g}, pred {len(ours_boxes)}, TP {ours_hit}/{len(gts)}",
        args.font,
    )
    gt_panel = add_title(gt_panel, "Ground Truth", args.font)

    panels = [baseline_panel, ours_panel, gt_panel]
    height = max(panel.size[1] for panel in panels)
    width = sum(panel.size[0] for panel in panels) + 8
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.size[0] + 4
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=95)


def main() -> None:
    args = parse_args()
    ids = load_ids(args)
    if not ids:
        raise SystemExit("No ids given. Use --ids or --ids-file.")

    out_dir = Path(args.out_dir)
    for sub in ["best_cases", "tie_cases", "failure_cases", "missing"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    baseline = make_model(args, args.baseline, args.baseline_fusion_mode)
    ours = make_model(args, args.ours, args.ours_fusion_mode)

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for image_id in ids:
            image_path = find_image(Path(args.image_dir), image_id)
            radar_path = Path(args.radar_root) / f"{image_id}.npz"
            gt_path = Path(args.gt_dir) / f"{image_id}.xml"
            status = "ok"
            if image_path is None:
                status = "missing_image"
            elif not radar_path.exists():
                status = "missing_radar"
            elif not gt_path.exists():
                status = "missing_gt"
            if status != "ok":
                rows.append({
                    "image_id": image_id,
                    "status": status,
                    "baseline_tp": "",
                    "ours_tp": "",
                    "gt_count": "",
                    "delta": "",
                    "output": "",
                })
                continue

            image = Image.open(image_path).convert("RGB")
            gts = read_voc_xml(gt_path)
            base_boxes = predict_txt(baseline, image_id, image, tmp_root / "baseline", False, args)
            ours_boxes = predict_txt(ours, image_id, image, tmp_root / "ours", args.ours_tta, args)
            base_hit = count_matches(base_boxes, gts, args.match_iou)
            ours_hit = count_matches(ours_boxes, gts, args.match_iou)
            delta = ours_hit - base_hit
            if delta > 0:
                bucket = "best_cases"
            elif delta == 0:
                bucket = "tie_cases"
            else:
                bucket = "failure_cases"
            out_path = out_dir / bucket / f"compare_{image_id}_b{base_hit}_o{ours_hit}_gt{len(gts)}.jpg"
            save_panel(image, image_id, base_boxes, ours_boxes, gts, base_hit, ours_hit, args, out_path)
            rows.append({
                "image_id": image_id,
                "status": "ok",
                "baseline_tp": base_hit,
                "ours_tp": ours_hit,
                "gt_count": len(gts),
                "delta": delta,
                "baseline_boxes": len(base_boxes),
                "ours_boxes": len(ours_boxes),
                "output": str(out_path),
            })
            print(f"{image_id}: baseline={base_hit}/{len(gts)} ours={ours_hit}/{len(gts)} -> {out_path}")

    csv_path = out_dir / "summary.csv"
    fieldnames = [
        "image_id", "status", "baseline_tp", "ours_tp", "gt_count",
        "delta", "baseline_boxes", "ours_boxes", "output",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    wins = sum(1 for row in ok_rows if int(row["delta"]) > 0)
    ties = sum(1 for row in ok_rows if int(row["delta"]) == 0)
    losses = sum(1 for row in ok_rows if int(row["delta"]) < 0)
    print(f"summary: ok={len(ok_rows)} wins={wins} ties={ties} losses={losses} -> {csv_path}")


if __name__ == "__main__":
    main()
