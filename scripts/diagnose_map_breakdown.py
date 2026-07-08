#!/usr/bin/env python3
"""Compare two saved mAP outputs with COCO per-class breakdowns."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.utils_map import preprocess_dr, preprocess_gt


METRIC_KEYS = [
    "mAP50-95",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
    "mIoU_o",
    "mIoU_d",
]


def pct(value: float | int | None) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.3f}"


def resolve_map_out(path: str | Path) -> Path:
    root = Path(path)
    if (root / "ground-truth").is_dir() and (root / "detection-results").is_dir():
        return root
    if (root / "map_out" / "ground-truth").is_dir():
        return root / "map_out"
    raise FileNotFoundError(f"Cannot find map_out under {root}")


def read_classes(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def load_parent_metrics(map_out: Path) -> dict[str, float]:
    metrics_path = map_out.parent / "paper_metrics.json"
    if not metrics_path.exists():
        return {}
    with metrics_path.open() as f:
        return json.load(f)


def mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid) * 100.0)


def coco_breakdown(map_out: Path, class_names: list[str]) -> dict[str, Any]:
    gt_dir = map_out / "ground-truth"
    dr_dir = map_out / "detection-results"
    gt = preprocess_gt(str(gt_dir), class_names)
    dr = preprocess_dr(str(dr_dir), class_names)

    with tempfile.TemporaryDirectory(prefix="map_breakdown_") as tmpdir:
        gt_json = Path(tmpdir) / "gt.json"
        dr_json = Path(tmpdir) / "dr.json"
        gt_json.write_text(json.dumps(gt))
        dr_json.write_text(json.dumps(dr))
        coco_gt = COCO(str(gt_json))
        coco_dt = coco_gt.loadRes(str(dr_json)) if dr else COCO()
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()

    precision = evaluator.eval["precision"]
    recall = evaluator.eval["recall"]
    iou_thrs = evaluator.params.iouThrs
    area_labels = list(evaluator.params.areaRngLbl)
    max_dets = list(evaluator.params.maxDets)
    idx_50 = int(np.argmin(np.abs(iou_thrs - 0.50)))
    idx_75 = int(np.argmin(np.abs(iou_thrs - 0.75)))
    area_all = area_labels.index("all")
    max_100 = max_dets.index(100)

    per_class: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "AP50-95": mean_valid(precision[:, :, idx, area_all, max_100]),
            "AP50": mean_valid(precision[idx_50, :, idx, area_all, max_100]),
            "AP75": mean_valid(precision[idx_75, :, idx, area_all, max_100]),
            "AR100": mean_valid(recall[:, idx, area_all, max_100]),
            "AR50": mean_valid(recall[idx_50 : idx_50 + 1, idx, area_all, max_100]),
            "AR75": mean_valid(recall[idx_75 : idx_75 + 1, idx, area_all, max_100]),
        }

    area_stats: dict[str, dict[str, float]] = {}
    for area in ["all", "small", "medium", "large"]:
        area_idx = area_labels.index(area)
        area_stats[area] = {
            "AP50-95": mean_valid(precision[:, :, :, area_idx, max_100]),
            "AP50": mean_valid(precision[idx_50, :, :, area_idx, max_100]),
            "AP75": mean_valid(precision[idx_75, :, :, area_idx, max_100]),
            "AR100": mean_valid(recall[:, :, area_idx, max_100]),
        }

    return {"per_class": per_class, "area": area_stats, "gt": gt, "dr": dr}


def score_summary(scores: list[float]) -> dict[str, float | int]:
    if not scores:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "ge_0.05": 0,
            "ge_0.10": 0,
            "ge_0.30": 0,
        }
    arr = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ge_0.05": int(np.sum(arr >= 0.05)),
        "ge_0.10": int(np.sum(arr >= 0.10)),
        "ge_0.30": int(np.sum(arr >= 0.30)),
    }


def count_profile(map_out: Path, class_names: list[str], gt: dict[str, Any], dr: list[dict[str, Any]]) -> dict[str, Any]:
    gt_counts = Counter(class_names[ann["category_id"] - 1] for ann in gt["annotations"])
    pred_counts = Counter(class_names[item["category_id"] - 1] for item in dr)
    scores_by_class: dict[str, list[float]] = defaultdict(list)
    pred_by_image = Counter(item["image_id"] for item in dr)
    for item in dr:
        scores_by_class[class_names[item["category_id"] - 1]].append(float(item["score"]))

    image_ids = [str(image["id"]) for image in gt["images"]]
    empty_images = sum(1 for image_id in image_ids if pred_by_image[image_id] == 0)
    preds_per_image = [pred_by_image[image_id] for image_id in image_ids]

    per_class: dict[str, dict[str, Any]] = {}
    for name in class_names:
        summary = score_summary(scores_by_class[name])
        summary["gt_count"] = int(gt_counts[name])
        summary["pred_count"] = int(pred_counts[name])
        per_class[name] = summary

    all_scores = [float(item["score"]) for item in dr]
    overall = score_summary(all_scores)
    overall.update(
        {
            "images": len(image_ids),
            "empty_images": empty_images,
            "preds_per_image_mean": float(np.mean(preds_per_image)) if preds_per_image else 0.0,
            "preds_per_image_median": float(np.median(preds_per_image)) if preds_per_image else 0.0,
            "preds_per_image_p90": float(np.percentile(preds_per_image, 90)) if preds_per_image else 0.0,
        }
    )
    return {"overall": overall, "per_class": per_class}


def analyse_one(label: str, root: str | Path, class_names: list[str]) -> dict[str, Any]:
    map_out = resolve_map_out(root)
    coco = coco_breakdown(map_out, class_names)
    profile = count_profile(map_out, class_names, coco["gt"], coco["dr"])
    return {
        "label": label,
        "path": str(map_out.parent),
        "metrics": load_parent_metrics(map_out),
        "area": coco["area"],
        "per_class": coco["per_class"],
        "profile": profile,
    }


def delta_table(old: dict[str, Any], new: dict[str, Any], keys: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key in keys:
        old_value = old.get(key, float("nan"))
        new_value = new.get(key, float("nan"))
        rows.append(
            {
                "metric": key,
                "old": old_value,
                "new": new_value,
                "delta": float(new_value) - float(old_value),
            }
        )
    return rows


def enrich_comparison(old: dict[str, Any], new: dict[str, Any], class_names: list[str]) -> dict[str, Any]:
    per_class = []
    for name in class_names:
        old_cls = old["per_class"][name]
        new_cls = new["per_class"][name]
        old_prof = old["profile"]["per_class"][name]
        new_prof = new["profile"]["per_class"][name]
        per_class.append(
            {
                "class": name,
                "gt_count": new_prof["gt_count"],
                "AP50-95_old": old_cls["AP50-95"],
                "AP50-95_new": new_cls["AP50-95"],
                "AP50-95_delta": new_cls["AP50-95"] - old_cls["AP50-95"],
                "AP50_old": old_cls["AP50"],
                "AP50_new": new_cls["AP50"],
                "AP50_delta": new_cls["AP50"] - old_cls["AP50"],
                "AR100_old": old_cls["AR100"],
                "AR100_new": new_cls["AR100"],
                "AR100_delta": new_cls["AR100"] - old_cls["AR100"],
                "pred_count_old": old_prof["pred_count"],
                "pred_count_new": new_prof["pred_count"],
                "score_median_old": old_prof["median"],
                "score_median_new": new_prof["median"],
            }
        )
    per_class.sort(key=lambda item: item["AP50-95_delta"])

    area = []
    for name in ["small", "medium", "large"]:
        old_area = old["area"][name]
        new_area = new["area"][name]
        area.append(
            {
                "area": name,
                "AP50-95_old": old_area["AP50-95"],
                "AP50-95_new": new_area["AP50-95"],
                "AP50-95_delta": new_area["AP50-95"] - old_area["AP50-95"],
                "AR100_old": old_area["AR100"],
                "AR100_new": new_area["AR100"],
                "AR100_delta": new_area["AR100"] - old_area["AR100"],
            }
        )

    return {
        "overall": delta_table(old["metrics"], new["metrics"], METRIC_KEYS),
        "area": area,
        "per_class": per_class,
        "prediction_profile": {
            "old": old["profile"]["overall"],
            "new": new["profile"]["overall"],
        },
    }


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    old = report["old"]
    new = report["new"]
    cmp = report["comparison"]
    lines = [
        "# mAP Breakdown",
        "",
        f"- old: `{old['label']}` -> `{old['path']}`",
        f"- new: `{new['label']}` -> `{new['path']}`",
        "",
        "## Overall",
        md_table(
            ["metric", "old", "new", "delta"],
            [[row["metric"], pct(row["old"]), pct(row["new"]), pct(row["delta"])] for row in cmp["overall"]],
        ),
        "",
        "## Area",
        md_table(
            ["area", "AP old", "AP new", "delta", "AR old", "AR new", "delta"],
            [
                [
                    row["area"],
                    pct(row["AP50-95_old"]),
                    pct(row["AP50-95_new"]),
                    pct(row["AP50-95_delta"]),
                    pct(row["AR100_old"]),
                    pct(row["AR100_new"]),
                    pct(row["AR100_delta"]),
                ]
                for row in cmp["area"]
            ],
        ),
        "",
        "## Per Class",
        md_table(
            ["class", "GT", "AP old", "AP new", "delta", "AP50 delta", "AR delta", "pred old", "pred new", "score med old", "score med new"],
            [
                [
                    row["class"],
                    str(row["gt_count"]),
                    pct(row["AP50-95_old"]),
                    pct(row["AP50-95_new"]),
                    pct(row["AP50-95_delta"]),
                    pct(row["AP50_delta"]),
                    pct(row["AR100_delta"]),
                    str(row["pred_count_old"]),
                    str(row["pred_count_new"]),
                    pct(row["score_median_old"]),
                    pct(row["score_median_new"]),
                ]
                for row in cmp["per_class"]
            ],
        ),
        "",
        "## Prediction Profile",
        md_table(
            ["label", "pred", "empty images", "pred/img mean", "score median", "score p90", ">=0.05", ">=0.10", ">=0.30"],
            [
                [
                    old["label"],
                    str(cmp["prediction_profile"]["old"]["count"]),
                    str(cmp["prediction_profile"]["old"]["empty_images"]),
                    pct(cmp["prediction_profile"]["old"]["preds_per_image_mean"]),
                    pct(cmp["prediction_profile"]["old"]["median"]),
                    pct(cmp["prediction_profile"]["old"]["p90"]),
                    str(cmp["prediction_profile"]["old"]["ge_0.05"]),
                    str(cmp["prediction_profile"]["old"]["ge_0.10"]),
                    str(cmp["prediction_profile"]["old"]["ge_0.30"]),
                ],
                [
                    new["label"],
                    str(cmp["prediction_profile"]["new"]["count"]),
                    str(cmp["prediction_profile"]["new"]["empty_images"]),
                    pct(cmp["prediction_profile"]["new"]["preds_per_image_mean"]),
                    pct(cmp["prediction_profile"]["new"]["median"]),
                    pct(cmp["prediction_profile"]["new"]["p90"]),
                    str(cmp["prediction_profile"]["new"]["ge_0.05"]),
                    str(cmp["prediction_profile"]["new"]["ge_0.10"]),
                    str(cmp["prediction_profile"]["new"]["ge_0.30"]),
                ],
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True, help="Baseline metrics root or map_out directory.")
    parser.add_argument("--new", required=True, help="New metrics root or map_out directory.")
    parser.add_argument("--old_label", default="old")
    parser.add_argument("--new_label", default="new")
    parser.add_argument("--classes_path", default="model_data/waterscenes.txt")
    parser.add_argument("--out_json", default="reproduction_reports/map_breakdown.json")
    parser.add_argument("--out_md", default="reproduction_reports/map_breakdown.md")
    args = parser.parse_args()

    class_names = read_classes(args.classes_path)
    old = analyse_one(args.old_label, args.old, class_names)
    new = analyse_one(args.new_label, args.new, class_names)
    report = {
        "old": old,
        "new": new,
        "comparison": enrich_comparison(old, new, class_names),
    }

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2))
    write_markdown(out_md, report)
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
