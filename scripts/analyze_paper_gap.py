#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


PAPER_TARGETS = {
    # Tables II-VI in ASY-VRNet paper.
    "mAP50-95": 42.8,
    "AR50-95": 46.3,
    "mIoU_o": 74.7,
    "mIoU_d": 99.6,
    "mAP_da": 38.8,
    "mIoUda_d": 93.7,
    "mAP_di": 39.5,
    "mIoUdi_d": 95.6,
    "mAP_sm": 36.7,
    "mIoUsm_o": 68.8,
}

TOLERANCE = {
    "mAP50-95": 1.0,
    "AR50-95": 1.0,
    "mIoU_o": 1.0,
    "mIoU_d": 0.3,
    "mAP_da": 2.0,
    "mIoUda_d": 2.0,
    "mAP_di": 2.0,
    "mIoUdi_d": 2.0,
    "mAP_sm": 2.0,
    "mIoUsm_o": 2.0,
}


def load_metrics(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def metric_value(metrics, key):
    value = metrics.get(key)
    if value is None and key == "AR50-95":
        value = metrics.get("AR100")
    return value


def compare_one(name, metrics):
    rows = []
    failed = []
    for key, target in PAPER_TARGETS.items():
        observed = metric_value(metrics, key)
        if observed is None:
            rows.append((key, target, None, None, "missing"))
            failed.append(key)
            continue
        gap = observed - target
        ok = abs(gap) <= TOLERANCE[key]
        rows.append((key, target, observed, gap, "ok" if ok else "gap"))
        if not ok:
            failed.append(key)
    return rows, failed


def format_table(title, rows):
    lines = [f"## {title}", "", "| metric | paper | observed | gap | status |", "|---|---:|---:|---:|---|"]
    for key, target, observed, gap, status in rows:
        observed_text = "NA" if observed is None else f"{observed:.3f}"
        gap_text = "NA" if gap is None else f"{gap:+.3f}"
        lines.append(f"| {key} | {target:.3f} | {observed_text} | {gap_text} | {status} |")
    lines.append("")
    return lines


def build_diagnosis(best_metrics, last_metrics, failed_best, failed_last):
    lines = ["## Diagnosis", ""]
    best_map = metric_value(best_metrics, "mAP50-95")
    last_map = metric_value(last_metrics, "mAP50-95")
    best_miou = metric_value(best_metrics, "mIoU_o")
    last_miou = metric_value(last_metrics, "mIoU_o")

    if not failed_best:
        lines.append("- Best checkpoint is within the reproduction tolerance for the tracked paper metrics.")
    else:
        lines.append("- Best checkpoint still has metric gaps: " + ", ".join(failed_best) + ".")

    if best_map is not None and last_map is not None:
        if last_map + 0.5 < best_map:
            lines.append("- Last checkpoint is worse than best on detection, so use best weights for paper comparison and inspect late-epoch overfitting.")
        elif last_map > best_map + 0.5:
            lines.append("- Last checkpoint is better than best on detection; best checkpoint selection may be tied to validation loss rather than paper mAP.")

    if best_miou is not None and last_miou is not None and abs(last_miou - best_miou) > 1.0:
        lines.append("- Segmentation differs noticeably between best and last; verify whether checkpoint selection should optimize detection, segmentation, or a joint score.")

    if failed_best or failed_last:
        lines.extend(
            [
                "- Next code-level checks should focus on paper-exact settings: 5-frame radar maps, 320x320 input, full WaterScenes validation split, 7 detection classes, 9 segmentation classes, AFF baseline fusion, and uncertainty loss.",
                "- If mAP/AR are low while segmentation matches, prioritize detection decoding, box scaling after letterbox resize, class-name/order mapping, NMS confidence, and COCO mAP conversion.",
                "- If adverse subset metrics are low or missing, verify `information_list.csv` parsing and dark/dim/small subset definitions against the paper.",
                "- If all metrics are low, verify radar NPZ alignment with image ids and that the evaluation uses the same `VOCradar_5_frames` directory as training.",
            ]
        )
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", required=True, help="paper_metrics.json for best checkpoint")
    parser.add_argument("--last", required=True, help="paper_metrics.json for last checkpoint")
    parser.add_argument("--out", required=True, help="Markdown report path")
    parser.add_argument("--json_out", default=None, help="Optional JSON summary path")
    args = parser.parse_args()

    best_metrics = load_metrics(args.best)
    last_metrics = load_metrics(args.last)
    best_rows, failed_best = compare_one("best", best_metrics)
    last_rows, failed_last = compare_one("last", last_metrics)

    lines = [
        "# ASY-VRNet Paper Reproduction Gap Report",
        "",
        "Paper targets are taken from ASY-VRNet Tables II, III, V and VI.",
        "",
    ]
    lines.extend(format_table("Best Checkpoint", best_rows))
    lines.extend(format_table("Last Checkpoint", last_rows))
    lines.extend(build_diagnosis(best_metrics, last_metrics, failed_best, failed_last))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "best_failed_metrics": failed_best,
        "last_failed_metrics": failed_last,
        "best_reproduced": not failed_best,
        "last_reproduced": not failed_last,
    }
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved gap report to {out_path}")
    if failed_best:
        print("Best checkpoint still has gaps:", ", ".join(failed_best))
    else:
        print("Best checkpoint is within tolerance.")


if __name__ == "__main__":
    main()
