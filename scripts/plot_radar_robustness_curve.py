#!/usr/bin/env python3
"""Radar-degradation robustness curve.

Evaluates one or more trained checkpoints at several inference-time radar drop
ratios (0 = full radar ... 1 = radar off) using the FROZEN paper evaluation
protocol, then writes a CSV and a mAP-vs-radar-availability plot. This is the
core robustness evidence for Innovation 3 (modality-dropout consistency):
a model trained with the consistency regularizer should degrade more
gracefully than the baseline as radar is removed.

Example (baseline vs innovation-3, same plot):

    python scripts/plot_radar_robustness_curve.py \
        --models "baseline=weights/baseline_best.pth=baseline" \
                 "innov3=logs_innovation3_consistency_phi_l_5frames_bs64_300e_320/best_epoch_weights.pth=baseline" \
        --ratios 0 0.25 0.5 0.75 1.0 \
        --out_root results/robustness
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Frozen paper evaluation protocol (matches docs/baseline_reproduction_record.md).
PAPER_EVAL_FLAGS = [
    "--phi", "l",
    "--input_shape", "320", "320",
    "--confidence", "0.001",
    "--max_boxes", "100",
    "--radar_root", "dataset/VOCradar_5_frames",
    "--radar_legacy_preprocess",
    "--no_radar_preserve_points",
    "--radar_source_order", "range,doppler,elevation,power",
    "--radar_target_order", "range,doppler,elevation,power",
    "--dark_times", "night",
    "--dim_lightings", "dim",
    "--dim_times", "daytime,night",
    "--dim_weathers", "overcast,rainy",
    "--small_area", "4096",
    "--small_area_space", "original",
]

# Metrics pulled from each paper_metrics.json into the curve CSV.
CURVE_METRICS = ["mAP50-95", "AP50", "AP75", "AR50-95", "mAP_da", "mAP_di", "mAP_sm", "mIoU_o"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--models", nargs="+", required=True,
        help="One or more 'label=model_path=fusion_mode' entries (fusion_mode optional, default baseline).",
    )
    p.add_argument("--ratios", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--out_root", default="results/robustness")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--metric", default="mAP50-95", help="Metric to plot on the y-axis.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Reuse an existing paper_metrics.json for a (model,ratio) instead of re-evaluating.")
    return p.parse_args()


def parse_model_entry(entry):
    parts = entry.split("=")
    if len(parts) == 2:
        label, model_path = parts
        fusion = "baseline"
    elif len(parts) == 3:
        label, model_path, fusion = parts
    else:
        raise ValueError(f"Bad --models entry: {entry!r}; expected label=path[=fusion]")
    return label, model_path, fusion


def run_eval(python, model_path, fusion, ratio, out_dir, skip_existing):
    metrics_json = Path(out_dir) / "paper_metrics.json"
    if skip_existing and metrics_json.exists():
        print(f"[skip] reuse {metrics_json}")
    else:
        cmd = [
            python, "eval_paper_metrics.py",
            "--model_path", model_path,
            "--fusion_mode", fusion,
            "--radar_drop_ratio", str(ratio),
            "--out_dir", str(out_dir),
            *PAPER_EVAL_FLAGS,
        ]
        print(f"[eval] {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
    with open(metrics_json, encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    out_root = PROJECT_ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    # results[label][ratio] = metrics dict
    results = {}
    for entry in args.models:
        label, model_path, fusion = parse_model_entry(entry)
        results[label] = {}
        for ratio in args.ratios:
            out_dir = out_root / f"{label}_drop{ratio}"
            metrics = run_eval(args.python, model_path, fusion, ratio, out_dir, args.skip_existing)
            results[label][ratio] = metrics
            print(f"  {label} drop={ratio}: {args.metric}={metrics.get(args.metric)}")

    # ---- write curve CSV ----
    csv_path = out_root / "robustness_curve.csv"
    import csv as _csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.writer(f)
        writer.writerow(["label", "radar_drop_ratio", "radar_availability"] + CURVE_METRICS)
        for label, by_ratio in results.items():
            for ratio in sorted(by_ratio):
                m = by_ratio[ratio]
                writer.writerow(
                    [label, ratio, round(1.0 - ratio, 3)] + [m.get(k) for k in CURVE_METRICS]
                )
    print(f"Saved curve CSV to {csv_path}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 5))
        for label, by_ratio in results.items():
            ratios = sorted(by_ratio)
            xs = [1.0 - r for r in ratios]  # radar availability
            ys = [by_ratio[r].get(args.metric) for r in ratios]
            plt.plot(xs, ys, marker="o", label=label)
        plt.xlabel("Radar availability (1 - drop ratio)")
        plt.ylabel(args.metric)
        plt.title(f"Radar-degradation robustness: {args.metric}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.gca().invert_xaxis()  # left = full radar, right = radar off
        png_path = out_root / "robustness_curve.png"
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {png_path}")
    except Exception as exc:  # plotting is best-effort
        print(f"Plot skipped ({exc}); CSV is still written.")


if __name__ == "__main__":
    main()
