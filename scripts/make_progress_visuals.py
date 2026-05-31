#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation" / "current_progress"
FIG = OUT / "figures"
SAMPLES = OUT / "samples"

PAPER_TARGETS = {
    "mAP50-95": 42.8,
    "AR50-95": 46.3,
    "mIoU_o": 74.7,
    "mIoU_d": 99.6,
    "mAP_da": 38.8,
    "mAP_di": 39.5,
    "mAP_sm": 36.7,
}

SEG_COLORS = np.array(
    [
        [0, 0, 0],
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
    ],
    dtype=np.uint8,
)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_series(path):
    p = ROOT / path
    if not p.exists():
        return []
    return [float(line.strip()) for line in p.read_text().splitlines() if line.strip()]


def save_bar_chart(metrics, keys, title, out_path):
    labels = keys
    paper = [PAPER_TARGETS[k] for k in keys]
    current = [metrics.get(k, np.nan) for k in keys]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    ax.bar(x - width / 2, paper, width, label="Paper", color="#6b7280")
    ax.bar(x + width / 2, current, width, label="Current", color="#2563eb")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, max(max(paper), max(current)) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    for i, value in enumerate(current):
        ax.text(i + width / 2, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_loss_chart(out_path):
    det_train = read_series("logs_paper_baseline_latestfix_ddpfix_5frames_uncert_bs128_100e_320/loss_2026_05_30_18_33_16/epoch_loss.txt")
    det_val = read_series("logs_paper_baseline_latestfix_ddpfix_5frames_uncert_bs128_100e_320/loss_2026_05_30_18_33_16/epoch_val_loss.txt")
    seg_train = read_series("logs_seg_paper_baseline_latestfix_ddpfix_5frames_uncert_bs128_100e_320/loss_2026_05_30_18_33_16/epoch_loss.txt")
    seg_val = read_series("logs_seg_paper_baseline_latestfix_ddpfix_5frames_uncert_bs128_100e_320/loss_2026_05_30_18_33_16/epoch_val_loss.txt")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)
    epochs = np.arange(1, len(det_train) + 1)
    axes[0].plot(epochs, det_train, label="train", color="#2563eb")
    axes[0].plot(epochs, det_val, label="val", color="#dc2626")
    axes[0].set_title("Detection Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    epochs = np.arange(1, len(seg_train) + 1)
    axes[1].plot(epochs, seg_train, label="train", color="#2563eb")
    axes[1].plot(epochs, seg_val, label="val", color="#dc2626")
    axes[1].set_title("Segmentation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_boxes(path):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            cls, left, top, right, bottom = parts
            score = None
        elif len(parts) == 6:
            cls, score, left, top, right, bottom = parts
        else:
            continue
        boxes.append((cls, score, tuple(map(float, (left, top, right, bottom)))))
    return boxes


def draw_boxes(image, boxes, color, with_score=False):
    draw = ImageDraw.Draw(image)
    w, h = image.size
    thick = max(2, int((w + h) / 700))
    for cls, score, box in boxes:
        left, top, right, bottom = box
        for t in range(thick):
            draw.rectangle([left + t, top + t, right - t, bottom - t], outline=color)
        label = cls if not with_score or score is None else f"{cls} {float(score):.2f}"
        draw.rectangle([left, max(0, top - 14), left + 7 * len(label) + 6, top], fill=color)
        draw.text((left + 3, max(0, top - 13)), label, fill=(255, 255, 255))


def colorize_mask(mask):
    mask = np.asarray(mask)
    mask = np.clip(mask, 0, len(SEG_COLORS) - 1)
    return Image.fromarray(SEG_COLORS[mask])


def overlay_mask(image, mask, alpha=0.45):
    color = colorize_mask(mask).resize(image.size, Image.NEAREST)
    return Image.blend(image.convert("RGB"), color.convert("RGB"), alpha)


def make_sample(image_id, out_path, pred_root):
    image_path = ROOT / "dataset" / "VOCdevkit" / "VOC2007" / "JPEGImages" / f"{image_id}.jpg"
    gt_seg_path = ROOT / "dataset" / "VOCdevkit" / "VOC2007" / "SegmentationClass" / f"{image_id}.png"
    pred_seg_path = pred_root / "segmentation-results" / f"{image_id}.png"
    gt_box_path = pred_root / "map_out" / "ground-truth" / f"{image_id}.txt"
    pred_box_path = pred_root / "map_out" / "detection-results" / f"{image_id}.txt"

    image = Image.open(image_path).convert("RGB")
    gt_boxes = parse_boxes(gt_box_path)
    pred_boxes = parse_boxes(pred_box_path)

    det = image.copy()
    draw_boxes(det, gt_boxes, (34, 197, 94), with_score=False)
    draw_boxes(det, pred_boxes, (239, 68, 68), with_score=True)

    gt_overlay = overlay_mask(image, Image.open(gt_seg_path))
    pred_overlay = overlay_mask(image, Image.open(pred_seg_path))

    panels = [image, det, gt_overlay, pred_overlay]
    titles = ["Image", "GT green / Pred red", "GT segmentation", "Pred segmentation"]
    thumb_w = 640
    thumb_h = int(image.size[1] * thumb_w / image.size[0])
    canvas = Image.new("RGB", (thumb_w * 2, (thumb_h + 28) * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, panel in enumerate(panels):
        panel = panel.resize((thumb_w, thumb_h), Image.BICUBIC)
        x = (idx % 2) * thumb_w
        y = (idx // 2) * (thumb_h + 28)
        draw.text((x + 8, y + 6), titles[idx], fill=(20, 20, 20))
        canvas.paste(panel, (x, y + 28))
    canvas.save(out_path, quality=92)


def find_sample_ids(metrics_root, limit=4):
    dr_dir = metrics_root / "map_out" / "detection-results"
    gt_dir = metrics_root / "map_out" / "ground-truth"
    candidates = []
    for path in sorted(dr_dir.glob("*.txt")):
        image_id = path.stem
        pred_count = len(parse_boxes(path))
        gt_count = len(parse_boxes(gt_dir / f"{image_id}.txt"))
        if gt_count and pred_count:
            candidates.append((abs(pred_count - gt_count), -gt_count, image_id))
    candidates.sort()
    return [item[2] for item in candidates[:limit]]


def write_readme(metrics, sample_ids):
    lines = [
        "# ASY-VRNet Current Progress Snapshot",
        "",
        "This folder summarizes the current reproduction progress using the completed bs128 baseline run.",
        "",
        "## Key status",
        "",
        "- Training completed: 100 epochs, 4 GPUs, batch size 128.",
        "- Full paper-style evaluation completed for `best` and `last` checkpoints.",
        "- The current strongest checkpoint by detection metric is `last_epoch_weights.pth`.",
        "- A follow-up segmentation-focused fine-tuning run has completed; its full paper-style evaluation is still running.",
        "",
        "## Main metrics",
        "",
        "| metric | paper | current last | gap |",
        "|---|---:|---:|---:|",
    ]
    for key in ["mAP50-95", "AR50-95", "mIoU_o", "mIoU_d", "mAP_da", "mAP_di", "mAP_sm"]:
        current = metrics.get(key, float("nan"))
        target = PAPER_TARGETS[key]
        lines.append(f"| {key} | {target:.2f} | {current:.2f} | {current - target:+.2f} |")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- Main metric comparison: `figures/main_metrics_vs_paper.png`",
            "- Adverse-condition comparison: `figures/adverse_metrics_vs_paper.png`",
            "- Training curves: `figures/training_curves_bs128.png`",
            "",
            "## Visual samples",
            "",
        ]
    )
    for image_id in sample_ids:
        lines.append(f"- `samples/{image_id}_summary.jpg`")
    lines.extend(
        [
            "",
            "## Current interpretation",
            "",
            "- Detection is still below the paper baseline, especially `mAP50-95` and `AR50-95`.",
            "- Object segmentation is higher than the paper table under the current evaluator.",
            "- Drivable-area segmentation is close but still lower than the paper value.",
            "- The next check should focus on paper-exact metric definitions, batch-size effects, and detection loss weighting.",
            "",
        ]
    )
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)

    metrics_root = ROOT / "paper_metrics_bs128_last_final"
    metrics = load_json(metrics_root / "paper_metrics.json")

    save_bar_chart(
        metrics,
        ["mAP50-95", "AR50-95", "mIoU_o", "mIoU_d"],
        "Main Metrics vs Paper",
        FIG / "main_metrics_vs_paper.png",
    )
    save_bar_chart(
        metrics,
        ["mAP_da", "mAP_di", "mAP_sm"],
        "Adverse Detection Metrics vs Paper",
        FIG / "adverse_metrics_vs_paper.png",
    )
    save_loss_chart(FIG / "training_curves_bs128.png")

    sample_ids = find_sample_ids(metrics_root, limit=4)
    for image_id in sample_ids:
        make_sample(image_id, SAMPLES / f"{image_id}_summary.jpg", metrics_root)

    write_readme(metrics, sample_ids)
    print(OUT)
    print("samples:", ", ".join(sample_ids))


if __name__ == "__main__":
    main()
