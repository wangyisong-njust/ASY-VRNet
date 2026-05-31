# ASY-VRNet Current Progress Snapshot

This folder summarizes the current reproduction progress using the completed bs128 baseline run.

## Key status

- Training completed: 100 epochs, 4 GPUs, batch size 128.
- Full paper-style evaluation completed for `best` and `last` checkpoints.
- The current strongest checkpoint by detection metric is `last_epoch_weights.pth`.
- A follow-up segmentation-focused fine-tuning run has completed; its full paper-style evaluation is still running.

## Main metrics

| metric | paper | current last | gap |
|---|---:|---:|---:|
| mAP50-95 | 42.80 | 33.44 | -9.36 |
| AR50-95 | 46.30 | 37.96 | -8.34 |
| mIoU_o | 74.70 | 79.96 | +5.26 |
| mIoU_d | 99.60 | 98.89 | -0.71 |
| mAP_da | 38.80 | 35.19 | -3.61 |
| mAP_di | 39.50 | 20.94 | -18.56 |
| mAP_sm | 36.70 | 31.64 | -5.06 |

## Figures

- Main metric comparison: `figures/main_metrics_vs_paper.png`
- Adverse-condition comparison: `figures/adverse_metrics_vs_paper.png`
- Training curves: `figures/training_curves_bs128.png`

## Visual samples

- `samples/23019_summary.jpg`
- `samples/24900_summary.jpg`
- `samples/50947_summary.jpg`
- `samples/17441_summary.jpg`

## Current interpretation

- Detection is still below the paper baseline, especially `mAP50-95` and `AR50-95`.
- Object segmentation is higher than the paper table under the current evaluator.
- Drivable-area segmentation is close but still lower than the paper value.
- The next check should focus on paper-exact metric definitions, batch-size effects, and detection loss weighting.
