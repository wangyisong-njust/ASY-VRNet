# Official Alignment Findings - 2026-06-01

## Summary

The current repository was compared against `GuanRunwei/ASY-VRNet` (`official/main`).
The official GitHub release is not a complete paper reproduction package: it does
not include pretrained weights, full train annotations, radar conversion code, or
paper-style metric evaluation. Some released defaults also conflict with the
paper, for example `model_data/waterscenes.txt` contains 4 detection classes,
while the paper reports WaterScenes with 7 categories.

## High-risk reproduction issue fixed

The local code normalized each radar `.npz` with a global min-max transform by
default. REVP maps are sparse and contain negative velocity/elevation values.
With global min-max normalization, zero-valued empty background becomes non-zero
whenever a channel contains negative values. This makes the radar branch see a
dense background response instead of sparse radar points.

The official dataloader does not normalize radar maps. The local code now:

- disables radar normalization by default (`ASY_RADAR_NORMALIZE=0`);
- keeps the option configurable for ablations;
- if normalization is explicitly enabled, normalizes only non-zero values per
  channel so empty background remains zero;
- passes the same radar preprocessing option through training callbacks,
  `eval_paper_metrics.py`, and `yolo.py`.

## Baseline rerun target

The next baseline run should use:

- `phi=nano` to match the paper-scale parameter count;
- input shape `320x320`;
- global batch size `16`;
- SGD, momentum `0.937`, weight decay `5e-4`;
- cosine LR schedule;
- 4-subtask uncertainty loss;
- EMA and mixed precision;
- 5-frame radar root;
- radar alignment `letterbox`;
- radar normalization disabled.

