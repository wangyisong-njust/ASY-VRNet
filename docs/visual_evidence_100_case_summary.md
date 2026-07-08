# Visual Evidence Summary

## Data

- Source image archive: `E:\BaiduNetdiskDownload\images\image.zip`
- Extracted candidate images locally:
  `outputs/candidate_val_rain_fog_night_images`
- Uploaded candidate images on the server:
  `/root/ASY-VRNet/image_candidate_val_rain_fog_night`
- Candidate radar NPZ on the server:
  `/root/ASY-VRNet/dataset/candidate_val_rain_fog_night_radar_npz`
- Candidate XML labels on the server:
  `/root/datasets/WaterScenes_data/candidate_val_rain_fog_night/xml`

## Inference Settings

```bash
--input-shape 512,512
--confidence 0.3
--radar-legacy-preprocess
--no-radar-preserve-points
--radar-source-order range,doppler,elevation,power
--radar-target-order range,doppler,elevation,power
```

## 100-Case Result

- Candidate type: validation split, adverse lighting/weather and small/dense
  targets.
- Total valid cases: 100
- Ours better: 65
- Tied: 24
- Ours worse: 11

The `TP x/N` value means class-aware one-to-one matches with ground truth at
IoU >= 0.5. It is not the number of drawn prediction boxes. Prediction count is
reported separately as `pred N` in regenerated figures.

This supports the current model's visual usefulness on selected hard cases.
It also gives a transparent screening process rather than manually editing
predictions.

## Recommended 9 Figures

Local folder:

`presentation/comparison_baseline_vs_ours_selected_512`

For a clearer teacher-facing visualization, use the cropped recovered-object
figures in:

`presentation/clear_visual_evidence_512`

In those figures:

- orange dashed boxes = GT objects missed by Baseline
- green thick boxes = GT objects recovered by Ours
- blue boxes = matched true-positive predictions

Recommended ids:

- `01702`: Baseline TP 6/18, Ours TP 13/18
- `35503`: Baseline TP 4/20, Ours TP 10/20
- `49445`: Baseline TP 3/17, Ours TP 9/17
- `19781`: Baseline TP 1/16, Ours TP 7/16
- `34307`: Baseline TP 9/14, Ours TP 14/14
- `35101`: Baseline TP 9/16, Ours TP 14/16
- `32712`: Baseline TP 7/14, Ours TP 12/14
- `02698`: Baseline TP 11/18, Ours TP 15/18
- `02757`: Baseline TP 10/18, Ours TP 14/18

For paper or presentation use, prefer the clearer cropped evidence figures.
The strongest visual examples are `01702`, `35503`, `49445`, `19781`, `34307`,
and `35101`.
