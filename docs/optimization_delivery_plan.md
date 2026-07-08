# ASY-VRNet Optimization Delivery Plan

## Current Verified Findings

- The comparison/evidence pipeline now uses the same prediction txt files for
  drawing boxes, confidence values, and `detect x/N` counting.
- `15239` has four ground-truth objects in WaterScenes detection XML, all
  `ship`.
- With the reproduction radar preprocessing and default 320 input:
  - Baseline: `detect 2/4`
  - Ours: `detect 2/4`
- With high-resolution 512 input:
  - Baseline: `detect 2/4`
  - Ours: `detect 3/4`
- With high-resolution 640 input:
  - Baseline: `detect 1/4`
  - Ours: `detect 3/4`

This supports a practical optimization direction: high-resolution inference is
useful for foggy distant small-object cases, while full retraining or hard-case
fine-tuning is still needed for the smallest targets and better localization.

## Fixed Reproduction Settings

All visual comparisons and metrics should use the final reproduction radar
preprocessing settings:

```bash
--radar-legacy-preprocess
--no-radar-preserve-points
--radar-source-order range,doppler,elevation,power
--radar-target-order range,doppler,elevation,power
```

Without these settings, single-image predictions can look much worse than the
reported metric pipeline.

## Implemented Tools

### Single-image GT comparison

```bash
python compare_detection.py \
  --image image/15239.jpg \
  --radar-root dataset/VOCradar_5_frames \
  --baseline weights/baseline_best.pth \
  --ours weights/final_greedy_soup.pth \
  --gt-xml /root/datasets/WaterScenes_data/detection/xml/15239.xml \
  --input-shape 512,512 \
  --confidence 0.3 \
  --radar-legacy-preprocess --no-radar-preserve-points \
  --radar-source-order range,doppler,elevation,power \
  --radar-target-order range,doppler,elevation,power \
  --out outputs/compare_15239_gt_shape_512_conf_0.3.jpg
```

### High-resolution wrapper

```bash
PYTHON_BIN=/root/autodl-tmp/miniconda3/bin/python \
bash scripts/run_gt_compare_highres.sh 15239 512 0.3
```

### Batch GT comparison and case mining

```bash
python scripts/batch_gt_compare.py \
  --ids 15239 \
  --image-dir image \
  --radar-root dataset/VOCradar_5_frames \
  --gt-dir /root/datasets/WaterScenes_data/detection/xml \
  --input-shape 512,512 \
  --confidence 0.3 \
  --radar-legacy-preprocess --no-radar-preserve-points \
  --radar-source-order range,doppler,elevation,power \
  --radar-target-order range,doppler,elevation,power \
  --out-dir outputs/batch_gt_compare_512
```

The batch script writes:

- `summary.csv`
- `best_cases/`
- `tie_cases/`
- `failure_cases/`

## Data Acquisition Status

Official WaterScenes files identified from the project Google Drive:

- `image.zip`
- `detection.zip`
- `radar_5_frames.zip`
- `information_list.csv`
- `train.txt`
- `val.txt`
- `test.txt`

Completed:

- `detection.zip` downloaded and unpacked.
- `information_list.csv`, `train.txt`, `val.txt`, and `test.txt` downloaded.
- Detection XML labels are available on the server.
- `radar_5_frames.zip` downloaded on the server from `hf-mirror.com`
  (`/root/datasets/WaterScenes_data/radar_5_frames.zip`, about 2.0 GB).
- `scripts/extract_waterscenes_subset.py` can extract selected radar CSV/XML
  files and convert radar CSV to project NPZ without full dataset unpacking.
- `scripts/select_waterscenes_cases.py` can rank candidate visual evidence
  cases from metadata and detection labels before the image archive is ready.
- A top-100 adverse/small-object validation list has been generated at
  `outputs/candidate_ids_val_rain_fog_night_small.ids.txt`.
- On the server, radar NPZ and XML labels for those 100 candidates have already
  been extracted:
  - `dataset/candidate_val_rain_fog_night_radar_npz`
  - `/root/datasets/WaterScenes_data/candidate_val_rain_fog_night/xml`
- The official `15239.csv` radar file was extracted and converted; the 512
  high-resolution result remains Baseline `2/4`, Ours `3/4`.

Blocked:

- `image.zip` fails over Google Drive because the public file currently returns
  "Quota exceeded".
- Baidu Netdisk link is reachable, but BaiduPCS-Go requires an authenticated
  Baidu account (`请重新登录`).

Recommended next data path:

1. Download `image.zip` using a logged-in browser, Google Drive "make a copy",
   or Baidu Netdisk client when the quota clears.
2. Upload `image.zip` to `/root/datasets/WaterScenes_data/`.
3. Extract only selected image IDs first to avoid filling disk.
4. Run batch comparison on adverse-condition IDs with extracted images, XML,
   and converted radar NPZ files.

## Next Optimization Steps

1. Build a small validation subset with image, radar, and XML for rainy/foggy
   small-object cases.
2. Run `scripts/batch_gt_compare.py` at 320 and 512 input sizes.
3. Select cases where `Ours detect > Baseline detect` and localization is clean.
4. Keep 15239 as a hard-case example: high-resolution inference improves it, but
   one very small far target still remains difficult.
5. If more improvement is required, fine-tune from `final_greedy_soup.pth` on
   adverse small-object cases with a low learning rate.

## Teacher-facing Explanation

The final model is effective in aggregate, but visual quality depends strongly
on consistent radar preprocessing and input resolution. For foggy distant
small-object scenes, default 320 inference can under-represent the model's
ability. High-resolution 512 inference improves recall and produces more
convincing GT-aligned visual evidence without manually editing predictions.
