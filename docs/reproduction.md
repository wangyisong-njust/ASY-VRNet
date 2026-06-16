# ASY-VRNet 复现与实验命令

更新日期：2026-06-12

本文只保留复现实验需要的配置、路径和命令。方法解释、实验结论和论文叙事见 `docs/report.md`。

## 仓库不包含的大文件（需自行准备）

为保持仓库精简，以下大文件不随代码提供，请按说明准备：

| 内容 | 获取方式 | 放置位置 |
| --- | --- | --- |
| WaterScenes 数据集（图像 + 5 帧雷达） | 公开数据集，按官方说明下载 | `dataset/VOCdevkit/`、`dataset/VOCradar_5_frames/` |
| ContextCluster 预训练骨干 `model_best.pth.tar` | 公开预训练权重，按 ContextCluster 官方发布下载 | `model_data/coc_small-bs128-lr0.001-wd0.05-dp0.0-distillnone-224/` |
| 最终单次前向模型 `greedy_soup_ms_full.pth`（194MB） | 不随仓库提供（超 GitHub 单文件上限）；用 `scripts/greedy_soup.py` 从创新点一与多尺度微调权重再生成（命令见“§3 生成单次前向最优模型汤”），或通过另行提供的链接获取 | `logs_innov2_soup/greedy_soup_ms_full.pth` |
| 训练中间检查点（创新点一 / 多尺度微调的 epoch 权重） | 不随仓库提供，按下文命令自行训练 | 各 `logs_*/` 目录 |

> 注：最终模型与中间检查点均不随仓库提供，按上表自行训练 / 再生成。仓库只保留最终代码、文档与紧凑的指标摘要（`paper_metrics_*/paper_metrics.json|csv`）。

## 固定评估口径

所有结果都必须使用同一套训练形态和评估口径，否则不可横向比较。

| 项目 | 取值 |
| --- | --- |
| 模型规模 | `phi=l` |
| 输入分辨率 | `320 x 320` |
| 雷达数据 | `dataset/VOCradar_5_frames` |
| 雷达通道 | `range,doppler,elevation,power` |
| 雷达预处理 | `legacy_preprocess=1`，不保留原始点 |
| fusion | baseline 结果用 `baseline`，可靠性门控实验用 `reliability` |
| confidence | `0.001` |
| NMS IoU | `0.5` |
| max boxes | `100` |
| dark 子集 | `time in {night}` |
| dim 子集 | `lighting in {dim}` 且 `time in {daytime, night}` 且 `weather in {overcast, rainy}` |
| small 子集 | 原图 GT 框面积 `<= 4096` |

## 关键产物

| 类型 | 路径 |
| --- | --- |
| 论文基线 best 权重 | `logs_legacy_highscore_phi_l_5frames_bs64_300e_320/best_epoch_weights.pth` |
| 论文基线评估结果 | `paper_metrics_legacy_highscore_best/paper_metrics.json` |
| 创新点一 best 权重 | `logs_innovation2_qfl_radar_phi_l_5frames_bs64_300e_320/best_epoch_weights.pth` |
| 创新点一 best 结果 | `paper_metrics_innovation2_qfl_radar_best/paper_metrics.json` |
| 当前最终权重 | `logs_innov2_soup/soup_innov2_best_multiscale_best.pth` |
| 单次前向最优结果 | `paper_metrics_innov2_soup_best_multiscale_best/paper_metrics.json` |
| 高精度 TTA 最优结果 | `paper_metrics_innov2_soup_tta_softnms_320_384/paper_metrics.json` |

## 1. 复现论文基线

重新训练基线：

```bash
bash scripts/run_train_legacy_highscore_4gpu.sh
```

重新评估已有基线 best/last：

```bash
bash scripts/after_train_eval_legacy_highscore.sh
```

只评估基线 best：

```bash
python3 eval_paper_metrics.py \
  --model_path logs_legacy_highscore_phi_l_5frames_bs64_300e_320/best_epoch_weights.pth \
  --fusion_mode baseline \
  --out_dir paper_metrics_legacy_highscore_best \
  --phi l --input_shape 320 320 --confidence 0.001 --max_boxes 100 \
  --radar_root dataset/VOCradar_5_frames --radar_legacy_preprocess --no_radar_preserve_points \
  --radar_source_order range,doppler,elevation,power --radar_target_order range,doppler,elevation,power \
  --dark_times night \
  --dim_lightings dim --dim_times daytime,night --dim_weathers overcast,rainy \
  --small_area 4096 --small_area_space original
```

基线 best 已复现结果：

| 指标 | 结果 |
| --- | ---: |
| mAP50-95 | 42.570 |
| AP50 | 73.080 |
| AP75 | 43.951 |
| AR50-95 | 47.162 |
| mIoU_o | 77.618 |
| mIoU_d | 98.764 |
| mAP_da | 42.892 |
| mAP_di | 40.163 |
| mAP_sm | 36.992 |

## 2. 训练创新点一

创新点一是 Radar-Prior Quality-Aligned Head：QFL + radar prior，baseline 融合，不改推理结构。

仅 QFL 消融：

```bash
HEAD_VARIANT=qfl bash scripts/run_train_innovation2_head_4gpu.sh
```

完整创新点一：

```bash
HEAD_VARIANT=qfl_radar bash scripts/run_train_innovation2_head_4gpu.sh
```

如果 GPU 不足，可改 DDP 卡数：

```bash
NPROC=3 CUDA_VISIBLE_DEVICES=1,2,3 HEAD_VARIANT=qfl_radar bash scripts/run_train_innovation2_head_4gpu.sh
```

评估创新点一 best/last：

```bash
EXP_NAME=innovation2_qfl_radar_phi_l_5frames_bs64_300e_320 \
FUSION_MODE=baseline \
BEST_OUT=paper_metrics_innovation2_qfl_radar_best \
LAST_OUT=paper_metrics_innovation2_qfl_radar_last \
bash scripts/after_train_eval_and_diagnose.sh
```

创新点一 best 已验证结果：

| 指标 | 结果 |
| --- | ---: |
| mAP50-95 | 49.958 |
| AP50 | 78.563 |
| AP75 | 52.962 |
| AR50-95 | 54.940 |
| mIoU_o | 79.225 |
| mIoU_d | 98.873 |
| mAP_da | 46.886 |
| mAP_di | 45.842 |
| mAP_sm | 44.736 |

## 3. 生成单次前向最优模型汤

当前最优方法是平均创新点一 best 与多尺度微调 best：

```bash
python3 scripts/make_checkpoint_soup.py \
  --out logs_innov2_soup/soup_innov2_best_multiscale_best.pth \
  logs_innovation2_qfl_radar_phi_l_5frames_bs64_300e_320/best_epoch_weights.pth \
  logs_innovation3_multiscale_ft_phi_l_5frames_bs48_e50_320/best_epoch_weights.pth
```

评估模型汤：

```bash
python3 eval_paper_metrics.py \
  --model_path logs_innov2_soup/soup_innov2_best_multiscale_best.pth \
  --fusion_mode baseline \
  --out_dir paper_metrics_innov2_soup_best_multiscale_best \
  --phi l --input_shape 320 320 --confidence 0.001 --max_boxes 100 \
  --radar_root dataset/VOCradar_5_frames --radar_legacy_preprocess --no_radar_preserve_points \
  --radar_source_order range,doppler,elevation,power --radar_target_order range,doppler,elevation,power \
  --dark_times night \
  --dim_lightings dim --dim_times daytime,night --dim_weathers overcast,rainy \
  --small_area 4096 --small_area_space original
```

单次前向结果：

| 指标 | 结果 |
| --- | ---: |
| mAP50-95 | 50.114 |
| AP50 | 78.819 |
| AP75 | 53.291 |
| AR50-95 | 54.827 |
| mIoU_o | 79.925 |
| mIoU_d | 98.886 |
| mAP_da | 46.785 |
| mAP_di | 46.091 |
| mAP_sm | 44.813 |

## 4. 高精度推理：Radar-Aware TTA

该步骤不需要重新训练，直接使用单次前向最优 soup 权重。它会使用 `320,384` 双尺度推理，并通过 radar-aware Soft-NMS 融合候选框。

```bash
python3 eval_paper_metrics.py \
  --model_path logs_innov2_soup/soup_innov2_best_multiscale_best.pth \
  --fusion_mode baseline \
  --out_dir paper_metrics_innov2_soup_tta_softnms_320_384 \
  --phi l --input_shape 320 320 --confidence 0.001 --max_boxes 100 \
  --radar_root dataset/VOCradar_5_frames --radar_legacy_preprocess --no_radar_preserve_points \
  --radar_source_order range,doppler,elevation,power --radar_target_order range,doppler,elevation,power \
  --dark_times night \
  --dim_lightings dim --dim_times daytime,night --dim_weathers overcast,rainy \
  --small_area 4096 --small_area_space original \
  --tta --tta_scales 320,384 --no_tta_flip \
  --tta_fusion softnms --tta_radar_alpha 0.5
```

高精度 TTA 结果：

| 指标 | 结果 |
| --- | ---: |
| mAP50-95 | 51.186 |
| AP50 | 79.494 |
| AP75 | 54.908 |
| AR50-95 | 59.491 |
| mIoU_o | 79.925 |
| mIoU_d | 98.886 |
| mAP_da | 47.465 |
| mAP_di | 48.622 |
| mAP_sm | 46.073 |

快速消融使用 `2007_val_subset400.txt`，只评估、不训练：

```bash
python3 eval_paper_metrics.py \
  --val_txt 2007_val_subset400.txt \
  --model_path logs_innov2_soup/soup_innov2_best_multiscale_best.pth \
  --fusion_mode baseline \
  --out_dir paper_metrics_quick400_ablate_tta_nms_alpha0 \
  --phi l --input_shape 320 320 --confidence 0.001 --max_boxes 100 \
  --radar_root dataset/VOCradar_5_frames --radar_legacy_preprocess --no_radar_preserve_points \
  --radar_source_order range,doppler,elevation,power --radar_target_order range,doppler,elevation,power \
  --dark_times night \
  --dim_lightings dim --dim_times daytime,night --dim_weathers overcast,rainy \
  --small_area 4096 --small_area_space original \
  --tta --tta_scales 320,384 --no_tta_flip \
  --tta_fusion nms --tta_radar_alpha 0.0
```

```bash
python3 eval_paper_metrics.py \
  --val_txt 2007_val_subset400.txt \
  --model_path logs_innov2_soup/soup_innov2_best_multiscale_best.pth \
  --fusion_mode baseline \
  --out_dir paper_metrics_quick400_ablate_tta_softnms_alpha0 \
  --phi l --input_shape 320 320 --confidence 0.001 --max_boxes 100 \
  --radar_root dataset/VOCradar_5_frames --radar_legacy_preprocess --no_radar_preserve_points \
  --radar_source_order range,doppler,elevation,power --radar_target_order range,doppler,elevation,power \
  --dark_times night \
  --dim_lightings dim --dim_times daytime,night --dim_weathers overcast,rainy \
  --small_area 4096 --small_area_space original \
  --tta --tta_scales 320,384 --no_tta_flip \
  --tta_fusion softnms --tta_radar_alpha 0.0
```

## 5. 其他实验命令

可靠性门控微调：

```bash
FT_MODE=reliability_fixed bash scripts/run_finetune_from_baseline_4gpu.sh
```

历史一致性正则实验：

```bash
bash scripts/run_train_innovation3_consistency_4gpu.sh
```

雷达退化鲁棒性曲线：

```bash
python3 scripts/plot_radar_robustness_curve.py \
  --models \
    "baseline=logs_legacy_highscore_phi_l_5frames_bs64_300e_320/best_epoch_weights.pth=baseline" \
    "innov3=logs_innovation3_consistency_phi_l_5frames_bs64_300e_320/best_epoch_weights.pth=baseline" \
  --ratios 0 0.25 0.5 0.75 1.0 \
  --out_root paper_metrics_robustness
```

## 6. 判定标准

| 指标 | 判定 |
| --- | --- |
| 主指标 | `mAP50-95` 高于基线 `42.570`，且相对上一个主线方法继续提升 |
| 子集指标 | `mAP_da`、`mAP_di`、`mAP_sm` 至少一个明显提升，其他不应大幅退化 |
| 定位质量 | `AP75` 提升优先级高，说明不是只提高低阈值召回 |
| 分割副指标 | `mIoU_o`、`mIoU_d` 不应明显下降 |
