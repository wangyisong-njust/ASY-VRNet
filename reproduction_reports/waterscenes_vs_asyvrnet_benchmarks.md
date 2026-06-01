# WaterScenes and ASY-VRNet Benchmark Notes

## Scope

The WaterScenes GitHub repository is a dataset/devkit repository. It provides
data loading, radar point projection and visualization utilities, but not the
full ASY-VRNet training pipeline, pretrained ASY-VRNet weights or the
paper-style multi-task evaluation script used in this repository.

## WaterScenes Paper Benchmarks

WaterScenes object detection benchmark uses camera (`C`) and radar-camera
(`C+R`) baselines. The table below reports mAP50-95 / mAP50.

| Model | Modality | mAP50-95 | mAP50 |
| --- | --- | ---: | ---: |
| Faster R-CNN | C | 47.8 | 81.1 |
| CenterNet | C | 54.7 | 82.9 |
| Deformable DETR | C | 56.5 | 84.0 |
| YOLOX-M | C | 57.8 | 85.1 |
| YOLOv8-M | C | 59.2 | 84.4 |
| YOLOX-M | C+R 1-frame | 59.5 | 86.1 |
| YOLOX-M | C+R 3-frames | 60.3 | 87.4 |
| YOLOv8-M | C+R 1-frame | 61.2 | 88.0 |
| YOLOv8-M | C+R 3-frames | 62.5 | 88.8 |

WaterScenes image semantic segmentation benchmark reports all-class mIoU.

| Model | mIoU | MPA | OA |
| --- | ---: | ---: | ---: |
| DeepLabv3+ | 82.6 | 89.9 | 95.2 |
| HRNet | 83.1 | 91.7 | 95.3 |
| SegNeXt | 85.3 | 92.8 | 95.4 |
| SegFormer | 85.7 | 93.1 | 95.4 |
| Mask2Former | 86.6 | 93.9 | 96.2 |

WaterScenes panoptic perception benchmark reports object detection plus
free-space and waterline segmentation.

| Model | Modality | Params M | Det mAP50 | Det mAP50-95 | Free-space mIoU | Waterline mIoU |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| YOLOP | C | 7.9 | 68.0 | 42.6 | 99.0 | 72.1 |
| HybridNets | C | 12.8 | 69.8 | 49.5 | 98.0 | 69.8 |
| Achelous-S0 | C+R | 1.6 | 81.1 | 51.0 | 99.3 | 65.0 |
| Achelous-S1 | C+R | 2.8 | 83.5 | 54.1 | 99.4 | 68.7 |
| Achelous-S2 | C+R | 5.3 | 85.5 | 56.0 | 99.6 | 72.2 |

## ASY-VRNet Paper Benchmarks

ASY-VRNet uses a different comparison table and reports object detection,
object semantic segmentation and drivable-area segmentation:

| Model | Modality | Params M | FLOPs G | mAP50-95 | AR50-95 |
| --- | --- | ---: | ---: | ---: | ---: |
| YOLOP | V | 7.90 | 18.60 | 37.9 | 43.5 |
| HybridNets | V | 12.83 | 15.60 | 39.1 | 44.2 |
| Achelous | V+R | 3.49 | 3.04 | 41.5 | 45.6 |
| ASY-VRNet | V+R | 4.12 | 3.26 | 42.8 | 46.3 |

| Model | Modality | Params M | FLOPs G | Object mIoU | Drivable mIoU |
| --- | --- | ---: | ---: | ---: | ---: |
| SegFormer-B0 | V | 3.71 | 5.29 | 73.5 | 99.4 |
| DeepLabV3+ | V | 5.81 | 20.60 | 71.6 | 99.2 |
| Achelous | V+R | 3.49 | 3.04 | 70.6 | 99.5 |
| ASY-VRNet | V+R | 4.12 | 3.26 | 74.7 | 99.6 |

The two papers therefore use overlapping datasets but different benchmark
groupings and metric subsets. Our reproduction scripts keep the ASY-VRNet
paper-style outputs separate from WaterScenes dataset benchmark numbers.
