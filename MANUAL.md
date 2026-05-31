# ASY-VRNet 使用说明书

> **ASY-VRNet**（Asymmetric Vision-Radar Network）是一个面向内河/近海水域感知的多任务深度学习模型，同时完成**目标检测**（船只、浮标、行人等 7 类）和**语义分割**（水面、障碍物等 9 类）两项任务，输入为 RGB 摄像头图像 + 4D 毫米波雷达点云。

---

## 目录

1. [环境准备](#1-环境准备)
2. [项目结构说明](#2-项目结构说明)
3. [从零开始：数据集准备](#3-从零开始数据集准备)
4. [训练](#4-训练)
5. [推理（预测）](#5-推理预测)
6. [结果在哪里 / 含义是什么](#6-结果在哪里--含义是什么)
7. [换数据集怎么做](#7-换数据集怎么做)
8. [常见问题](#8-常见问题)

---

## 1. 环境准备

### 1.1 硬件要求

| 项目 | 最低要求 |
|------|---------|
| GPU | NVIDIA GPU，显存 ≥ 8 GB |
| 磁盘 | 数据集约 15 GB，模型约 200 MB |
| 内存 | ≥ 16 GB |

### 1.2 安装依赖

```bash
# 进入项目目录
cd <project-root>

# 安装依赖（推荐使用 conda 虚拟环境）
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install timm==1.0.17 albumentations pandas pillow tqdm scipy opencv-python gdown
```

---

## 2. 项目结构说明

```
ASY-VRNet/
├── train.py                  # 训练入口
├── yolo.py                   # 推理接口（封装模型加载+前向推理）
├── run_predict.py            # 批量推理脚本
├── prepare_dataset_full.py   # 数据集转换脚本（WaterScenes → 本项目格式）
├── load_pretrained.py        # 加载 CoC 预训练权重
│
├── nets/
│   └── efficient_vrnet.py    # 模型主体（EfficientVRNet）
├── backbone/
│   └── fusion/vr_coc.py      # 双流融合骨干网络（视觉+雷达）
├── neck/
│   └── coc_fpn_dual.py       # 非对称 FPN 颈部
├── head/
│   └── decouplehead.py       # 解耦检测头（YOLOX 风格）
│
├── model_data/
│   ├── waterscenes.txt       # 检测类别名称（7类）
│   └── coc_small-.../        # CoC ImageNet 预训练权重
│
├── dataset/
│   ├── VOCdevkit/VOC2007/    # 转换后的训练数据
│   │   ├── JPEGImages/       # 图像
│   │   ├── Annotations/      # XML 检测标注
│   │   └── SegmentationClass/# 分割掩码（PNG）
│   └── VOCradar/             # 雷达特征图（NPZ 格式）
│
├── logs/                     # 训练输出（权重、loss 曲线）
└── predict_output/           # 推理结果图片
```

---

## 3. 从零开始：数据集准备

本项目使用 **WaterScenes** 数据集。

### 3.1 下载数据

**Step 1**：访问以下链接，手动下载 `image.zip`（10.6 GB）：
```
https://drive.google.com/drive/folders/1ts_Jl91FlhliurzIOxx6DP2qChCGLDBj
```

**Step 2**：用 gdown 下载其余文件（约 800 MB）：
```bash
cd <project-root>/dataset

# 检测标注（40 MB）
python3 -m gdown 1FUKEI43Ns5uJ-wZycMKgtC3p6BoGJPiY -O detection.zip

# 分割掩码（159 MB）
python3 -m gdown 12stUbabDDLi4C8EY_6MBRNBD4znvLIzh -O semantic.zip

# 雷达 CSV（545 MB）
python3 -m gdown 15c1Y4qnsTqygEbhHKkZS44u3ERgFu0Fe -O radar.zip
```

**Step 3**：把 `image.zip` 也移到 `dataset/` 目录下：
```bash
mv /path/to/image.zip <project-root>/dataset/image.zip
```

### 3.2 解压

```bash
cd <project-root>/dataset

unzip image.zip     -d WaterScenes_Full/
unzip detection.zip -d WaterScenes_Full/
unzip semantic.zip  -d WaterScenes_Full/
unzip radar.zip     -d WaterScenes_Full/
```

解压后目录结构如下：
```
dataset/WaterScenes_Full/
├── image/               # 54120 张 JPG（1920×1080）
├── detection/yolo/      # 每张图对应的 YOLO 格式检测标注
├── semantic/SegmentationClass/  # 分割掩码 PNG
└── radar/               # 每帧的雷达点云 CSV
```

### 3.3 转换为训练格式

```bash
cd <project-root>

python3 prepare_dataset_full.py
```

**该脚本做了什么：**
- 从 54120 个样本中筛选出四路数据（图像+检测+分割+雷达）都齐全的样本
- 随机选取 5000 个（可在脚本顶部修改 `MAX_SAMPLES`）
- 将 YOLO 格式标注转为 VOC XML 格式
- 将雷达 CSV 转为 `[4, 512, 512]` 的 NPZ 特征图
- 按 8:2 划分训练集/验证集
- 生成 `2007_train.txt`、`2007_val.txt`

**输出：**
```
dataset/VOCdevkit/VOC2007/   ← 图像+标注+分割掩码
dataset/VOCradar/            ← 雷达 NPZ 文件
2007_train.txt               ← 训练集路径+检测框标注
2007_val.txt                 ← 验证集路径+检测框标注
```

> 转换 5000 样本约需 15 分钟。

---

## 4. 训练

### 4.1 下载预训练权重（可选但推荐）

```bash
cd <project-root>/model_data

python3 -m gdown --folder \
  "https://drive.google.com/drive/folders/1KM5iBMeN8nYZEMuXk5Xge1ZjXE3qI1Kc" \
  -O coc_small-bs128-lr0.001-wd0.05-dp0.0-distillnone-224/
```

预训练权重来自 ContextCluster（CoC-small）在 ImageNet 上训练的结果，用于初始化视觉骨干网络，可显著加速收敛。

### 4.2 修改训练配置

打开 `train.py`，根据需要修改以下参数（约在第 150-200 行）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `UnFreeze_Epoch` | `100` | 总训练轮数，数据多时建议 100-300 |
| `Unfreeze_batch_size` | `4` | batch 大小，显存不足时调小 |
| `Init_lr` | `1e-2` | 初始学习率 |
| `phi` | `'l'` | 模型规模：`nano/tiny/s/m/l`，l 最大最准 |
| `radar_file_path` | 见文件 | 雷达 NPZ 目录路径 |
| `VOCdevkit_path` | 见文件 | VOCdevkit 根目录路径 |

### 4.3 开始训练

```bash
cd <project-root>

python3 train.py
```

训练过程中终端会显示：
```
Epoch 1/100:  50%|█████  | 500/1000 [05:12, detection loss=45.2, segmentation loss=0.82, f score=0.312, lr=0.008]
```

| 指标 | 含义 |
|------|------|
| `detection loss` | 检测任务训练损失（越低越好） |
| `segmentation loss` | 分割任务训练损失（越低越好） |
| `f score` | 分割 F1 分数（越高越好，满分 1.0） |
| `lr` | 当前学习率 |

### 4.4 训练输出

训练结果保存在 `logs/` 目录：

```
logs/
├── best_epoch_weights.pth        # 验证集损失最低的权重（推荐用这个）
├── last_epoch_weights.pth        # 最后一轮的权重
├── ep010-loss*-det_val_loss*-seg_val_loss*.pth  # 每 10 epoch 保存一次
└── loss_*/
    ├── epoch_loss.png            # 训练/验证损失曲线图
    └── epoch_map.png             # mAP 曲线图
```

---

## 5. 推理（预测）

### 5.1 批量推理验证集

```bash
cd <project-root>

python3 run_predict.py
```

**运行后输出：**
```
验证集共 1000 张图片
Detection: 100%|██████████| 1000/1000
检测完成：1000/1000 张已保存到 predict_output/
```

结果图保存在 `predict_output/` 目录，文件名格式为 `det_<图片ID>.jpg`。

### 5.2 单张图片推理

```bash
cd <project-root>

python3 predict.py
```

运行后会提示输入图片路径和图片 ID（用于查找对应雷达文件）。

### 5.3 修改推理配置

打开 `yolo.py`，找到 `_defaults` 字典（约第 25 行）：

```python
_defaults = {
    "model_path"  : 'logs/best_epoch_weights.pth',  # 权重路径
    "classes_path": 'model_data/waterscenes.txt',    # 类别文件
    "radar_root"  : 'dataset/VOCradar',  # 雷达NPZ目录
    "input_shape" : [512, 512],                      # 输入分辨率
    "phi"         : 'l',                             # 模型规模（需与训练一致）
    "confidence"  : 0.3,                             # 检测置信度阈值
    "nms_iou"     : 0.5,                             # NMS IoU 阈值
}
```

> **注意**：`phi` 必须与训练时使用的 `phi` 一致，否则权重加载会失败。

---

## 6. 结果在哪里 / 含义是什么

### 6.1 推理结果图片

位置：`predict_output/det_<ID>.jpg`

图片上会绘制：
- **彩色检测框**：每个检测到的目标，框上标注类别名和置信度分数
- 7 个检测类别：`pier`（码头）、`buoy`（浮标）、`sailor`（水手）、`ship`（船）、`boat`（小艇）、`vessel`（船舶）、`kayak`（皮划艇）

### 6.2 训练损失曲线

位置：`logs/loss_<时间戳>/epoch_loss.png`

- 蓝色线：训练集损失
- 橙色线：验证集损失
- 两条线都下降且趋于平行 = 正常收敛；验证集损失回升 = 过拟合

### 6.3 分割 mIoU

训练过程中每 5 个 epoch 打印一次：
```
===> mIoU: 72.88; mPA: 79.29; Accuracy: 98.85
```

| 指标 | 含义 |
|------|------|
| `mIoU` | 各类别 IoU 的平均值，越高越好（满分 100%） |
| `mPA` | 平均像素精度（mean Pixel Accuracy） |
| `Accuracy` | 全局像素分类准确率 |

9 个分割类别（WaterScenes）：`background`、`ship`、`buoy`、`sailor`、`pier`、`boat`、`vessel`、`kayak`、`water`

### 6.4 保存的权重文件

| 文件 | 说明 |
|------|------|
| `best_epoch_weights.pth` | 验证集综合损失最低的轮次，**推理时用这个** |
| `last_epoch_weights.pth` | 最后一轮，可用于继续训练 |
| `ep010-loss*.pth` | 每 10 轮定期存档，用于回溯 |

---

## 7. 换数据集怎么做

### 7.1 数据格式要求

本项目需要四路对齐数据：

| 数据 | 格式 | 说明 |
|------|------|------|
| RGB 图像 | JPG，任意分辨率 | 训练时统一缩放到 512×512 |
| 检测标注 | YOLO txt（`class cx cy w h`，归一化） 或 VOC XML | 每张图一个文件 |
| 分割掩码 | PNG，灰度图 | 像素值 = 类别 ID（0=背景） |
| 雷达点云 | NPZ，shape `[4, H, W]`（range/doppler/elevation/power） | 与图像同名 |

### 7.2 替换为自定义数据集

**Step 1：准备图像和标注**

将数据组织为以下结构：
```
my_dataset/
├── image/          # *.jpg
├── detection/yolo/ # *.txt（YOLO格式）
├── semantic/SegmentationClass/  # *.png（分割掩码）
└── radar/          # *.csv 或直接放 *.npz
```

**Step 2：如果雷达是 CSV 格式，转为 NPZ**

编辑 `radar_csv_to_npz.py` 顶部路径后运行：
```bash
python3 radar_csv_to_npz.py \
  --csv_dir my_dataset/radar/ \
  --out_dir dataset/VOCradar/
```

CSV 文件需包含列：`u`（像素列）、`v`（像素行）、`range`、`doppler`、`elevation`、`power`。

**Step 3：修改数据集转换脚本**

复制并修改 `prepare_dataset_full.py`：

```python
# 修改以下路径
FULL_DIR   = Path("/path/to/my_dataset")      # 你的数据集目录
MAX_SAMPLES = None                             # None = 使用全部样本

# 修改类别名称（与分割掩码的像素值对应）
CLASSES = ['class1', 'class2', ...]           # 检测类别
```

```bash
python3 prepare_dataset_full.py
```

**Step 4：修改类别文件**

编辑 `model_data/waterscenes.txt`，每行一个检测类别名（与 `CLASSES` 列表顺序一致）：
```
class1
class2
class3
```

**Step 5：修改 train.py 中的分割类别数**

```python
# 找到这一行，改为你的分割类别数（含背景）
num_classes_seg = 9   # ← 改为你的类别数
```

**Step 6：修改 yolo.py 中的分割类别数**

```python
self.net = EfficientVRNet(num_classes=self.num_classes, num_seg_classes=9, ...)
#                                                        ↑ 改为你的分割类别数
```

**Step 7：重新训练**

```bash
cd <project-root>
python3 train.py
```

### 7.3 只换检测类别（保留分割结构）

如果只是把水域场景换成其他场景（同样需要检测+分割），检测类别数不同时还需修改：

```python
# train.py 中
num_classes = 7   # ← 改为新类别数

# yolo.py _defaults 中
"classes_path": 'model_data/my_classes.txt'  # ← 指向新类别文件
```

---

## 8. 常见问题

### Q: 报错 `No module named 'timm.models.layers.helpers'`

timm 版本问题。运行：
```bash
pip install timm==1.0.17
```

### Q: 推理结果没有检测框

可能原因：
1. **模型未收敛**：检测任务需要较多数据（建议 ≥ 5000 张），训练轮数不足
2. **置信度阈值太高**：修改 `yolo.py` 中 `"confidence": 0.3` 改为 `0.15`
3. **phi 不一致**：`yolo.py` 的 `phi` 必须与训练时的 `phi` 相同

### Q: 显存不足（OOM）

修改 `train.py`：
```python
Unfreeze_batch_size = 2   # 从4改为2
```

### Q: 训练速度很慢

确认 GPU 在使用：
```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

如果返回 `False`，检查 CUDA 驱动和 torch 版本是否匹配。

### Q: 换了数据集后分割效果差

- 检查分割掩码的像素值是否从 0 开始连续编号
- 确认 `num_classes_seg` 与掩码中实际类别数一致
- 适当增加训练轮数（`UnFreeze_Epoch`）

---

## 快速参考命令

```bash
# 1. 进入项目目录
cd <project-root>

# 2. 转换数据集
python3 prepare_dataset_full.py

# 3. 训练
python3 train.py

# 4. 批量推理
python3 run_predict.py

# 5. 查看训练进度（实时）
tail -f /tmp/train_log.txt | grep -E "loss|mIoU|Epoch"

# 6. 查看结果图片
ls predict_output/
```
