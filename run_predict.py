"""
批量推理测试脚本：对验证集图片运行检测+分割推理，保存结果图片
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, '.')

import numpy as np
from PIL import Image
from tqdm import tqdm

from yolo import YOLO
from deeplab import DeeplabV3

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "predict_output"
VAL_TXT    = PROJECT_ROOT / "2007_val.txt"
RADAR_ROOT = os.environ.get("ASY_RADAR_ROOT", str(PROJECT_ROOT / "dataset" / "VOCradar"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取验证集图片列表
with open(VAL_TXT) as f:
    val_lines = [l.strip().split()[0] for l in f.readlines() if l.strip()]

print(f"验证集共 {len(val_lines)} 张图片")

# 加载检测模型
det_model = YOLO()

success = 0
for img_path in tqdm(val_lines, desc="Detection"):
    try:
        img_path = Path(img_path)
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / img_path
        image = Image.open(img_path)
        # 从路径提取文件名作为 image_id（需要匹配雷达文件）
        stem = img_path.stem
        r_image = det_model.detect_image(image, stem, crop=False, count=False)
        save_path = OUTPUT_DIR / f"det_{stem}.jpg"
        r_image.save(save_path)
        success += 1
    except Exception as e:
        print(f"[WARN] {img_path}: {e}")

print(f"检测完成：{success}/{len(val_lines)} 张已保存到 {OUTPUT_DIR}/")
print("推理测试通过！")
