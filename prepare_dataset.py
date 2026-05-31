"""
WaterScenes 样本数据集准备脚本

输入（WaterScenes 原始格式）:
  WaterScenes_Samples/
    image/          *.jpg  (1920×1080)
    detection/yolo/ *.txt  (YOLO 格式: class cx cy w h, normalized)
    semantic/SegmentationClass/ *.png (灰度分割掩码)
    radar/          *.csv

输出（代码所需格式）:
  VOCdevkit/VOC2007/
    JPEGImages/     *.jpg
    Annotations/    *.xml  (VOC XML 检测标注)
    SegmentationClass/ *.png
    ImageSets/Main/  train.txt, val.txt
  VOCradar/         *.npz  [4, 512, 512]
  2007_train.txt    (检测训练标注)
  2007_val.txt
"""

import os
import sys
import shutil
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ====================== 配置 ======================
PROJECT_ROOT = Path(os.environ.get("ASY_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
SAMPLE_DIR   = Path(os.environ.get("WATERSCENES_SAMPLE_DIR", PROJECT_ROOT / "dataset" / "WaterScenes_Samples"))
OUTPUT_DIR   = Path(os.environ.get("ASY_DATASET_DIR", PROJECT_ROOT / "dataset"))
VOC_DIR      = OUTPUT_DIR / "VOCdevkit" / "VOC2007"
RADAR_DIR    = OUTPUT_DIR / "VOCradar"

IMG_W, IMG_H = 1920, 1080   # 原图分辨率
FEAT_W, FEAT_H = 512, 512   # 雷达特征图目标分辨率
VAL_RATIO = 0.2             # 验证集比例

CLASSES = ['pier', 'buoy', 'sailor', 'ship', 'boat', 'vessel', 'kayak']
# ==================================================


def make_dirs():
    for d in [
        VOC_DIR / "JPEGImages",
        VOC_DIR / "Annotations",
        VOC_DIR / "SegmentationClass",
        VOC_DIR / "ImageSets" / "Main",
        RADAR_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def yolo_to_voc_xml(txt_path: Path, stem: str, img_w=IMG_W, img_h=IMG_H) -> str:
    """将 YOLO txt 标注转换为 VOC XML 字符串"""
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = stem + ".jpg"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text  = str(img_w)
    ET.SubElement(size, "height").text = str(img_h)
    ET.SubElement(size, "depth").text  = "3"

    lines = txt_path.read_text().strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id, cx, cy, bw, bh = int(parts[0]), *[float(x) for x in parts[1:]]
        xmin = int((cx - bw / 2) * img_w)
        ymin = int((cy - bh / 2) * img_h)
        xmax = int((cx + bw / 2) * img_w)
        ymax = int((cy + bh / 2) * img_h)
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(img_w, xmax), min(img_h, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = CLASSES[cls_id]
        ET.SubElement(obj, "difficult").text = "0"
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(xmin)
        ET.SubElement(bbox, "ymin").text = str(ymin)
        ET.SubElement(bbox, "xmax").text = str(xmax)
        ET.SubElement(bbox, "ymax").text = str(ymax)

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    return "\n".join(xml_str.split("\n")[1:])  # 去掉 xml 声明行


def csv_to_npz(csv_path: Path, out_path: Path):
    """雷达 CSV → npz 特征图 [4, FEAT_H, FEAT_W]"""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        np.savez_compressed(str(out_path), np.zeros((4, FEAT_H, FEAT_W), np.float32))
        return

    feature_map = np.zeros((4, FEAT_H, FEAT_W), dtype=np.float32)
    required = {'u', 'v', 'range', 'doppler', 'elevation', 'power'}
    if not required.issubset(df.columns) or df.empty:
        np.savez_compressed(str(out_path), feature_map)
        return

    scale_u = FEAT_W / IMG_W
    scale_v = FEAT_H / IMG_H
    df = df.copy()
    df['us'] = (df['u'] * scale_u).astype(int).clip(0, FEAT_W - 1)
    df['vs'] = (df['v'] * scale_v).astype(int).clip(0, FEAT_H - 1)

    # 同一像素多点时保留 power 最大的（升序写入，大值覆盖小值）
    df = df.sort_values('power', ascending=True)
    us = df['us'].values
    vs = df['vs'].values
    feature_map[0, vs, us] = df['range'].values.astype(np.float32)
    feature_map[1, vs, us] = df['doppler'].values.astype(np.float32)
    feature_map[2, vs, us] = df['elevation'].values.astype(np.float32)
    feature_map[3, vs, us] = df['power'].values.astype(np.float32)
    np.savez_compressed(str(out_path), feature_map)


def annotation_line(img_path: Path, xml_path: Path) -> str:
    """生成 2007_train.txt 格式的一行"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    try:
        line = img_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        line = str(img_path)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in CLASSES:
            continue
        cls_id = CLASSES.index(cls)
        b = obj.find('bndbox')
        xmin = b.find('xmin').text
        ymin = b.find('ymin').text
        xmax = b.find('xmax').text
        ymax = b.find('ymax').text
        line += f" {xmin},{ymin},{xmax},{ymax},{cls_id}"
    return line


def main():
    make_dirs()

    img_dir  = SAMPLE_DIR / "image"
    det_dir  = SAMPLE_DIR / "detection" / "yolo"
    seg_dir  = SAMPLE_DIR / "semantic" / "SegmentationClass"
    rad_dir  = SAMPLE_DIR / "radar"

    stems = sorted([p.stem for p in img_dir.glob("*.jpg")])
    print(f"共 {len(stems)} 个样本")

    # -------- 1. 复制图像 / 转换 XML / 复制分割掩码 / 转换雷达 --------
    print("处理图像、标注、分割掩码、雷达...")
    for stem in tqdm(stems):
        # 图像
        shutil.copy(img_dir / (stem + ".jpg"),
                    VOC_DIR / "JPEGImages" / (stem + ".jpg"))

        # 检测 → VOC XML
        txt = det_dir / (stem + ".txt")
        xml_str = yolo_to_voc_xml(txt, stem) if txt.exists() else yolo_to_voc_xml.__wrapped__ if False else "<annotation/>"
        if txt.exists():
            xml_str = yolo_to_voc_xml(txt, stem)
        else:
            xml_str = "<annotation/>"
        (VOC_DIR / "Annotations" / (stem + ".xml")).write_text(xml_str)

        # 分割掩码
        seg_src = seg_dir / (stem + ".png")
        if seg_src.exists():
            shutil.copy(seg_src, VOC_DIR / "SegmentationClass" / (stem + ".png"))

        # 雷达 CSV → NPZ
        rad_src = rad_dir / (stem + ".csv")
        if rad_src.exists():
            csv_to_npz(rad_src, RADAR_DIR / (stem + ".npz"))
        else:
            np.savez_compressed(str(RADAR_DIR / (stem + ".npz")),
                                np.zeros((4, FEAT_H, FEAT_W), np.float32))

    # -------- 2. 生成 ImageSets/Main/train.txt & val.txt --------
    random.seed(42)
    random.shuffle(stems)
    n_val = max(1, int(len(stems) * VAL_RATIO))
    val_stems   = set(stems[:n_val])
    train_stems = set(stems[n_val:])

    (VOC_DIR / "ImageSets" / "Main" / "train.txt").write_text("\n".join(sorted(train_stems)) + "\n")
    (VOC_DIR / "ImageSets" / "Main" / "val.txt").write_text("\n".join(sorted(val_stems)) + "\n")
    print(f"train: {len(train_stems)}, val: {len(val_stems)}")

    # -------- 3. 生成 2007_train.txt & 2007_val.txt --------
    print("生成 2007_train.txt / 2007_val.txt ...")
    for split, stem_set in [("train", train_stems), ("val", val_stems)]:
        lines = []
        for stem in sorted(stem_set):
            img_path = VOC_DIR / "JPEGImages" / (stem + ".jpg")
            xml_path = VOC_DIR / "Annotations" / (stem + ".xml")
            if xml_path.exists():
                lines.append(annotation_line(img_path, xml_path))
        (PROJECT_ROOT / f"2007_{split}.txt").write_text("\n".join(lines) + "\n")

    print("=== 完成！===")
    print(f"  图像:    {VOC_DIR}/JPEGImages/")
    print(f"  XML:     {VOC_DIR}/Annotations/")
    print(f"  分割:    {VOC_DIR}/SegmentationClass/")
    print(f"  雷达:    {RADAR_DIR}/")
    print(f"  标注:    {PROJECT_ROOT}/2007_train.txt  {PROJECT_ROOT}/2007_val.txt")


if __name__ == "__main__":
    main()
