"""
WaterScenes 完整数据集准备脚本

输入（WaterScenes 完整数据集解压后）:
  WaterScenes_Full/
    image/          *.jpg  (1920×1080)
    detection/yolo/ *.txt  (YOLO: class cx cy w h, normalized)
    semantic/SegmentationClass/ *.png
    radar/          *.csv

输出:
  VOCdevkit/VOC2007/
    JPEGImages/     *.jpg
    Annotations/    *.xml
    SegmentationClass/ *.png
    ImageSets/Main/  train.txt, val.txt
  VOCradar/         *.npz  [4, 512, 512]
  2007_train.txt
  2007_val.txt
"""

import os
import shutil
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ====================== 配置 ======================
PROJECT_ROOT = Path(os.environ.get("ASY_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
FULL_DIR     = Path(os.environ.get("WATERSCENES_FULL_DIR", PROJECT_ROOT / "dataset" / "WaterScenes_Full"))
OUTPUT_DIR   = Path(os.environ.get("ASY_DATASET_DIR", PROJECT_ROOT / "dataset"))
VOC_DIR      = OUTPUT_DIR / "VOCdevkit" / "VOC2007"
RADAR_DIR    = OUTPUT_DIR / "VOCradar"

IMG_W, IMG_H   = 1920, 1080
FEAT_W, FEAT_H = 512, 512
MAX_SAMPLES    = int(os.environ.get("ASY_MAX_SAMPLES", "5000"))   # 全部用可传 0
RANDOM_SEED    = 42

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


def read_split(path: Path):
    if not path.exists():
        return None
    return [Path(line.strip().split()[0]).stem for line in path.read_text().splitlines() if line.strip()]


def yolo_to_voc_xml(txt_path: Path, stem: str) -> str:
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = stem + ".jpg"
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text  = str(IMG_W)
    ET.SubElement(size, "height").text = str(IMG_H)
    ET.SubElement(size, "depth").text  = "3"

    if txt_path.exists():
        for line in txt_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = [float(x) for x in parts[1:]]
            xmin = max(0,     int((cx - bw / 2) * IMG_W))
            ymin = max(0,     int((cy - bh / 2) * IMG_H))
            xmax = min(IMG_W, int((cx + bw / 2) * IMG_W))
            ymax = min(IMG_H, int((cy + bh / 2) * IMG_H))
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
    return "\n".join(xml_str.split("\n")[1:])


def csv_to_npz(csv_path: Path, out_path: Path):
    feature_map = np.zeros((4, FEAT_H, FEAT_W), dtype=np.float32)
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        np.savez_compressed(str(out_path), feature_map)
        return

    required = {'u', 'v', 'range', 'doppler', 'elevation', 'power'}
    if not required.issubset(df.columns) or df.empty:
        np.savez_compressed(str(out_path), feature_map)
        return

    df = df.copy()
    df['us'] = (df['u'] * (FEAT_W / IMG_W)).astype(int).clip(0, FEAT_W - 1)
    df['vs'] = (df['v'] * (FEAT_H / IMG_H)).astype(int).clip(0, FEAT_H - 1)
    df = df.sort_values('power', ascending=True)
    us, vs = df['us'].values, df['vs'].values
    feature_map[0, vs, us] = df['range'].values.astype(np.float32)
    feature_map[1, vs, us] = df['doppler'].values.astype(np.float32)
    feature_map[2, vs, us] = df['elevation'].values.astype(np.float32)
    feature_map[3, vs, us] = df['power'].values.astype(np.float32)
    np.savez_compressed(str(out_path), feature_map)


def annotation_line(img_path: Path, xml_path: Path) -> str:
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
        line += f" {b.find('xmin').text},{b.find('ymin').text},{b.find('xmax').text},{b.find('ymax').text},{cls_id}"
    return line


def main():
    make_dirs()

    img_dir = FULL_DIR / "image"
    det_dir = FULL_DIR / "detection" / "yolo"
    seg_dir = FULL_DIR / "semantic" / "SegmentationClass"
    rad_dir = FULL_DIR / "radar"

    # 以 image 目录为基准，找出四路数据都齐全的样本
    all_stems = sorted([p.stem for p in img_dir.glob("*.jpg")])
    print(f"image 目录共 {len(all_stems)} 张图")

    # 过滤：要求 detection + semantic + radar 都存在
    valid = []
    for stem in all_stems:
        if ((det_dir / (stem + ".txt")).exists() and
            (seg_dir / (stem + ".png")).exists() and
            (rad_dir / (stem + ".csv")).exists()):
            valid.append(stem)

    print(f"四路数据齐全: {len(valid)} 个样本")

    valid_set = set(valid)
    official_train = read_split(FULL_DIR / "train.txt")
    official_val = read_split(FULL_DIR / "val.txt")
    if official_train and official_val:
        train_stems = [stem for stem in official_train if stem in valid_set]
        val_stems = [stem for stem in official_val if stem in valid_set]
        valid = sorted(set(train_stems + val_stems))
        print(f"使用官方划分: train={len(train_stems)}, val={len(val_stems)}")
    else:
        if MAX_SAMPLES > 0 and len(valid) > MAX_SAMPLES:
            random.seed(RANDOM_SEED)
            random.shuffle(valid)
            valid = sorted(valid[:MAX_SAMPLES])
            print(f"随机选取 {len(valid)} 个样本")
        random.seed(RANDOM_SEED)
        shuffled = valid[:]
        random.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * 0.2))
        val_stems = sorted(shuffled[:n_val])
        train_stems = sorted(shuffled[n_val:])

    # -------- 处理每个样本 --------
    print("处理图像、标注、分割、雷达...")
    for stem in tqdm(valid):
        # 图像
        shutil.copy(img_dir / (stem + ".jpg"),
                    VOC_DIR / "JPEGImages" / (stem + ".jpg"))

        # 检测 XML
        xml_str = yolo_to_voc_xml(det_dir / (stem + ".txt"), stem)
        (VOC_DIR / "Annotations" / (stem + ".xml")).write_text(xml_str)

        # 分割掩码
        shutil.copy(seg_dir / (stem + ".png"),
                    VOC_DIR / "SegmentationClass" / (stem + ".png"))

        # 雷达 CSV → NPZ
        csv_to_npz(rad_dir / (stem + ".csv"), RADAR_DIR / (stem + ".npz"))

    if MAX_SAMPLES > 0 and len(valid) > MAX_SAMPLES:
        random.seed(RANDOM_SEED)
        sampled = set(random.sample(valid, MAX_SAMPLES))
        train_stems = [stem for stem in train_stems if stem in sampled]
        val_stems = [stem for stem in val_stems if stem in sampled]
        valid = sorted(sampled)
        print(f"按官方划分抽样 {len(valid)} 个样本: train={len(train_stems)}, val={len(val_stems)}")

    (VOC_DIR / "ImageSets" / "Main" / "train.txt").write_text("\n".join(train_stems) + "\n")
    (VOC_DIR / "ImageSets" / "Main" / "val.txt").write_text(  "\n".join(val_stems)   + "\n")
    print(f"train: {len(train_stems)}, val: {len(val_stems)}")

    # -------- 生成 2007_train.txt / 2007_val.txt --------
    print("生成标注文件...")
    for split, stem_list in [("train", train_stems), ("val", val_stems)]:
        lines = []
        for stem in stem_list:
            img_path = VOC_DIR / "JPEGImages" / (stem + ".jpg")
            xml_path = VOC_DIR / "Annotations" / (stem + ".xml")
            lines.append(annotation_line(img_path, xml_path))
        (PROJECT_ROOT / f"2007_{split}.txt").write_text("\n".join(lines) + "\n")

    print("=== 完成！===")
    print(f"  总样本: {len(valid)}, train: {len(train_stems)}, val: {len(val_stems)}")
    print(f"  图像: {VOC_DIR}/JPEGImages/")
    print(f"  雷达: {RADAR_DIR}/")


if __name__ == "__main__":
    main()
