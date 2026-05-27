import argparse
import csv
import json
import os
import shutil

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from utils.radar_utils import load_radar_npz, radar_to_tensor
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_map import get_coco_map
from utils_seg.utils_metrics import compute_mIoU
from yolo import YOLO


COCO_STAT_NAMES = [
    "mAP50-95",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ASY-VRNet with paper-style metrics.")
    parser.add_argument("--val_txt", default="2007_val.txt")
    parser.add_argument("--model_path", default="logs/best_epoch_weights.pth")
    parser.add_argument("--classes_path", default="model_data/waterscenes.txt")
    parser.add_argument("--radar_root", default="/mnt/f/ASY-VRNet/dataset/VOCradar")
    parser.add_argument("--vocdevkit_path", default="/mnt/f/ASY-VRNet/dataset/VOCdevkit")
    parser.add_argument("--out_dir", default="paper_metrics_out")
    parser.add_argument("--phi", default="l")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--input_shape", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--no_cuda", action="store_false", dest="cuda")
    return parser.parse_args()


def write_detection_gt(annotation_line, class_names, gt_dir):
    line = annotation_line.split()
    image_id = os.path.splitext(os.path.basename(line[0]))[0]
    with open(os.path.join(gt_dir, image_id + ".txt"), "w") as f:
        for box in line[1:]:
            left, top, right, bottom, cls_id = map(int, box.split(","))
            f.write(f"{class_names[cls_id]} {left} {top} {right} {bottom}\n")
    return image_id, line[0]


def save_segmentation_png(model, image, image_id, args, pred_dir):
    image = cvtColor(image)
    original_h, original_w = np.array(image).shape[:2]
    image_data = resize_image(image, (args.input_shape[1], args.input_shape[0]), True)
    image_data = np.expand_dims(
        np.transpose(preprocess_input(np.array(image_data, dtype=np.float32)), (2, 0, 1)),
        0,
    )

    radar_data = load_radar_npz(args.radar_root, image_id, image.size, args.input_shape)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    images = torch.from_numpy(image_data).to(device)
    radar = radar_to_tensor(radar_data, device=device)

    with torch.no_grad():
        seg = model.net(images, radar)[1][0]
        seg = F.softmax(seg.permute(1, 2, 0), dim=-1).cpu().numpy()

    iw, ih = image.size
    h, w = args.input_shape
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    seg = seg[dy:dy + nh, dx:dx + nw]
    seg = cv2.resize(seg, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
    seg = seg.argmax(axis=-1).astype(np.uint8)
    Image.fromarray(seg).save(os.path.join(pred_dir, image_id + ".png"))


def write_metrics(out_dir, coco_stats, seg_metrics):
    metrics = {name: float(value) for name, value in zip(COCO_STAT_NAMES, coco_stats)}
    metrics.update({k: float(v) for k, v in seg_metrics.items()})

    json_path = os.path.join(out_dir, "paper_metrics.json")
    csv_path = os.path.join(out_dir, "paper_metrics.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([key, value])
    print(f"Saved metrics to {json_path} and {csv_path}")


def main():
    args = parse_args()
    class_names, num_classes = get_classes(args.classes_path)
    with open(args.val_txt, encoding="utf-8") as f:
        val_lines = [line.strip() for line in f if line.strip()]

    if os.path.exists(args.out_dir):
        shutil.rmtree(args.out_dir)
    det_gt_dir = os.path.join(args.out_dir, "map_out", "ground-truth")
    det_dr_dir = os.path.join(args.out_dir, "map_out", "detection-results")
    seg_pred_dir = os.path.join(args.out_dir, "segmentation-results")
    os.makedirs(det_gt_dir, exist_ok=True)
    os.makedirs(det_dr_dir, exist_ok=True)
    os.makedirs(seg_pred_dir, exist_ok=True)

    model = YOLO(
        model_path=args.model_path,
        classes_path=args.classes_path,
        radar_root=args.radar_root,
        input_shape=args.input_shape,
        phi=args.phi,
        confidence=args.confidence,
        nms_iou=args.nms_iou,
        cuda=args.cuda,
    )

    image_ids = []
    for annotation_line in tqdm(val_lines, desc="Evaluating"):
        image_id, image_path = write_detection_gt(annotation_line, class_names, det_gt_dir)
        image_ids.append(image_id)
        image = Image.open(image_path)
        model.get_map_txt(image_id, image, class_names, os.path.join(args.out_dir, "map_out"))
        save_segmentation_png(model, image, image_id, args, seg_pred_dir)

    coco_stats = get_coco_map(class_names=class_names, path=os.path.join(args.out_dir, "map_out"))
    gt_dir = os.path.join(args.vocdevkit_path, "VOC2007", "SegmentationClass")
    _, ious, pa_recall, precision = compute_mIoU(gt_dir, seg_pred_dir, image_ids, 9, None)
    seg_metrics = {
        "mIoU": np.nanmean(ious) * 100,
        "mPA": np.nanmean(pa_recall) * 100,
        "mPrecision": np.nanmean(precision) * 100,
    }
    write_metrics(args.out_dir, coco_stats, seg_metrics)


if __name__ == "__main__":
    main()
