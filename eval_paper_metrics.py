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
    parser.add_argument("--radar_root", default=os.environ.get("ASY_RADAR_ROOT", "dataset/VOCradar"))
    parser.add_argument("--vocdevkit_path", default=os.environ.get("ASY_VOCDEVKIT", "dataset/VOCdevkit"))
    parser.add_argument(
        "--info_csv",
        default=os.environ.get("ASY_INFO_CSV", "dataset/WaterScenes_Full/information_list.csv"),
        help="WaterScenes information_list.csv for adverse-condition subsets.",
    )
    parser.add_argument("--out_dir", default="paper_metrics_out")
    parser.add_argument("--phi", default="l")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--nms_iou", type=float, default=0.5)
    parser.add_argument("--input_shape", type=int, nargs=2, default=[320, 320])
    parser.add_argument("--num_seg_classes", type=int, default=9)
    parser.add_argument("--small_area", type=float, default=32 * 32)
    parser.add_argument("--disable_subset_metrics", action="store_true")
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


def summarize_segmentation(ious, pa_recall, precision):
    object_classes = np.arange(1, 8)
    drivable_class = 8
    metrics = {
        "mIoU_all": np.nanmean(ious) * 100,
        "mPA_all": np.nanmean(pa_recall) * 100,
        "mPrecision_all": np.nanmean(precision) * 100,
        "mIoU_o": np.nanmean(ious[object_classes]) * 100,
        "mIoU_d": ious[drivable_class] * 100,
    }
    for idx, value in enumerate(ious):
        metrics[f"IoU_class_{idx}"] = value * 100
    return metrics


def coco_stats_to_metrics(coco_stats, prefix=""):
    metrics = {prefix + name: float(value * 100) for name, value in zip(COCO_STAT_NAMES, coco_stats)}
    metrics[prefix + "AR50-95"] = metrics[prefix + "AR100"]
    return metrics


def has_small_gt_object(annotation_line, input_shape, small_area):
    line = annotation_line.split()
    image_path = line[0]
    if len(line) <= 1:
        return False

    with Image.open(image_path) as image:
        image_w, image_h = image.size

    input_h, input_w = input_shape
    scale = min(input_w / image_w, input_h / image_h)
    for box in line[1:]:
        left, top, right, bottom, _ = map(int, box.split(","))
        scaled_area = max(right - left, 0) * max(bottom - top, 0) * scale * scale
        if scaled_area <= small_area:
            return True
    return False


def load_adverse_subsets(info_csv, image_ids, annotation_by_id, input_shape, small_area):
    image_id_set = set(image_ids)
    subsets = {"dark": set(), "dim": set(), "small": set()}

    if os.path.exists(info_csv):
        with open(info_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                image_id = row.get("id", "")
                if image_id not in image_id_set:
                    continue
                if row.get("time") == "night":
                    subsets["dark"].add(image_id)
                if row.get("lighting") == "dim":
                    subsets["dim"].add(image_id)
    else:
        print(f"Warning: info csv not found, skip dark/dim subset metrics: {info_csv}")

    for image_id, annotation_line in annotation_by_id.items():
        if has_small_gt_object(annotation_line, input_shape, small_area):
            subsets["small"].add(image_id)

    return {name: [image_id for image_id in image_ids if image_id in ids] for name, ids in subsets.items()}


def copy_detection_subset(map_out_dir, subset_dir, image_ids):
    src_gt_dir = os.path.join(map_out_dir, "ground-truth")
    src_dr_dir = os.path.join(map_out_dir, "detection-results")
    dst_gt_dir = os.path.join(subset_dir, "ground-truth")
    dst_dr_dir = os.path.join(subset_dir, "detection-results")
    os.makedirs(dst_gt_dir, exist_ok=True)
    os.makedirs(dst_dr_dir, exist_ok=True)

    for image_id in image_ids:
        gt_src = os.path.join(src_gt_dir, image_id + ".txt")
        dr_src = os.path.join(src_dr_dir, image_id + ".txt")
        shutil.copy2(gt_src, os.path.join(dst_gt_dir, image_id + ".txt"))
        if os.path.exists(dr_src):
            shutil.copy2(dr_src, os.path.join(dst_dr_dir, image_id + ".txt"))
        else:
            open(os.path.join(dst_dr_dir, image_id + ".txt"), "w").close()


def compute_subset_metrics(class_names, gt_dir, seg_pred_dir, map_out_dir, out_dir, image_ids, args):
    metrics = {}
    for subset_name, subset_ids in image_ids.items():
        metrics[f"{subset_name}_count"] = len(subset_ids)
        if not subset_ids:
            print(f"Skip {subset_name} subset metrics: empty subset.")
            continue

        subset_map_dir = os.path.join(out_dir, "subsets", subset_name, "map_out")
        copy_detection_subset(map_out_dir, subset_map_dir, subset_ids)
        subset_coco_stats = get_coco_map(class_names=class_names, path=subset_map_dir)
        subset_det_metrics = coco_stats_to_metrics(subset_coco_stats, prefix=f"{subset_name}_")
        metrics.update(subset_det_metrics)

        _, ious, pa_recall, precision = compute_mIoU(
            gt_dir,
            seg_pred_dir,
            subset_ids,
            args.num_seg_classes,
            None,
        )
        subset_seg_metrics = summarize_segmentation(ious, pa_recall, precision)
        metrics.update({f"{subset_name}_{key}": float(value) for key, value in subset_seg_metrics.items()})

    metrics["mAP_da"] = metrics.get("dark_mAP50-95")
    metrics["mIoUda_d"] = metrics.get("dark_mIoU_d")
    metrics["mAP_di"] = metrics.get("dim_mAP50-95")
    metrics["mIoUdi_d"] = metrics.get("dim_mIoU_d")
    metrics["mAP_sm"] = metrics.get("small_mAP50-95")
    metrics["mIoUsm_o"] = metrics.get("small_mIoU_o")
    return metrics


def write_metrics(out_dir, coco_stats, seg_metrics, subset_metrics=None):
    metrics = coco_stats_to_metrics(coco_stats)
    metrics["AR50-95"] = metrics["AR100"]
    metrics["paper_mAP50-95"] = metrics["mAP50-95"]
    metrics["paper_AR50-95"] = metrics["AR50-95"]
    metrics.update({k: float(v) for k, v in seg_metrics.items()})
    if subset_metrics:
        metrics.update(subset_metrics)

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
    annotation_by_id = {}
    for annotation_line in tqdm(val_lines, desc="Evaluating"):
        image_id, image_path = write_detection_gt(annotation_line, class_names, det_gt_dir)
        image_ids.append(image_id)
        annotation_by_id[image_id] = annotation_line
        image = Image.open(image_path)
        model.get_map_txt(image_id, image, class_names, os.path.join(args.out_dir, "map_out"))
        save_segmentation_png(model, image, image_id, args, seg_pred_dir)

    map_out_dir = os.path.join(args.out_dir, "map_out")
    coco_stats = get_coco_map(class_names=class_names, path=map_out_dir)
    gt_dir = os.path.join(args.vocdevkit_path, "VOC2007", "SegmentationClass")
    _, ious, pa_recall, precision = compute_mIoU(gt_dir, seg_pred_dir, image_ids, args.num_seg_classes, None)
    seg_metrics = summarize_segmentation(ious, pa_recall, precision)

    subset_metrics = {}
    if not args.disable_subset_metrics:
        subsets = load_adverse_subsets(
            args.info_csv,
            image_ids,
            annotation_by_id,
            args.input_shape,
            args.small_area,
        )
        subset_metrics = compute_subset_metrics(
            class_names,
            gt_dir,
            seg_pred_dir,
            map_out_dir,
            args.out_dir,
            subsets,
            args,
        )

    write_metrics(args.out_dir, coco_stats, seg_metrics, subset_metrics)


if __name__ == "__main__":
    main()
