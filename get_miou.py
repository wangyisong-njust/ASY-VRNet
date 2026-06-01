import argparse
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from eval_paper_metrics import env_bool, resolve_path, save_segmentation_png
from utils_seg.utils_metrics import compute_mIoU, show_results
from yolo import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
SEG_CLASSES = ["background", "ship", "buoy", "sailor", "pier", "boat", "vessel", "kayak", "water"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ASY-VRNet fused segmentation mIoU on a WaterScenes split."
    )
    parser.add_argument("--val_txt", default="2007_val.txt")
    parser.add_argument("--model_path", default=os.environ.get("ASY_MODEL_PATH", "logs/best_epoch_weights.pth"))
    parser.add_argument("--classes_path", default=os.environ.get("ASY_CLASSES_PATH", "model_data/waterscenes.txt"))
    parser.add_argument("--radar_root", default=os.environ.get("ASY_RADAR_ROOT", "dataset/VOCradar_5_frames"))
    parser.add_argument("--vocdevkit_path", default=os.environ.get("ASY_VOCDEVKIT", "dataset/VOCdevkit"))
    parser.add_argument("--out_dir", default="miou_out")
    parser.add_argument("--phi", default=os.environ.get("ASY_PHI", "l"))
    parser.add_argument("--input_shape", type=int, nargs=2, default=[320, 320])
    parser.add_argument("--num_seg_classes", type=int, default=9)
    parser.add_argument("--radar_channels", type=int, default=int(os.environ.get("ASY_RADAR_CHANNELS", "4")))
    parser.add_argument("--radar_align_mode", default=os.environ.get("ASY_RADAR_ALIGN_MODE", "letterbox"))
    parser.add_argument("--radar_normalize", action="store_true", default=env_bool("ASY_RADAR_NORMALIZE", False))
    parser.add_argument("--no_radar_normalize", action="store_false", dest="radar_normalize")
    parser.add_argument("--radar_preserve_points", action="store_true", default=env_bool("ASY_RADAR_PRESERVE_POINTS", True))
    parser.add_argument("--no_radar_preserve_points", action="store_false", dest="radar_preserve_points")
    parser.add_argument("--radar_source_order", default=os.environ.get("ASY_RADAR_SOURCE_ORDER", "range,doppler,elevation,power"))
    parser.add_argument("--radar_target_order", default=os.environ.get("ASY_RADAR_TARGET_ORDER", "range,elevation,velocity,power"))
    parser.add_argument("--fusion_mode", default=os.environ.get("ASY_FUSION_MODE", "baseline"))
    parser.add_argument("--task_loss", default=os.environ.get("ASY_TASK_LOSS", "uncertainty"))
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--no_cuda", action="store_false", dest="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    args.val_txt = str(resolve_path(args.val_txt))
    args.model_path = str(resolve_path(args.model_path))
    args.classes_path = str(resolve_path(args.classes_path))
    args.radar_root = str(resolve_path(args.radar_root))
    args.vocdevkit_path = str(resolve_path(args.vocdevkit_path))
    args.out_dir = str(resolve_path(args.out_dir))

    pred_dir = Path(args.out_dir) / "detection-results"
    pred_dir.mkdir(parents=True, exist_ok=True)

    with open(args.val_txt, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    model = YOLO(
        model_path=args.model_path,
        classes_path=args.classes_path,
        radar_root=args.radar_root,
        input_shape=args.input_shape,
        num_seg_classes=args.num_seg_classes,
        radar_in_channels=args.radar_channels,
        radar_align_mode=args.radar_align_mode,
        radar_normalize=args.radar_normalize,
        radar_preserve_points=args.radar_preserve_points,
        radar_source_order=args.radar_source_order,
        radar_target_order=args.radar_target_order,
        fusion_mode=args.fusion_mode,
        task_loss_mode=args.task_loss,
        phi=args.phi,
        cuda=args.cuda,
    )

    image_ids = []
    for line in tqdm(lines, desc="Segmentation"):
        image_path = resolve_path(line.split()[0])
        image_id = image_path.stem
        image_ids.append(image_id)
        image = Image.open(image_path)
        save_segmentation_png(model, image, image_id, args, str(pred_dir))

    gt_dir = Path(args.vocdevkit_path) / "VOC2007" / "SegmentationClass"
    hist, ious, pa_recall, precision = compute_mIoU(
        str(gt_dir),
        str(pred_dir),
        image_ids,
        args.num_seg_classes,
        SEG_CLASSES,
    )
    show_results(args.out_dir, hist, ious, pa_recall, precision, SEG_CLASSES)


if __name__ == "__main__":
    main()
