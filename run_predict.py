"""Batch detection visualization for validation image/radar pairs."""
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, '.')

from PIL import Image
from tqdm import tqdm

from yolo import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_shape(value):
    parts = [int(part) for part in value.replace(",", " ").split()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--input_shape must contain two integers, for example 320,320")
    return parts


def parse_args():
    parser = argparse.ArgumentParser(description="Save ASY-VRNet detection visualizations for a txt split.")
    parser.add_argument("--val_txt", default=str(PROJECT_ROOT / "2007_val.txt"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "predict_output"))
    parser.add_argument("--model_path", default=os.environ.get("ASY_MODEL_PATH", ""))
    parser.add_argument("--classes_path", default=os.environ.get("ASY_CLASSES_PATH", ""))
    parser.add_argument("--radar_root", default=os.environ.get("ASY_RADAR_ROOT", str(PROJECT_ROOT / "dataset" / "VOCradar_5_frames")))
    parser.add_argument("--input_shape", type=parse_shape, default=None)
    parser.add_argument("--phi", default=os.environ.get("ASY_PHI", ""))
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--limit", type=int, default=0, help="0 means process the whole split.")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(resolve_path(args.val_txt), encoding="utf-8") as f:
        val_lines = [line.strip().split()[0] for line in f if line.strip()]
    if args.limit > 0:
        val_lines = val_lines[:args.limit]

    kwargs = {
        "radar_root": args.radar_root,
    }
    for key in ["model_path", "classes_path", "input_shape", "phi", "confidence"]:
        value = getattr(args, key)
        if value not in (None, ""):
            kwargs[key] = value
    det_model = YOLO(**kwargs)

    success = 0
    for img_path in tqdm(val_lines, desc="Detection"):
        try:
            img_path = resolve_path(img_path)
            image = Image.open(img_path)
            stem = img_path.stem
            r_image = det_model.detect_image(image, stem, crop=False, count=False)
            r_image.save(output_dir / f"det_{stem}.jpg")
            success += 1
        except Exception as exc:
            print(f"[WARN] {img_path}: {exc}")

    print(f"Saved detection visualizations: {success}/{len(val_lines)} -> {output_dir}")


if __name__ == "__main__":
    main()
