import argparse
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_lines(path):
    if not path.exists():
        raise FileNotFoundError(f"missing txt file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_split(split_name, lines, vocdevkit, radar_root, limit):
    missing = {
        "image": 0,
        "segmentation": 0,
        "radar": 0,
        "empty_boxes": 0,
    }
    checked = 0
    for line in lines[:limit]:
        parts = line.split()
        if len(parts) < 2:
            missing["empty_boxes"] += 1
            continue
        image_path = resolve_path(parts[0])
        stem = image_path.stem
        seg_path = vocdevkit / "VOC2007" / "SegmentationClass" / f"{stem}.png"
        radar_path = radar_root / f"{stem}.npz"
        missing["image"] += int(not image_path.exists())
        missing["segmentation"] += int(not seg_path.exists())
        missing["radar"] += int(not radar_path.exists())
        checked += 1
    print(f"{split_name}: total={len(lines)} checked={checked} missing={missing}")
    return sum(missing.values())


def main():
    parser = argparse.ArgumentParser(description="Check ASY-VRNet VOC + radar dataset paths.")
    parser.add_argument("--train_txt", default=os.environ.get("ASY_TRAIN_TXT", "2007_train.txt"))
    parser.add_argument("--val_txt", default=os.environ.get("ASY_VAL_TXT", "2007_val.txt"))
    parser.add_argument("--vocdevkit", default=os.environ.get("ASY_VOCDEVKIT", str(PROJECT_ROOT / "dataset" / "VOCdevkit")))
    parser.add_argument("--radar_root", default=os.environ.get("ASY_RADAR_ROOT", str(PROJECT_ROOT / "dataset" / "VOCradar_5_frames")))
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    train_txt = resolve_path(args.train_txt)
    val_txt = resolve_path(args.val_txt)
    vocdevkit = resolve_path(args.vocdevkit)
    radar_root = resolve_path(args.radar_root)

    print(f"train_txt={train_txt.resolve()}")
    print(f"val_txt={val_txt.resolve()}")
    print(f"vocdevkit={vocdevkit}")
    print(f"radar_root={radar_root}")

    train_lines = read_lines(train_txt)
    val_lines = read_lines(val_txt)
    errors = 0
    errors += check_split("train", train_lines, vocdevkit, radar_root, args.limit)
    errors += check_split("val", val_lines, vocdevkit, radar_root, args.limit)
    if errors:
        raise SystemExit("dataset check failed")
    print("dataset check passed")


if __name__ == "__main__":
    main()
