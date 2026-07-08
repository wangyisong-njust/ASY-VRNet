"""Extract selected WaterScenes files without unpacking the full dataset.

The official WaterScenes image archive is large, and server disk is limited.
This helper extracts only requested ids from zip archives and converts
radar_5_frames CSV files into the NPZ format used by this project.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radar_csv_to_npz import csv_to_feature_map


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a small WaterScenes subset.")
    parser.add_argument("--ids", nargs="*", default=None, help="Image ids to extract.")
    parser.add_argument("--ids-file", default="", help="Optional file containing one id per line.")
    parser.add_argument("--image-zip", default="", help="Optional WaterScenes image.zip path.")
    parser.add_argument("--radar-zip", default="", help="Optional WaterScenes radar_5_frames.zip path.")
    parser.add_argument("--detection-zip", default="", help="Optional WaterScenes detection.zip path.")
    parser.add_argument("--image-out", default="image")
    parser.add_argument("--radar-csv-out", default="dataset/WaterScenes_radar_5_frames_csv")
    parser.add_argument("--radar-npz-out", default="dataset/VOCradar_5_frames")
    parser.add_argument("--xml-out", default="dataset/WaterScenes_detection_xml")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    return parser.parse_args()


def load_ids(args: argparse.Namespace) -> list[str]:
    ids: list[str] = []
    if args.ids_file:
        for line in Path(args.ids_file).read_text(encoding="utf-8", errors="ignore").splitlines():
            item = line.strip().split()[0] if line.strip() else ""
            if item:
                ids.append(Path(item).stem)
    if args.ids:
        ids.extend(Path(item).stem for item in args.ids)
    return sorted(dict.fromkeys(ids))


def find_zip_member(zf: zipfile.ZipFile, wanted_name: str, exts: tuple[str, ...]) -> str | None:
    wanted_stem = Path(wanted_name).stem
    for name in zf.namelist():
        path = Path(name)
        if path.stem == wanted_stem and path.suffix.lower() in exts:
            return name
    return None


def extract_one(zf: zipfile.ZipFile, member: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)
    return True


def extract_images(ids: list[str], image_zip: Path, image_out: Path) -> dict[str, str]:
    results: dict[str, str] = {}
    if not image_zip:
        return results
    with zipfile.ZipFile(image_zip) as zf:
        for image_id in ids:
            member = find_zip_member(zf, image_id, IMAGE_EXTS)
            if member is None:
                results[image_id] = "missing_image"
                continue
            out_path = image_out / f"{image_id}{Path(member).suffix.lower()}"
            extract_one(zf, member, out_path)
            results[image_id] = str(out_path)
    return results


def extract_xml(ids: list[str], detection_zip: Path, xml_out: Path) -> dict[str, str]:
    results: dict[str, str] = {}
    if not detection_zip:
        return results
    with zipfile.ZipFile(detection_zip) as zf:
        for image_id in ids:
            member = find_zip_member(zf, image_id, (".xml",))
            if member is None:
                results[image_id] = "missing_xml"
                continue
            out_path = xml_out / f"{image_id}.xml"
            extract_one(zf, member, out_path)
            results[image_id] = str(out_path)
    return results


def extract_and_convert_radar(
    ids: list[str],
    radar_zip: Path,
    csv_out: Path,
    npz_out: Path,
    height: int,
    width: int,
) -> dict[str, str]:
    results: dict[str, str] = {}
    if not radar_zip:
        return results
    csv_out.mkdir(parents=True, exist_ok=True)
    npz_out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(radar_zip) as zf:
        for image_id in ids:
            member = find_zip_member(zf, image_id, (".csv",))
            if member is None:
                results[image_id] = "missing_radar"
                continue
            csv_path = csv_out / f"{image_id}.csv"
            npz_path = npz_out / f"{image_id}.npz"
            extract_one(zf, member, csv_path)
            feature_map = csv_to_feature_map(str(csv_path), height, width)
            import numpy as np

            np.savez_compressed(str(npz_path), feature_map)
            results[image_id] = str(npz_path)
    return results


def main() -> None:
    args = parse_args()
    ids = load_ids(args)
    if not ids:
        raise SystemExit("No ids given. Use --ids or --ids-file.")

    image_results = extract_images(ids, Path(args.image_zip) if args.image_zip else None, Path(args.image_out))
    xml_results = extract_xml(ids, Path(args.detection_zip) if args.detection_zip else None, Path(args.xml_out))
    radar_results = extract_and_convert_radar(
        ids,
        Path(args.radar_zip) if args.radar_zip else None,
        Path(args.radar_csv_out),
        Path(args.radar_npz_out),
        args.height,
        args.width,
    )

    print("image_id,image,radar_npz,xml")
    for image_id in ids:
        print(
            f"{image_id},"
            f"{image_results.get(image_id, '')},"
            f"{radar_results.get(image_id, '')},"
            f"{xml_results.get(image_id, '')}"
        )


if __name__ == "__main__":
    main()
