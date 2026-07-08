"""Select WaterScenes ids for visual evidence mining.

The script uses only metadata and detection XML labels, so it can run before
the large image archive is available. It prioritizes adverse scenes and small
objects, then writes an id list for subset extraction and batch comparison.
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select WaterScenes candidate cases.")
    parser.add_argument("--info-csv", required=True, help="WaterScenes information_list.csv.")
    parser.add_argument("--detection-zip", default="", help="Optional detection.zip path.")
    parser.add_argument("--xml-dir", default="", help="Optional unpacked detection XML directory.")
    parser.add_argument("--split-file", default="", help="Optional split txt to restrict ids.")
    parser.add_argument("--out", default="outputs/candidate_ids_adverse_small.txt")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--small-area", type=float, default=0.004, help="Small-object area ratio threshold.")
    parser.add_argument(
        "--require-keyword",
        nargs="*",
        default=None,
        help="Keep only cases whose metadata contains one of these keywords.",
    )
    return parser.parse_args()


def load_split(path: str) -> set[str] | None:
    if not path:
        return None
    ids = set()
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        item = line.strip().split()[0] if line.strip() else ""
        if item:
            ids.add(Path(item).stem)
    return ids


def load_info(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get("num") or row.get("id") or row.get("image_id") or next(iter(row.values()))
            rows[Path(str(image_id)).stem] = {k: str(v).lower() for k, v in row.items()}
    return rows


def read_xml_from_zip(zf: zipfile.ZipFile, image_id: str) -> str | None:
    suffix = f"/{image_id}.xml"
    direct = f"detection/xml/{image_id}.xml"
    names = zf.namelist()
    if direct in names:
        return zf.read(direct).decode("utf-8", "ignore")
    for name in names:
        if name.endswith(suffix) or name == f"{image_id}.xml":
            return zf.read(name).decode("utf-8", "ignore")
    return None


def parse_xml(xml_text: str) -> tuple[int, int, list[float]]:
    root = ET.fromstring(xml_text)
    width = int(root.findtext("size/width") or 1920)
    height = int(root.findtext("size/height") or 1080)
    areas = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin") or 0)
        ymin = float(box.findtext("ymin") or 0)
        xmax = float(box.findtext("xmax") or 0)
        ymax = float(box.findtext("ymax") or 0)
        area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin) / max(1.0, width * height)
        areas.append(area)
    return width, height, areas


def iter_label_stats(args: argparse.Namespace, ids: set[str] | None):
    if args.detection_zip:
        with zipfile.ZipFile(args.detection_zip) as zf:
            xml_ids = sorted({Path(name).stem for name in zf.namelist() if name.endswith(".xml")})
            for image_id in xml_ids:
                if ids is not None and image_id not in ids:
                    continue
                text = read_xml_from_zip(zf, image_id)
                if text:
                    yield image_id, parse_xml(text)
    elif args.xml_dir:
        for path in sorted(Path(args.xml_dir).glob("*.xml")):
            image_id = path.stem
            if ids is not None and image_id not in ids:
                continue
            yield image_id, parse_xml(path.read_text(encoding="utf-8", errors="ignore"))
    else:
        raise SystemExit("Provide --detection-zip or --xml-dir.")


def score_case(info: dict[str, str], obj_count: int, areas: list[float], small_area: float) -> float:
    text = " ".join(info.values())
    score = 0.0
    for keyword, weight in [
        ("fog", 4.0),
        ("rain", 3.0),
        ("night", 3.0),
        ("low", 2.0),
        ("normal", -0.5),
    ]:
        if keyword in text:
            score += weight
    small_count = sum(1 for area in areas if area <= small_area)
    score += min(obj_count, 12) * 0.35
    score += small_count * 1.25
    if obj_count >= 8:
        score += 2.0
    return score


def main() -> None:
    args = parse_args()
    info = load_info(Path(args.info_csv))
    split_ids = load_split(args.split_file)
    rows = []
    for image_id, (_, _, areas) in iter_label_stats(args, split_ids):
        meta = info.get(image_id, {})
        meta_text = " ".join(meta.values())
        if args.require_keyword and not any(keyword.lower() in meta_text for keyword in args.require_keyword):
            continue
        obj_count = len(areas)
        if obj_count == 0:
            continue
        small_count = sum(1 for area in areas if area <= args.small_area)
        score = score_case(meta, obj_count, areas, args.small_area)
        rows.append((score, image_id, obj_count, small_count, min(areas), " ".join(meta.values())))

    rows.sort(reverse=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "score", "obj_count", "small_count", "min_area", "metadata"])
        for score, image_id, obj_count, small_count, min_area, meta in rows[: args.top_k]:
            writer.writerow([image_id, f"{score:.3f}", obj_count, small_count, f"{min_area:.6f}", meta])

    ids_out = out.with_suffix(".ids.txt")
    ids_out.write_text("\n".join(row[1] for row in rows[: args.top_k]) + "\n", encoding="utf-8")
    print(f"wrote {min(len(rows), args.top_k)} cases -> {out}")
    print(f"wrote ids -> {ids_out}")


if __name__ == "__main__":
    main()
