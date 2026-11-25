"""Inspect actual BDD100K directory layout, JSON categories, and segmentation mask values."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Allow running directly as `python scripts/inspect_bdd.py` without PYTHONPATH=src
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import load_project_configs  # noqa: E402

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def discover_subdirectories(root: Path, max_depth: int = 3) -> list[str]:
    """List relative directory paths up to max_depth under root."""
    dirs: list[str] = []
    if not root.exists():
        return dirs
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            rel = path.relative_to(root)
            if len(rel.parts) <= max_depth:
                dirs.append(rel.as_posix())
    return dirs


def find_json_label_files(root: Path) -> list[Path]:
    """Locate JSON annotation files inside the dataset root."""
    return sorted(root.rglob("*.json"))


def inspect_detection_json(json_path: Path, max_frames: int = 5000) -> dict[str, Any]:
    """Inspect a BDD100K detection JSON file for schema and category distributions."""
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and "frames" in data:
        frames = data["frames"]
        schema_type = "dict['frames']"
    elif isinstance(data, list):
        frames = data
        schema_type = "list"
    else:
        return {
            "path": str(json_path),
            "schema_type": type(data).__name__,
            "frame_count": 0,
            "categories": {},
            "attribute_keys": [],
        }

    category_counts: Counter[str] = Counter()
    attribute_keys: set[str] = set()
    sample_frames = frames[:max_frames]

    for frame in sample_frames:
        if not isinstance(frame, dict):
            continue
        attrs = frame.get("attributes")
        if isinstance(attrs, dict):
            attribute_keys.update(attrs.keys())

        labels = frame.get("labels") or []
        for label in labels:
            if not isinstance(label, dict):
                continue
            cat = label.get("category")
            if isinstance(cat, str):
                category_counts[cat] += 1

    return {
        "path": str(json_path),
        "schema_type": schema_type,
        "total_frames_in_file": len(frames),
        "inspected_frames": len(sample_frames),
        "categories": dict(sorted(category_counts.items())),
        "attribute_keys": sorted(attribute_keys),
    }


def inspect_mask_directory(mask_dir: Path, sample_limit: int = 20) -> dict[str, Any]:
    """Inspect shapes, dtypes, and unique pixel values/RGB tuples in a mask directory."""
    png_files = sorted(mask_dir.rglob("*.png"))
    if not png_files:
        return {"dir": str(mask_dir), "png_count": 0}

    sampled = png_files[:sample_limit]
    shapes: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    unique_gray_values: set[int] = set()
    unique_rgb_tuples: set[tuple[int, int, int]] = set()

    for mask_path in sampled:
        arr = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if arr is None:
            continue
        shapes[str(arr.shape)] += 1
        dtypes[str(arr.dtype)] += 1
        if arr.ndim == 2:
            unique_gray_values.update(int(v) for v in np.unique(arr))
        elif arr.ndim == 3:
            flat = arr.reshape(-1, arr.shape[2])
            uniq = np.unique(flat, axis=0)
            for row in uniq[:50]:
                unique_rgb_tuples.add(tuple(int(c) for c in row[:3]))

    return {
        "dir": str(mask_dir),
        "png_count": len(png_files),
        "sampled": len(sampled),
        "shapes": dict(shapes),
        "dtypes": dict(dtypes),
        "unique_gray_values": sorted(unique_gray_values),
        "sample_unique_bgr_tuples": sorted(unique_rgb_tuples)[:30],
    }


def run_inspection(
    bdd_root: Path,
    sample_masks: int = 20,
    max_json_frames: int = 5000,
) -> dict[str, Any]:
    """Run full inspection of BDD100K files at bdd_root."""
    if not bdd_root.exists():
        raise FileNotFoundError(f"BDD100K root does not exist: {bdd_root}")

    subdirs = discover_subdirectories(bdd_root, max_depth=3)
    json_files = find_json_label_files(bdd_root)
    json_reports = [inspect_detection_json(p, max_frames=max_json_frames) for p in json_files[:10]]

    candidate_mask_dirs: list[Path] = []
    for rel in subdirs:
        low = rel.lower()
        if any(tok in low for tok in ("drivable", "lane", "seg", "label", "mask", "colormap")):
            candidate_path = bdd_root / rel
            if any(candidate_path.glob("*.png")):
                candidate_mask_dirs.append(candidate_path)

    mask_reports = [
        inspect_mask_directory(d, sample_limit=sample_masks) for d in candidate_mask_dirs[:12]
    ]

    image_counts: dict[str, int] = {}
    for rel in subdirs:
        p = bdd_root / rel
        if "image" in rel.lower() or rel.split("/")[-1] in {"train", "val", "test"}:
            cnt = sum(1 for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
            if cnt > 0:
                image_counts[rel] = cnt

    return {
        "bdd_root": str(bdd_root),
        "subdirs_depth_3": subdirs,
        "image_counts": image_counts,
        "json_files_found": [str(p) for p in json_files],
        "json_reports": json_reports,
        "mask_reports": mask_reports,
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for BDD100K inspection."""
    parser = argparse.ArgumentParser(
        description="Inspect actual BDD100K structure, JSON labels, and PNG mask pixel values."
    )
    parser.add_argument(
        "--bdd-root",
        type=Path,
        default=None,
        help="Optional override for cfg.paths.bdd_root.",
    )
    parser.add_argument(
        "--sample-masks",
        type=int,
        default=20,
        help="Number of PNG masks per folder to sample (default: 20).",
    )
    parser.add_argument(
        "--max-json-frames",
        type=int,
        default=5000,
        help="Maximum JSON frames to scan per file (default: 5000).",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save inspection report as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for BDD100K dataset inspection."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    cfg = load_project_configs()
    bdd_root = args.bdd_root if args.bdd_root is not None else Path(cfg.paths.bdd_root)

    if not bdd_root.exists():
        logger.error(
            "BDD100K root not found at: %s. "
            "Download dataset first or pass --bdd-root <path>.",
            bdd_root,
        )
        sys.exit(1)

    report = run_inspection(
        bdd_root=bdd_root,
        sample_masks=args.sample_masks,
        max_json_frames=args.max_json_frames,
    )
    logger.info("BDD100K Inspection Summary:\n%s", json.dumps(report, indent=2))

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        with args.save_json.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        logger.info("Saved inspection report to %s", args.save_json)


if __name__ == "__main__":
    main()
