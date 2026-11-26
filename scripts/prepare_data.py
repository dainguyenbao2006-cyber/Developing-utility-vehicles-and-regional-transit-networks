"""Prepare splits, writable YOLO structure, and segmentation masks from original BDD100K."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data.bdd_dataset import resolve_bdd_seg_dirs  # noqa: E402
from data.converters import (  # noqa: E402
    _extract_bdd_frames,
    convert_bdd_json_to_yolo_dir,
    create_subset_split,
    link_or_copy_file,
    rasterize_bdd_json_seg_to_dirs,
    write_split_file,
)
from utils.config import load_project_configs  # noqa: E402
from utils.constants import DET_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def resolve_bdd_images_base(bdd_root: Path, override_subpath: str | None = None) -> Path:
    """Locate the parent directory containing 'train' and 'val' image folders."""
    if override_subpath is not None:
        return bdd_root / override_subpath

    candidates = [
        bdd_root / "images" / "100k",
        bdd_root / "bdd100k" / "images" / "100k",
        bdd_root / "bdd100k" / "bdd100k" / "images" / "100k",
        bdd_root / "bdd100k_images_100k" / "bdd100k" / "images" / "100k",
    ]
    for candidate in candidates:
        if (candidate / "train").exists() or (candidate / "val").exists():
            return candidate
    return candidates[0]


def find_split_detection_json(
    bdd_root: Path,
    labels_dir: Path | None,
    split_name: str,
) -> Path | None:
    """Find the BDD100K detection JSON file corresponding to 'train' or 'val'."""
    search_roots = (
        [labels_dir]
        if labels_dir is not None
        else [
            bdd_root / "labels" / "det_20",
            bdd_root / "labels",
            bdd_root / "bdd100k" / "labels" / "det_20",
            bdd_root / "bdd100k" / "labels",
            bdd_root / "bdd100k_labels_release" / "bdd100k" / "labels",
            bdd_root,
        ]
    )
    for root in search_roots:
        if root is None or not root.exists():
            continue
        for cand in sorted(root.rglob(f"*{split_name}*.json")):
            low = cand.name.lower()
            if "det" in low or "bdd100k_labels_images" in low:
                return cand
        fallback = sorted(root.glob(f"*{split_name}*.json"))
        if fallback:
            return fallback[0]
    return None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for BDD100K split generation and label/mask conversion."""
    default_work = Path(os.environ.get("BDD_WORK_DIR", "data/bdd_prepared"))
    parser = argparse.ArgumentParser(
        description="Create splits, writable YOLO dataset, and segmentation masks from BDD100K."
    )
    parser.add_argument("--bdd-root", type=Path, default=None, help="Override cfg.paths.bdd_root.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=default_work,
        help="Writable output directory for YOLO links/labels and rasterized masks.",
    )
    parser.add_argument(
        "--images-subpath",
        type=str,
        default=None,
        help="Optional relative path inside bdd_root containing train/val image folders.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Optional directory containing BDD100K detection JSON files.",
    )
    parser.add_argument(
        "--yolo-yaml",
        type=Path,
        default=Path("configs/bdd_yolo.yaml"),
        help="Ultralytics dataset YAML file to update.",
    )
    parser.add_argument("--max-train", type=int, default=None, help="Max train subset size.")
    parser.add_argument("--max-val", type=int, default=None, help="Max val subset size.")
    return parser.parse_args()


def prepare_dataset_splits(
    bdd_root: Path,
    work_dir: Path,
    splits_dir: Path,
    yolo_yaml_path: Path,
    seed: int = 42,
    images_subpath: str | None = None,
    labels_dir: Path | None = None,
    max_train: int | None = None,
    max_val: int | None = None,
) -> dict[str, Any]:
    """Build splits, writable YOLO image/label folders, and PNG masks (read-only root safe)."""
    images_base = resolve_bdd_images_base(bdd_root, images_subpath)
    work_dir = work_dir.resolve()
    splits_dir = splits_dir.resolve()
    summary: dict[str, Any] = {}

    for split_name, max_items in (("train", max_train), ("val", max_val)):
        split_img_dir = images_base / split_name
        if not split_img_dir.exists():
            logger.warning("Skipping missing image split directory: %s", split_img_dir)
            continue

        # Scan recursively to catch nested subfolders like trainA / trainB on Kaggle
        stem_to_path: dict[str, Path] = {}
        for p in sorted(split_img_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                stem_to_path.setdefault(p.stem, p.resolve())

        if not stem_to_path:
            logger.warning("No images found inside %s", split_img_dir)
            continue

        chosen_stems = create_subset_split(
            list(stem_to_path.keys()), max_samples=max_items, seed=seed
        )
        split_out = write_split_file(chosen_stems, splits_dir / f"{split_name}.txt")
        logger.info(
            "Found %d total '%s' images; wrote %d stems to %s (seed=%d)",
            len(stem_to_path),
            split_name,
            len(chosen_stems),
            split_out,
            seed,
        )

        # Link/copy chosen images into writable <work_dir>/yolo/images/<split>
        yolo_img_out = work_dir / "yolo" / "images" / split_name
        yolo_lbl_out = work_dir / "yolo" / "labels" / split_name
        yolo_img_out.mkdir(parents=True, exist_ok=True)
        yolo_lbl_out.mkdir(parents=True, exist_ok=True)

        for stem in chosen_stems:
            src_img = stem_to_path[stem]
            link_or_copy_file(src_img, yolo_img_out / src_img.name)

        first_img = stem_to_path[chosen_stems[0]]
        with Image.open(first_img) as sample_im:
            detected_wh = sample_im.size

        det_json = find_split_detection_json(bdd_root, labels_dir, split_name)
        if det_json is not None:
            logger.info("Loading annotations once from %s ...", det_json)
            with det_json.open("r", encoding="utf-8") as handle:
                frames = _extract_bdd_frames(json.load(handle))

            chosen_set = set(chosen_stems)
            det_stats = convert_bdd_json_to_yolo_dir(
                json_path=det_json,
                images_dir=split_img_dir,
                labels_out_dir=yolo_lbl_out,
                allowed_stems=chosen_set,
                stem_to_image=stem_to_path,
                preloaded_frames=frames,
                fixed_size_wh=detected_wh,
            )

            _, existing_drv, existing_lane = resolve_bdd_seg_dirs(
                bdd_root, split_name, work_dir=work_dir
            )
            seg_stats: dict[str, int] = {}
            if existing_drv is None or existing_lane is None:
                drv_out = work_dir / "seg_masks" / "drivable" / split_name
                lane_out = work_dir / "seg_masks" / "lane" / split_name
                seg_stats = rasterize_bdd_json_seg_to_dirs(
                    json_path=det_json,
                    images_dir=split_img_dir,
                    drivable_out_dir=drv_out,
                    lane_out_dir=lane_out,
                    allowed_stems=chosen_set,
                    preloaded_frames=frames,
                    fixed_size_wh=detected_wh,
                )
            summary[split_name] = {
                "total_discovered": len(stem_to_path),
                "selected": len(chosen_stems),
                "det": det_stats,
                "seg": seg_stats,
            }
        else:
            logger.warning("No detection JSON found for '%s'.", split_name)

    yolo_root = work_dir / "yolo"
    if (yolo_root / "images" / "train").exists() and (yolo_root / "images" / "val").exists():
        yolo_cfg = {
            "path": yolo_root.as_posix(),
            "train": "images/train",
            "val": "images/val",
            "names": {idx: name for idx, name in enumerate(DET_CLASSES)},
        }
        yolo_yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with yolo_yaml_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(yolo_cfg, handle, sort_keys=False)
        logger.info("Updated Ultralytics dataset config: %s -> %s", yolo_yaml_path, yolo_root)

    return summary


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    cfg = load_project_configs()
    bdd_root = (args.bdd_root if args.bdd_root is not None else Path(cfg.paths.bdd_root)).resolve()

    if not bdd_root.exists():
        logger.error("BDD100K root does not exist: %s", bdd_root)
        sys.exit(1)

    prepare_dataset_splits(
        bdd_root=bdd_root,
        work_dir=args.work_dir,
        splits_dir=Path(cfg.paths.splits_dir),
        yolo_yaml_path=args.yolo_yaml,
        seed=int(cfg.project.seed),
        images_subpath=args.images_subpath,
        labels_dir=args.labels_dir,
        max_train=args.max_train,
        max_val=args.max_val,
    )


if __name__ == "__main__":
    main()
