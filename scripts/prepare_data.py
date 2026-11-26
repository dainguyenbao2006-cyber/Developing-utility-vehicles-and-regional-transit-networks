"""Prepare deterministic splits and Ultralytics YOLO labels directly on original BDD100K layout."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data.converters import (  # noqa: E402
    convert_bdd_json_to_yolo_dir,
    create_subset_split,
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
    search_roots = [labels_dir] if labels_dir is not None else [
        bdd_root / "labels" / "det_20",
        bdd_root / "labels",
        bdd_root / "bdd100k" / "labels" / "det_20",
        bdd_root / "bdd100k" / "labels",
        bdd_root,
    ]
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


def _derive_yolo_label_dir(split_img_dir: Path) -> Path:
    """Mirror Ultralytics img2label_paths rule replacing last 'images' segment with 'labels'."""
    parts = list(split_img_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "images":
            parts[idx] = "labels"
            return Path(*parts)
    return split_img_dir.parent.parent / "labels_yolo" / split_img_dir.name


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for BDD100K split generation and YOLO label conversion."""
    parser = argparse.ArgumentParser(
        description="Create deterministic splits and convert BDD100K JSON labels for YOLO11."
    )
    parser.add_argument("--bdd-root", type=Path, default=None, help="Override cfg.paths.bdd_root.")
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
        help="Ultralytics dataset YAML file to update with generated split lists.",
    )
    parser.add_argument("--max-train", type=int, default=None, help="Max train subset size.")
    parser.add_argument("--max-val", type=int, default=None, help="Max val subset size.")
    return parser.parse_args()


def main() -> None:
    """Entry point for preparing splits, YOLO labels, and Ultralytics dataset YAML."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    cfg = load_project_configs()
    seed = int(cfg.project.seed)
    bdd_root = (args.bdd_root if args.bdd_root is not None else Path(cfg.paths.bdd_root)).resolve()
    splits_dir = Path(cfg.paths.splits_dir).resolve()

    if not bdd_root.exists():
        logger.error("BDD100K root does not exist: %s", bdd_root)
        sys.exit(1)

    images_base = resolve_bdd_images_base(bdd_root, args.images_subpath)
    yolo_split_files: dict[str, str] = {}

    for split_name, max_items in (("train", args.max_train), ("val", args.max_val)):
        split_img_dir = images_base / split_name
        if not split_img_dir.exists():
            logger.warning("Skipping missing image split directory: %s", split_img_dir)
            continue

        stem_to_path: dict[str, Path] = {}
        for p in sorted(split_img_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                stem_to_path[p.stem] = p.resolve()

        chosen_stems = create_subset_split(
            list(stem_to_path.keys()), max_samples=max_items, seed=seed
        )
        split_out = write_split_file(chosen_stems, splits_dir / f"{split_name}.txt")
        logger.info("Wrote %d stems to %s (seed=%d)", len(chosen_stems), split_out, seed)

        yolo_list_path = splits_dir / f"yolo_{split_name}.txt"
        image_lines = [stem_to_path[s].as_posix() for s in chosen_stems]
        write_split_file(image_lines, yolo_list_path)
        yolo_split_files[split_name] = yolo_list_path.as_posix()

        det_json = find_split_detection_json(bdd_root, args.labels_dir, split_name)
        if det_json is not None:
            yolo_lbl_dir = _derive_yolo_label_dir(split_img_dir)
            convert_bdd_json_to_yolo_dir(
                json_path=det_json,
                images_dir=split_img_dir,
                labels_out_dir=yolo_lbl_dir,
                allowed_stems=set(chosen_stems),
            )
        else:
            logger.warning(
                "No detection JSON found for '%s'; pass --labels-dir if stored separately.",
                split_name,
            )

    if "train" in yolo_split_files and "val" in yolo_split_files:
        yolo_cfg = {
            "path": _PROJECT_ROOT.resolve().as_posix(),
            "train": yolo_split_files["train"],
            "val": yolo_split_files["val"],
            "names": {idx: name for idx, name in enumerate(DET_CLASSES)},
        }
        args.yolo_yaml.parent.mkdir(parents=True, exist_ok=True)
        with args.yolo_yaml.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(yolo_cfg, handle, sort_keys=False)
        logger.info("Updated Ultralytics dataset config: %s", args.yolo_yaml)


if __name__ == "__main__":
    main()
