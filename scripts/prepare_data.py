"""Prepare deterministic subset splits and convert BDD100K detection JSON labels to YOLO."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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

logger = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for data preparation."""
    parser = argparse.ArgumentParser(
        description="Create deterministic train/val splits and YOLO label files from BDD100K."
    )
    parser.add_argument("--bdd-root", type=Path, default=None, help="Override cfg.paths.bdd_root.")
    parser.add_argument(
        "--images-subpath",
        type=str,
        default="images/100k",
        help="Relative directory inside bdd_root containing train/val folders.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="Directory containing BDD100K detection JSON files.",
    )
    parser.add_argument(
        "--yolo-out",
        type=Path,
        default=Path("data/bdd100k_yolo/labels"),
        help="Output directory for converted YOLO .txt label files.",
    )
    parser.add_argument("--max-train", type=int, default=None, help="Max train subset size.")
    parser.add_argument("--max-val", type=int, default=None, help="Max val subset size.")
    return parser.parse_args()


def main() -> None:
    """Entry point for preparing splits and YOLO labels."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    cfg = load_project_configs()
    seed = int(cfg.project.seed)
    bdd_root = args.bdd_root if args.bdd_root is not None else Path(cfg.paths.bdd_root)
    splits_dir = Path(cfg.paths.splits_dir)

    if not bdd_root.exists():
        logger.error("BDD100K root does not exist: %s", bdd_root)
        sys.exit(1)

    for split_name, max_items in (("train", args.max_train), ("val", args.max_val)):
        split_img_dir = bdd_root / args.images_subpath / split_name
        if not split_img_dir.exists():
            logger.warning("Skipping missing image split directory: %s", split_img_dir)
            continue

        all_stems = [
            p.stem
            for p in split_img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        chosen = create_subset_split(all_stems, max_samples=max_items, seed=seed)
        split_out = write_split_file(chosen, splits_dir / f"{split_name}.txt")
        logger.info("Wrote %d stems to %s (seed=%d)", len(chosen), split_out, seed)

        if args.labels_dir is not None:
            json_candidates = sorted(args.labels_dir.glob(f"*{split_name}*.json"))
            if not json_candidates:
                logger.warning("No JSON label file found matching *%s*.json", split_name)
                continue
            convert_bdd_json_to_yolo_dir(
                json_path=json_candidates[0],
                images_dir=split_img_dir,
                labels_out_dir=args.yolo_out / split_name,
                allowed_stems=set(chosen),
            )


if __name__ == "__main__":
    main()
