"""Download the BDD100K dataset via KaggleHub."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import kagglehub

DEFAULT_KAGGLE_DATASET = "solesensei/solesensei_bdd100k"

logger = logging.getLogger(__name__)


def download_bdd100k(
    dataset_handle: str = DEFAULT_KAGGLE_DATASET,
    output_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """Download BDD100K from KaggleHub and return the local dataset path."""
    kwargs: dict[str, object] = {"force_download": force_download}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        kwargs["output_dir"] = str(output_dir)

    logger.info("Downloading Kaggle dataset '%s'...", dataset_handle)
    downloaded_path = kagglehub.dataset_download(dataset_handle, **kwargs)
    return Path(downloaded_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for BDD100K download."""
    parser = argparse.ArgumentParser(
        description="Download the BDD100K dataset using KaggleHub."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_KAGGLE_DATASET,
        help=f"KaggleHub dataset handle (default: {DEFAULT_KAGGLE_DATASET}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional target directory (if omitted, uses KaggleHub default cache).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved download configuration without downloading files.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for downloading BDD100K."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    if args.dry_run:
        target = str(args.output_dir) if args.output_dir else "<kagglehub cache>"
        logger.info(
            "[DRY RUN] Would download '%s' -> %s (force=%s)",
            args.dataset,
            target,
            args.force,
        )
        return

    dataset_path = download_bdd100k(
        dataset_handle=args.dataset,
        output_dir=args.output_dir,
        force_download=args.force,
    )
    logger.info("Dataset downloaded to: %s", dataset_path)
    logger.info(
        "Update 'paths.bdd_root' in configs/paths.yaml (or BDD_ROOT in .env) to this path."
    )


if __name__ == "__main__":
    main()
