"""Train and benchmark YOLO11 object detection model on BDD100K dataset."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

# Prevent Ultralytics from warning about non-writable home/config directory
# in sandboxed/restricted environments.
os.environ.setdefault("YOLO_CONFIG_DIR", tempfile.gettempdir())

# Allow running directly as `python scripts/train_detector.py` without manual PYTHONPATH setup
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ultralytics import YOLO  # noqa: E402

from utils.config import load_project_configs  # noqa: E402
from utils.constants import DET_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for detector training and benchmark."""
    parser = argparse.ArgumentParser(
        description="Train or benchmark YOLO11 on BDD100K dataset."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/bdd_yolo.yaml",
        help="Path to dataset YAML config (default: configs/bdd_yolo.yaml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture or pretrained weights (e.g. yolo11s.pt, yolo11n.pt).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size (overrides config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (e.g. '0', 'cpu', 'cuda').",
    )
    parser.add_argument(
        "--output-weights",
        type=Path,
        default=None,
        help="Path to save the best model weights (default from configs/paths.yaml).",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run validation benchmark on an existing weights file instead of training.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Checkpoint path to evaluate when --benchmark-only is set.",
    )
    return parser.parse_args()


def train_detector(
    data_yaml: str | Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    device: str | None,
    output_weights: Path,
    train_cfg: dict[str, Any],
) -> Path:
    """Execute training of YOLO11 detector and copy best checkpoint to output_weights."""
    logger.info("Initializing YOLO model: %s", model_name)
    model = YOLO(model_name)

    train_args: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": img_size,
        "optimizer": train_cfg.get("optimizer", "AdamW"),
        "lr0": float(train_cfg.get("lr0", 0.001)),
        "workers": int(train_cfg.get("workers", 4)),
        "amp": bool(train_cfg.get("amp", True)),
        "close_mosaic": int(train_cfg.get("close_mosaic", 10)),
        "fliplr": float(train_cfg.get("fliplr", 0.5)),
        "flipud": float(train_cfg.get("flipud", 0.0)),
        "project": "runs/detect",
        "name": "yolo11_bdd",
        "exist_ok": True,
    }
    if device is not None:
        train_args["device"] = device

    logger.info("Starting training with arguments:\n%s", train_args)
    results = model.train(**train_args)

    # Resolve best weights path from training run directory
    save_dir = (
        Path(results.save_dir)
        if hasattr(results, "save_dir")
        else Path("runs/detect/yolo11_bdd")
    )
    best_pt = save_dir / "weights" / "best.pt"

    output_weights.parent.mkdir(parents=True, exist_ok=True)
    if best_pt.exists():
        shutil.copy2(best_pt, output_weights)
        logger.info("Best weights successfully saved to: %s", output_weights)
    else:
        logger.warning("Could not find best.pt at %s; check save_dir: %s", best_pt, save_dir)

    return output_weights


def evaluate_benchmark(
    weights_path: str | Path,
    data_yaml: str | Path,
    img_size: int,
    device: str | None,
) -> dict[str, float]:
    """Run validation benchmark and print standardized evaluation report."""
    logger.info("Evaluating detector weights '%s' on %s...", weights_path, data_yaml)
    model = YOLO(str(weights_path))

    val_args: dict[str, Any] = {
        "data": str(data_yaml),
        "imgsz": img_size,
        "batch": 16,
        "device": device if device is not None else "cpu",
        "verbose": False,
    }
    metrics = model.val(**val_args)

    report = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "Precision": float(metrics.box.mp),
        "Recall": float(metrics.box.mr),
    }

    logger.info("=== YOLO Benchmark Evaluation Summary ===")
    logger.info("mAP@0.5:      %.4f", report["mAP50"])
    logger.info("mAP@0.5:0.95: %.4f", report["mAP50_95"])
    logger.info("Precision:    %.4f", report["Precision"])
    logger.info("Recall:       %.4f", report["Recall"])

    if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
        logger.info("--- Class-specific AP50 ---")
        for i, c_name in enumerate(DET_CLASSES):
            if i < len(metrics.box.maps):
                ap = float(metrics.box.maps[i])
                report[f"AP50_{c_name}"] = ap
                logger.info("  %s: %.4f", c_name.ljust(15), ap)

    return report


def main() -> None:
    """CLI entry point for YOLO11 training and benchmarking."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    cfg = load_project_configs()

    det_cfg = cfg.detector
    model_name = args.model or det_cfg.model.architecture
    epochs = args.epochs if args.epochs is not None else int(det_cfg.train.epochs)
    batch_size = args.batch_size if args.batch_size is not None else int(det_cfg.train.batch_size)
    img_size = args.imgsz if args.imgsz is not None else int(det_cfg.model.img_size)
    device = args.device or cfg.project.device
    output_weights = args.output_weights or Path(cfg.paths.detector_weights)

    if args.benchmark_only:
        weights_to_eval = args.weights or str(output_weights)
        evaluate_benchmark(
            weights_path=weights_to_eval,
            data_yaml=args.data,
            img_size=img_size,
            device=device,
        )
        return

    train_detector(
        data_yaml=args.data,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        device=device,
        output_weights=output_weights,
        train_cfg=det_cfg.train.to_dict(),
    )


if __name__ == "__main__":
    main()
