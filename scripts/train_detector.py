"""Train and benchmark YOLO11 object detection models on BDD100K (TV3)."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("YOLO_CONFIG_DIR", tempfile.gettempdir())

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.models.detector import YOLO11Detector  # noqa: E402
from src.utils.constants import DET_CLASS_TO_ID, DET_CLASSES  # noqa: E402

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def load_detector_config(
    config_path: str | Path = "configs/detector_yolo11.yaml",
) -> dict[str, Any]:
    """Load detector YAML configuration dictionary."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Detector config not found: {cfg_file}")

    with cfg_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in detector config: {cfg_file}")
    return data


def validate_bdd_yolo_yaml(data_yaml: str | Path) -> dict[str, Any]:
    """Validate that a YOLO dataset YAML matches DET_CLASSES in exact order."""
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as handle:
        data_cfg = yaml.safe_load(handle)
    if not isinstance(data_cfg, dict):
        raise ValueError(f"Invalid dataset YAML structure in: {yaml_path}")

    names = data_cfg.get("names")
    if isinstance(names, dict):
        loaded = tuple(names.get(i) for i in range(len(DET_CLASSES)))
    elif isinstance(names, list):
        loaded = tuple(names)
    else:
        raise ValueError(f"Missing or invalid 'names' mapping in {yaml_path}")

    if loaded != DET_CLASSES:
        raise ValueError(f"Dataset YAML classes mismatch: {loaded} != {DET_CLASSES}")

    return data_cfg


def count_yolo_label_instances(labels_dir: str | Path) -> dict[str, int]:
    """Count object instances per canonical class from YOLO .txt label files."""
    root = Path(labels_dir)
    if not root.exists():
        raise FileNotFoundError(f"Labels directory not found: {root}")

    counts: dict[str, int] = {name: 0 for name in DET_CLASSES}
    for label_file in sorted(root.rglob("*.txt")):
        with label_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                cls_id = int(line.split()[0])
                if 0 <= cls_id < len(DET_CLASSES):
                    counts[DET_CLASSES[cls_id]] += 1
    return counts


def _resolve_image_for_label(label_file: Path, images_root: Path, labels_root: Path) -> Path | None:
    """Resolve the corresponding image path for a YOLO label file."""
    rel_stem = label_file.relative_to(labels_root).with_suffix("")
    for ext in _IMAGE_EXTENSIONS:
        candidate = images_root / f"{rel_stem}{ext}"
        if candidate.exists():
            return candidate.resolve()
    return None


def build_oversampled_train_yaml(
    data_yaml: str | Path,
    rare_classes: list[str],
    oversample_factor: int = 2,
    output_dir: str | Path = "outputs/oversampling",
    seed: int = 42,
) -> Path:
    """Create a derived dataset YAML oversampling images that contain rare classes."""
    if oversample_factor < 1:
        raise ValueError(f"oversample_factor must be >= 1, got {oversample_factor}")

    data_cfg = validate_bdd_yolo_yaml(data_yaml)
    dataset_root = Path(data_cfg["path"])
    train_rel = Path(data_cfg.get("train", "images/train"))
    images_root = dataset_root / train_rel
    if train_rel.parts[0] == "images":
        labels_rel = Path("labels") / train_rel.relative_to("images")
    else:
        labels_rel = Path("labels/train")
    labels_root = dataset_root / labels_rel

    if not images_root.exists() or not labels_root.exists():
        raise FileNotFoundError(
            f"Expected train images ({images_root}) and labels ({labels_root}) to exist."
        )

    rare_ids = {DET_CLASS_TO_ID[name] for name in rare_classes if name in DET_CLASS_TO_ID}
    image_entries: list[str] = []

    for label_file in sorted(labels_root.rglob("*.txt")):
        img_path = _resolve_image_for_label(label_file, images_root, labels_root)
        if img_path is None:
            continue

        with label_file.open("r", encoding="utf-8") as handle:
            cls_ids = {
                int(line.split()[0])
                for line in handle
                if line.strip()
            }

        repeats = oversample_factor if (cls_ids & rare_ids) else 1
        for _ in range(repeats):
            image_entries.append(str(img_path))

    if not image_entries:
        raise ValueError(f"No matched training images/labels found under {dataset_root}")

    rng = random.Random(seed)
    rng.shuffle(image_entries)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "train_oversampled.txt"
    manifest_path.write_text("\n".join(image_entries) + "\n", encoding="utf-8")

    derived_cfg = dict(data_cfg)
    derived_cfg["train"] = str(manifest_path.resolve())
    derived_yaml = out_dir / "bdd_yolo_oversampled.yaml"
    with derived_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(derived_cfg, handle, sort_keys=False)

    logger.info(
        "Created oversampled train manifest (%d entries, factor=%d) at %s",
        len(image_entries),
        oversample_factor,
        derived_yaml,
    )
    return derived_yaml


def train_detector(
    data_yaml: str | Path,
    model_name: str,
    epochs: int,
    batch_size: int,
    img_size: int,
    device: str | None,
    output_weights: Path,
    train_cfg: dict[str, Any],
    run_name: str | None = None,
) -> Path:
    """Train YOLO11 on BDD100K and copy best.pt to output_weights."""
    validate_bdd_yolo_yaml(data_yaml)

    logger.info("Initializing YOLO model: %s", model_name)
    model = YOLO(model_name)

    resolved_name = run_name or str(train_cfg.get("name", "yolo11_bdd"))
    train_args: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(epochs),
        "batch": int(batch_size),
        "imgsz": int(img_size),
        "optimizer": str(train_cfg.get("optimizer", "AdamW")),
        "lr0": float(train_cfg.get("lr0", 0.001)),
        "workers": int(train_cfg.get("workers", 4)),
        "amp": bool(train_cfg.get("amp", True)),
        "cos_lr": bool(train_cfg.get("cos_lr", True)),
        "warmup_epochs": int(train_cfg.get("warmup_epochs", 3)),
        "mosaic": float(train_cfg.get("mosaic", 1.0)),
        "close_mosaic": int(train_cfg.get("close_mosaic", 10)),
        "fliplr": float(train_cfg.get("fliplr", 0.5)),
        "flipud": float(train_cfg.get("flipud", 0.0)),
        "copy_paste": float(train_cfg.get("copy_paste", 0.0)),
        "seed": int(train_cfg.get("seed", 42)),
        "project": str(train_cfg.get("project", "runs/detect")),
        "name": resolved_name,
        "exist_ok": True,
    }
    if device is not None:
        train_args["device"] = device

    logger.info("Starting YOLO training with args: %s", train_args)
    results = model.train(**train_args)

    save_dir = (
        Path(results.save_dir)
        if hasattr(results, "save_dir")
        else Path(train_args["project"]) / resolved_name
    )
    best_pt = save_dir / "weights" / "best.pt"

    output_weights.parent.mkdir(parents=True, exist_ok=True)
    if best_pt.exists():
        shutil.copy2(best_pt, output_weights)
        logger.info("Saved best checkpoint to %s", output_weights)
    else:
        logger.warning("Could not locate best.pt under %s", save_dir)

    return output_weights


def evaluate_single_model(
    weights_path: str | Path,
    data_yaml: str | Path,
    img_size: int = 640,
    batch_size: int = 16,
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    device: str | None = None,
    latency_warmup: int = 10,
    latency_iters: int = 50,
    validate_classes: bool = True,
) -> dict[str, Any]:
    """Evaluate one detector checkpoint on val set and measure single-frame latency."""
    validate_bdd_yolo_yaml(data_yaml)

    detector = YOLO11Detector(
        weights_path=weights_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        img_size=img_size,
        device=device,
        validate_classes=validate_classes,
    )
    metrics = detector.benchmark_val(data_yaml=data_yaml, img_size=img_size, batch_size=batch_size)
    latency_ms = detector.measure_latency(
        img_size=img_size,
        warmup=latency_warmup,
        iters=latency_iters,
    )

    row: dict[str, Any] = {
        "model": Path(weights_path).name,
        "img_size": int(img_size),
        "params_m": metrics.get("params_m", float("nan")),
        "gflops": metrics.get("gflops", float("nan")),
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "ap50_car": metrics.get("ap50_car", float("nan")),
        "ap50_motorcycle": metrics.get("ap50_motorcycle", float("nan")),
        "ap50_pedestrian": metrics.get("ap50_pedestrian", float("nan")),
        "ap50_traffic_sign": metrics.get("ap50_traffic_sign", float("nan")),
        "latency_ms": latency_ms,
    }
    return row


def format_benchmark_markdown(rows: list[dict[str, Any]]) -> str:
    """Format benchmark comparison rows as a Markdown table."""
    header = (
        "| Model | Img Size | Params (M) | GFLOPs | mAP@0.5 | mAP@0.5:0.95 | "
        "AP50 (`car`) | AP50 (`motorcycle`) | AP50 (`pedestrian`) | "
        "AP50 (`traffic_sign`) | Latency (ms/frame) |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['img_size']} | {row['params_m']:.2f} | "
            f"{row['gflops']:.2f} | {row['map50']:.4f} | {row['map50_95']:.4f} | "
            f"{row['ap50_car']:.4f} | {row['ap50_motorcycle']:.4f} | "
            f"{row['ap50_pedestrian']:.4f} | {row['ap50_traffic_sign']:.4f} | "
            f"{row['latency_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def save_benchmark_outputs(rows: list[dict[str, Any]], markdown_path: Path) -> None:
    """Write benchmark results to both Markdown and CSV files."""
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(format_benchmark_markdown(rows), encoding="utf-8")

    csv_path = markdown_path.with_suffix(".csv")
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    logger.info("Saved benchmark reports to %s and %s", markdown_path, csv_path)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for YOLO11 training and benchmark comparison."""
    parser = argparse.ArgumentParser(description="Train and benchmark YOLO11 on BDD100K.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/detector_yolo11.yaml"),
        help="Path to detector YAML config.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/bdd_yolo.yaml"),
        help="Path to YOLO dataset YAML config.",
    )
    parser.add_argument("--model", type=str, default=None, help="Pretrained/architecture model.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--imgsz", type=int, default=None, help="Override input image size.")
    parser.add_argument("--device", type=str, default=None, help="Device ('0', 'cuda', 'cpu').")
    parser.add_argument("--run-name", type=str, default=None, help="Training run folder name.")
    parser.add_argument(
        "--output-weights",
        type=Path,
        default=Path("weights/detector_best.pt"),
        help="Destination path for best checkpoint.",
    )
    parser.add_argument(
        "--oversample-rare",
        action="store_true",
        help="Oversample images containing rare classes (motorcycle, bicycle, traffic_sign).",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Evaluate trained weights without running training.",
    )
    parser.add_argument(
        "--compare-models",
        nargs="+",
        default=None,
        help="List of model checkpoints/weights to compare in benchmark mode.",
    )
    parser.add_argument(
        "--benchmark-out",
        type=Path,
        default=Path("outputs/detector_benchmark.md"),
        help="Output Markdown path for benchmark table (CSV written alongside).",
    )
    parser.add_argument(
        "--skip-class-check",
        action="store_true",
        help="Skip DET_CLASSES validation (only for untrained COCO baseline models).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for YOLO11 training and benchmarking."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    cfg = load_detector_config(args.config)

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    bench_cfg = cfg.get("benchmark", {})

    img_size = int(args.imgsz if args.imgsz is not None else model_cfg.get("img_size", 640))
    batch_size = int(
        args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 16)
    )
    conf_thresh = float(model_cfg.get("conf_threshold", 0.35))
    iou_thresh = float(model_cfg.get("iou_threshold", 0.45))

    if args.benchmark_only:
        models_to_eval = args.compare_models or [str(args.output_weights)]
        rows: list[dict[str, Any]] = []
        for weight_item in models_to_eval:
            row = evaluate_single_model(
                weights_path=weight_item,
                data_yaml=args.data,
                img_size=img_size,
                batch_size=int(bench_cfg.get("batch_size", batch_size)),
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
                device=args.device,
                latency_warmup=int(bench_cfg.get("latency_warmup", 10)),
                latency_iters=int(bench_cfg.get("latency_iters", 50)),
                validate_classes=not args.skip_class_check,
            )
            rows.append(row)

        save_benchmark_outputs(rows, args.benchmark_out)
        print(format_benchmark_markdown(rows))
        return

    data_yaml = args.data
    if args.oversample_rare:
        default_rare = ["motorcycle", "bicycle", "traffic_sign"]
        rare_classes = list(bench_cfg.get("rare_classes", default_rare))
        data_yaml = build_oversampled_train_yaml(
            data_yaml=args.data,
            rare_classes=rare_classes,
            oversample_factor=int(bench_cfg.get("rare_oversample_factor", 2)),
            seed=int(train_cfg.get("seed", 42)),
        )

    model_name = args.model or str(model_cfg.get("architecture", "yolo11s.pt"))
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 50))

    train_detector(
        data_yaml=data_yaml,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
        device=args.device,
        output_weights=args.output_weights,
        train_cfg=train_cfg,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
