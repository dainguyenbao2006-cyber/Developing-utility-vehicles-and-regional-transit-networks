"""CLI script for evaluating Detection, BDD100K/Cityscapes Segmentation, and End-to-End FPS."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from adas.alert_manager import ADASAnalyzer  # noqa: E402
from data.bdd_dataset import BDDSegmentationDataset, resolve_bdd_seg_dirs  # noqa: E402
from data.cityscapes_dataset import CityscapesRoadDataset  # noqa: E402
from data.transforms import get_seg_val_transforms  # noqa: E402
from evaluation import (  # noqa: E402
    benchmark_end_to_end_latency,
    evaluate_segmenter_bdd_loader,
    evaluate_segmenter_cityscapes_loader,
    save_evaluation_outputs,
)
from models.detector import YOLO11Detector  # noqa: E402
from models.segmenter import DeepLabV3PlusSegmenter  # noqa: E402
from utils.config import load_project_configs  # noqa: E402
from visualization.hud import HUDVisualizer  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for unified project evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate ADAS-Vision models on BDD100K, Cityscapes, and 720p latency."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "detection", "segmentation", "cityscapes", "latency"],
        default="all",
        help="Evaluation task to run (default: all).",
    )
    parser.add_argument(
        "--yolo-data",
        type=Path,
        default=Path("configs/bdd_yolo.yaml"),
        help="Ultralytics dataset YAML for detection validation.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Validation batch size.")
    parser.add_argument("--device", type=str, default=None, help="Device override ('cuda'/'cpu').")
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=20,
        help="Number of 720p frames for end-to-end latency benchmark.",
    )
    parser.add_argument(
        "--allow-missing-weights",
        action="store_true",
        help="Allow latency benchmark to run even when model .pt checkpoints are absent.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/eval_report.json"),
        help="Output path for JSON evaluation report.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/evaluation_report.md"),
        help="Output path for Markdown evaluation report.",
    )
    return parser.parse_args()


def main() -> None:
    """Run requested evaluation suites and save JSON + Markdown reports."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()
    cfg = load_project_configs()

    req_device = args.device or str(cfg.project.device)
    resolved_device = (
        req_device if (req_device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    torch_device = torch.device(resolved_device)

    det_weights = Path(cfg.paths.detector_weights)
    seg_weights = Path(cfg.paths.segmenter_weights)
    bdd_root = Path(cfg.paths.bdd_root)
    cs_root = Path(cfg.paths.cityscapes_root)
    seg_size_hw = tuple(cfg.segmenter.model.input_size_hw)
    seg_thresholds = list(cfg.segmenter.model.thresholds)

    report: dict[str, Any] = {"device": resolved_device}

    detector: YOLO11Detector | None = None
    if det_weights.exists() and args.mode in {"all", "detection", "latency"}:
        detector = YOLO11Detector(
            weights_path=det_weights,
            conf_threshold=float(cfg.detector.model.conf_threshold),
            iou_threshold=float(cfg.detector.model.iou_threshold),
            img_size=int(cfg.detector.model.img_size),
            device=resolved_device,
        )

    segmenter: DeepLabV3PlusSegmenter | None = None
    if seg_weights.exists() and args.mode in {"all", "segmentation", "cityscapes", "latency"}:
        segmenter = DeepLabV3PlusSegmenter(
            weights_path=seg_weights,
            size_hw=seg_size_hw,
            thresholds=seg_thresholds,
            device=resolved_device,
        )

    # 1. Detection evaluation
    if args.mode in {"all", "detection"}:
        if detector is not None and args.yolo_data.exists():
            report["detection"] = detector.benchmark_val(
                data_yaml=args.yolo_data,
                batch_size=args.batch_size,
            )
        else:
            logger.warning(
                "Skipping detection eval (weights exists=%s, yolo_data exists=%s).",
                det_weights.exists(),
                args.yolo_data.exists(),
            )

    # 2. BDD100K Segmentation evaluation
    if args.mode in {"all", "segmentation"}:
        val_imgs, drv_val, lane_val = resolve_bdd_seg_dirs(bdd_root, "val")
        val_split_file = Path(cfg.paths.splits_dir) / "val.txt"
        if segmenter is not None and val_imgs.exists():
            bdd_ds = BDDSegmentationDataset(
                images_dir=val_imgs,
                drivable_dir=drv_val,
                lane_dir=lane_val,
                size_hw=segmenter.size_hw,
                transform=get_seg_val_transforms(segmenter.size_hw),
                split_file=val_split_file if val_split_file.exists() else None,
            )
            bdd_loader = DataLoader(bdd_ds, batch_size=args.batch_size, shuffle=False)
            report["segmentation_bdd"] = evaluate_segmenter_bdd_loader(
                model=segmenter.model,
                loader=bdd_loader,
                device=torch_device,
                thresholds=segmenter.thresholds,
            )
        else:
            logger.warning(
                "Skipping BDD100K seg eval (segmenter loaded=%s, val_imgs exists=%s).",
                segmenter is not None,
                val_imgs.exists(),
            )

    # 3. Cityscapes cross-dataset road union evaluation
    if args.mode in {"all", "cityscapes"}:
        if segmenter is not None and cs_root.exists():
            cs_ds = CityscapesRoadDataset(
                cityscapes_root=cs_root,
                split="val",
                size_hw=segmenter.size_hw,
                transform=get_seg_val_transforms(segmenter.size_hw),
            )
            if len(cs_ds) > 0:
                cs_loader = DataLoader(cs_ds, batch_size=args.batch_size, shuffle=False)
                report["segmentation_cityscapes"] = evaluate_segmenter_cityscapes_loader(
                    model=segmenter.model,
                    loader=cs_loader,
                    device=torch_device,
                    thresholds=segmenter.thresholds,
                )
        else:
            logger.warning(
                "Skipping Cityscapes eval (segmenter loaded=%s, cs_root exists=%s).",
                segmenter is not None,
                cs_root.exists(),
            )

    # 4. End-to-End 720p Latency & FPS benchmark
    if args.mode in {"all", "latency"}:
        if (detector is not None and segmenter is not None) or args.allow_missing_weights:
            adas_dict = cfg.adas.to_dict()
            analyzer = ADASAnalyzer.from_config(adas_dict)
            visualizer = HUDVisualizer.from_config(adas_dict)
            report["latency"] = benchmark_end_to_end_latency(
                detector=detector,
                segmenter=segmenter,
                analyzer=analyzer,
                visualizer=visualizer,
                frame_hw=(720, 1280),
                warmup_runs=2,
                benchmark_runs=args.benchmark_runs,
            )
            logger.info("Latency benchmark (720p): %s", report["latency"])
        else:
            logger.warning(
                "Skipping latency benchmark because weights are missing "
                "(pass --allow-missing-weights to benchmark ADAS + HUD overhead)."
            )

    save_evaluation_outputs(
        report=report,
        json_path=args.output_json,
        markdown_path=args.output_md,
    )


if __name__ == "__main__":
    main()
