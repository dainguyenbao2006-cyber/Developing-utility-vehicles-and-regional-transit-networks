"""Command-line interface for running end-to-end ADAS video/image/webcam inference."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from pipeline import ADASPipeline, PipelineRunSummary
from utils.config import DEFAULT_CONFIG_DIR, load_project_configs

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser supporting both `cli run --source ...` and `cli --source ...`."""
    parser = argparse.ArgumentParser(
        prog="adas-cli",
        description="Run ADAS-Vision end-to-end object detection, segmentation, and HUD overlay.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["run"],
        default="run",
        help="Optional 'run' subcommand for compatibility.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input video file (.mp4/.avi), image file/folder, or webcam index (e.g. '0').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/demo_out.mp4"),
        help="Output video file path (default: outputs/demo_out.mp4).",
    )
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=Path("outputs/events.csv"),
        help="CSV path for ADAS alert event logging (default: outputs/events.csv).",
    )
    parser.add_argument(
        "--config-paths",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "paths.yaml",
        help="Path to paths YAML config (default: configs/paths.yaml).",
    )
    parser.add_argument(
        "--config-detector",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "detector_yolo11.yaml",
        help="Path to detector YAML config (default: configs/detector_yolo11.yaml).",
    )
    parser.add_argument(
        "--config-segmenter",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "segmenter_deeplab.yaml",
        help="Path to segmenter YAML config (default: configs/segmenter_deeplab.yaml).",
    )
    parser.add_argument(
        "--config-adas",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "adas_rules.yaml",
        help="Path to ADAS rules YAML config (default: configs/adas_rules.yaml).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional compute device override ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless without opening an OpenCV preview window.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process.",
    )
    parser.add_argument(
        "--allow-missing-weights",
        action="store_true",
        help="Allow running HUD/ADAS dry-run when weight checkpoints are not yet present.",
    )
    return parser


def run_cli(argv: Sequence[str] | None = None) -> PipelineRunSummary:
    """Parse CLI arguments, build ADASPipeline, and execute inference."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_project_configs(
        paths_cfg=args.config_paths,
        detector_cfg=args.config_detector,
        segmenter_cfg=args.config_segmenter,
        adas_cfg=args.config_adas,
    )
    if args.device is not None:
        cfg_dict = cfg.to_dict()
        cfg_dict["project"]["device"] = args.device
        from utils.config import ConfigNode

        cfg = ConfigNode(cfg_dict)

    pipeline = ADASPipeline.from_configs(
        cfg=cfg,
        log_csv_path=args.log_csv,
        require_weights=not args.allow_missing_weights,
    )
    summary = pipeline.run_source(
        source=args.source,
        output_video_path=args.output,
        max_frames=args.max_frames,
        show_display=not args.no_display,
    )
    logger.info(
        "Completed run: %d frames | Avg FPS: %.2f | Video: %s | Events CSV: %s",
        summary.frames_processed,
        summary.avg_fps,
        summary.output_video_path,
        summary.log_csv_path,
    )
    return summary


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_cli()


if __name__ == "__main__":
    main()
