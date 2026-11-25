"""End-to-end ADAS inference pipeline streaming video, webcam, or image folders."""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from adas.alert_manager import ADASAnalyzer
from models.detector import YOLO11Detector
from models.segmenter import DeepLabV3PlusSegmenter
from utils.config import ConfigNode
from utils.constants import SEG_CHANNELS
from utils.types import Detection, FrameAnalysis
from visualization.hud import EventLogger, HUDVisualizer

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class DetectorProtocol(Protocol):
    """Duck-typed interface for object detectors used in ADASPipeline."""

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]: ...


class SegmenterProtocol(Protocol):
    """Duck-typed interface for semantic segmenters used in ADASPipeline."""

    def predict(self, frame_bgr: np.ndarray, return_prob: bool = False) -> np.ndarray: ...


@dataclass
class PipelineRunSummary:
    """Summary statistics collected over a pipeline execution run."""

    frames_processed: int
    elapsed_sec: float
    avg_fps: float
    output_video_path: Path | None = None
    log_csv_path: Path | None = None
    class_counts: dict[str, int] = field(default_factory=dict)
    alert_counts: dict[str, int] = field(default_factory=dict)
    fps_history: list[float] = field(default_factory=list)


FrameCallback = Callable[[np.ndarray, FrameAnalysis, float], None]


def _iter_image_frames(source_path: Path) -> Iterator[np.ndarray]:
    """Yield BGR frames from a single image file or a directory of images."""
    if source_path.is_file():
        files = [source_path]
    else:
        files = [
            p
            for p in sorted(source_path.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

    for img_path in files:
        frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame is not None:
            yield frame
        else:
            logger.warning("Skipping unreadable image file: %s", img_path)


class ADASPipeline:
    """Coordinates Detector, Segmenter, ADASAnalyzer, HUDVisualizer, and EventLogger."""

    def __init__(
        self,
        detector: DetectorProtocol | None,
        segmenter: SegmenterProtocol | None,
        analyzer: ADASAnalyzer,
        visualizer: HUDVisualizer,
        event_logger: EventLogger | None = None,
    ) -> None:
        self.detector = detector
        self.segmenter = segmenter
        self.analyzer = analyzer
        self.visualizer = visualizer
        self.event_logger = event_logger
        self._ema_fps: float | None = None

    @classmethod
    def from_configs(
        cls,
        cfg: ConfigNode,
        log_csv_path: str | Path | None = None,
        require_weights: bool = True,
    ) -> ADASPipeline:
        """Instantiate full ADASPipeline from loaded project ConfigNode."""
        device = str(cfg.project.device)
        det_weights = Path(cfg.paths.detector_weights)
        seg_weights = Path(cfg.paths.segmenter_weights)

        detector: DetectorProtocol | None = None
        if det_weights.exists():
            det_model_cfg = cfg.detector.model
            detector = YOLO11Detector(
                weights_path=det_weights,
                conf_threshold=float(det_model_cfg.conf_threshold),
                iou_threshold=float(det_model_cfg.iou_threshold),
                img_size=int(det_model_cfg.img_size),
                device=device,
            )
        elif require_weights:
            raise FileNotFoundError(
                f"Detector weights not found at '{det_weights}'. "
                "Train detector first or pass require_weights=False for dry/demo mode."
            )
        else:
            logger.warning(
                "Detector weights '%s' not found; running without detection.", det_weights
            )

        segmenter: SegmenterProtocol | None = None
        if seg_weights.exists():
            seg_model_cfg = cfg.segmenter.model
            segmenter = DeepLabV3PlusSegmenter(
                weights_path=seg_weights,
                size_hw=tuple(seg_model_cfg.input_size_hw),
                thresholds=list(seg_model_cfg.thresholds),
                device=device,
            )
        elif require_weights:
            raise FileNotFoundError(
                f"Segmenter weights not found at '{seg_weights}'. "
                "Train segmenter first or pass require_weights=False for dry/demo mode."
            )
        else:
            logger.warning(
                "Segmenter weights '%s' not found; running without segmentation.", seg_weights
            )

        adas_dict = cfg.adas.to_dict()
        analyzer = ADASAnalyzer.from_config(adas_dict)
        visualizer = HUDVisualizer.from_config(adas_dict)
        event_logger = EventLogger(log_csv_path) if log_csv_path is not None else None

        return cls(
            detector=detector,
            segmenter=segmenter,
            analyzer=analyzer,
            visualizer=visualizer,
            event_logger=event_logger,
        )

    def reset(self) -> None:
        """Reset temporal states across ADAS rules and FPS smoothing."""
        self.analyzer.reset()
        self._ema_fps = None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
    ) -> tuple[np.ndarray, FrameAnalysis, float]:
        """Run end-to-end detection, segmentation, ADAS evaluation, and HUD rendering."""
        t0 = time.perf_counter()
        height, width = frame_bgr.shape[:2]

        detections: list[Detection] = (
            self.detector.predict(frame_bgr) if self.detector is not None else []
        )
        if self.segmenter is not None:
            seg_mask = self.segmenter.predict(frame_bgr, return_prob=False)
        else:
            seg_mask = np.zeros((len(SEG_CHANNELS), height, width), dtype=np.uint8)

        analysis = self.analyzer.analyze_frame(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            detections=detections,
            seg_mask=seg_mask,
            frame_shape=(height, width),
        )

        if self.event_logger is not None:
            self.event_logger.log(analysis)

        dt = max(1e-6, time.perf_counter() - t0)
        inst_fps = 1.0 / dt
        if self._ema_fps is None:
            self._ema_fps = inst_fps
        else:
            self._ema_fps = 0.8 * self._ema_fps + 0.2 * inst_fps

        rendered_bgr = self.visualizer.draw_overlay(
            frame_bgr=frame_bgr,
            detections=detections,
            seg_mask=seg_mask,
            analysis=analysis,
            fps=self._ema_fps,
        )
        return rendered_bgr, analysis, float(self._ema_fps)

    def run_source(
        self,
        source: str | Path | int,
        output_video_path: str | Path | None = None,
        max_frames: int | None = None,
        show_display: bool = False,
        default_fps: float = 25.0,
        frame_callback: FrameCallback | None = None,
    ) -> PipelineRunSummary:
        """Stream frames from webcam, video file, or image path/folder and write outputs."""
        self.reset()

        is_webcam = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
        source_fps = default_fps
        cap: cv2.VideoCapture | None = None
        image_iter: Iterator[np.ndarray] | None = None

        if is_webcam:
            cam_idx = int(source)
            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open webcam index: {cam_idx}")
            cap_fps = cap.get(cv2.CAP_PROP_FPS)
            if cap_fps and cap_fps > 1.0:
                source_fps = float(cap_fps)
        else:
            src_path = Path(source)
            if not src_path.exists():
                raise FileNotFoundError(f"Input source not found: {src_path}")

            if src_path.is_dir() or src_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_iter = _iter_image_frames(src_path)
            else:
                cap = cv2.VideoCapture(str(src_path))
                if not cap.isOpened():
                    raise RuntimeError(f"Failed to open video file: {src_path}")
                cap_fps = cap.get(cv2.CAP_PROP_FPS)
                if cap_fps and cap_fps > 1.0:
                    source_fps = float(cap_fps)

        writer: cv2.VideoWriter | None = None
        out_path = Path(output_video_path) if output_video_path is not None else None

        class_counts: Counter[str] = Counter()
        alert_counts: Counter[str] = Counter()
        fps_history: list[float] = []
        frame_idx = 0
        run_start = time.perf_counter()

        try:
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break

                if cap is not None:
                    ok, frame_bgr = cap.read()
                    if not ok or frame_bgr is None:
                        break
                elif image_iter is not None:
                    try:
                        frame_bgr = next(image_iter)
                    except StopIteration:
                        break
                else:
                    break

                timestamp_sec = frame_idx / max(1.0, source_fps)
                rendered_bgr, analysis, cur_fps = self.process_frame(
                    frame_bgr=frame_bgr,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                )

                fps_history.append(cur_fps)
                for det in analysis.detections:
                    class_counts[det.class_name] += 1
                for alert in analysis.alerts:
                    alert_counts[alert.alert_type] += 1

                if out_path is not None:
                    if writer is None:
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        h, w = rendered_bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(out_path), fourcc, source_fps, (w, h))
                    writer.write(rendered_bgr)

                if frame_callback is not None:
                    frame_callback(rendered_bgr, analysis, cur_fps)

                if show_display:
                    cv2.imshow("ADAS-Vision HUD", rendered_bgr)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break

                frame_idx += 1
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            if show_display:
                cv2.destroyAllWindows()

        elapsed = max(1e-6, time.perf_counter() - run_start)
        avg_fps = frame_idx / elapsed if frame_idx > 0 else 0.0
        log_path = self.event_logger.csv_path if self.event_logger is not None else None

        logger.info(
            "Processed %d frames in %.2fs (avg %.2f FPS).",
            frame_idx,
            elapsed,
            avg_fps,
        )
        return PipelineRunSummary(
            frames_processed=frame_idx,
            elapsed_sec=elapsed,
            avg_fps=avg_fps,
            output_video_path=out_path,
            log_csv_path=log_path,
            class_counts=dict(class_counts),
            alert_counts=dict(alert_counts),
            fps_history=fps_history,
        )
