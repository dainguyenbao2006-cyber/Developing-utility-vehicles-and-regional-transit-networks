"""Unit and integration tests for Milestone 6: HUDVisualizer, EventLogger, Pipeline, and CLI."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from adas.alert_manager import ADASAnalyzer
from cli import build_parser, run_cli
from pipeline import ADASPipeline
from utils.config import load_project_configs
from utils.types import ADASAlert, Detection, FrameAnalysis
from visualization.hud import EventLogger, HUDVisualizer, strip_accents_for_cv2


class _StubDetector:
    """Deterministic stub detector for testing ADASPipeline without GPU weights."""

    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        return list(self._detections)


class _StubSegmenter:
    """Deterministic stub segmenter returning a centered drivable lane mask."""

    def predict(self, frame_bgr: np.ndarray, return_prob: bool = False) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        mask = np.zeros((3, h, w), dtype=np.float32 if return_prob else np.uint8)
        mask[0, h // 2 :, w // 4 : (3 * w) // 4] = 1
        mask[2, h // 2 :, w // 4 : w // 4 + 3] = 1
        return mask


def test_strip_accents_for_cv2() -> None:
    raw = "CẢNH BÁO: ĐANG ĐÈ VẠCH ĐƯỜNG!"
    cleaned = strip_accents_for_cv2(raw)
    assert cleaned == "CANH BAO: DANG DE VACH DUONG!"
    assert all(32 <= ord(ch) < 127 for ch in cleaned)


def test_hud_visualizer_does_not_mutate_inputs() -> None:
    frame = np.full((240, 360, 3), 80, dtype=np.uint8)
    frame_snapshot = frame.copy()

    seg_mask = np.zeros((3, 240, 360), dtype=np.uint8)
    seg_mask[0, 120:, 90:270] = 1
    seg_mask[1, 120:, :90] = 1
    seg_mask[2, 120:, 88:92] = 1
    mask_snapshot = seg_mask.copy()

    det = Detection(
        bbox_xyxy=(140.0, 130.0, 220.0, 215.0),
        confidence=0.91,
        class_id=0,
        class_name="car",
    )
    alert = ADASAlert(
        alert_type="FCW",
        severity="CRITICAL",
        message="NGUY CƠ VA CHẠM PHÍA TRƯỚC!",
        target_bbox=det.bbox_xyxy,
    )
    analysis = FrameAnalysis(
        frame_idx=4,
        timestamp_sec=0.16,
        detections=[det],
        ldw_status="NORMAL",
        fcw_status="CRITICAL",
        alerts=[alert],
    )

    viz = HUDVisualizer(mask_alpha=0.35)
    out = viz.draw_overlay(
        frame_bgr=frame,
        detections=[det],
        seg_mask=seg_mask,
        analysis=analysis,
        fps=24.5,
    )

    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not np.shares_memory(out, frame)
    assert np.array_equal(frame, frame_snapshot)
    assert np.array_equal(seg_mask, mask_snapshot)
    assert not np.array_equal(out, frame)


def test_hud_visualizer_resizes_mismatched_mask_and_validates_input() -> None:
    viz = HUDVisualizer()
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    small_mask = np.ones((3, 50, 80), dtype=np.uint8)

    out = viz.draw_overlay(frame_bgr=frame, detections=[], seg_mask=small_mask)
    assert out.shape == (200, 320, 3)

    with pytest.raises(TypeError, match="Expected frame_bgr as np.ndarray"):
        viz.draw_overlay("bad", detections=[], seg_mask=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Expected BGR frame with shape"):
        viz.draw_overlay(np.zeros((100, 100), dtype=np.uint8), detections=[], seg_mask=None)

    with pytest.raises(ValueError, match="Expected frame dtype uint8"):
        viz.draw_overlay(np.zeros((100, 100, 3), dtype=np.float32), detections=[], seg_mask=None)


def test_event_logger_writes_csv_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "logs" / "events.csv"
    ev_logger = EventLogger(csv_path)

    # Frame with no alerts should not add data rows
    ev_logger.log(FrameAnalysis(frame_idx=0, timestamp_sec=0.0, alerts=[]))

    # Frame with two alerts
    a1 = ADASAlert(
        alert_type="PCW",
        severity="CRITICAL",
        message="Pedestrian in lane",
        target_bbox=(100.0, 120.0, 140.0, 200.0),
    )
    a2 = ADASAlert(alert_type="LDW", severity="WARNING", message="Touching left line")
    ev_logger.log(FrameAnalysis(frame_idx=5, timestamp_sec=0.2, alerts=[a1, a2]))

    assert len(ev_logger.records) == 2
    with csv_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["alert_type"] == "PCW"
    assert rows[0]["severity"] == "CRITICAL"
    assert "bbox=" in rows[0]["extra_info"]
    assert rows[1]["alert_type"] == "LDW"


def test_pipeline_end_to_end_on_synthetic_video(tmp_path: Path) -> None:
    in_video = tmp_path / "input.mp4"
    out_video = tmp_path / "out" / "overlay.mp4"
    out_csv = tmp_path / "out" / "events.csv"

    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(in_video), fourcc, 20.0, (width, height))
    for i in range(4):
        synthetic_frame = np.full((height, width, 3), 40 + i * 10, dtype=np.uint8)
        writer.write(synthetic_frame)
    writer.release()

    close_car = Detection(
        bbox_xyxy=(110.0, 120.0, 210.0, 230.0),
        confidence=0.95,
        class_id=0,
        class_name="car",
    )
    cfg = load_project_configs()
    pipeline = ADASPipeline(
        detector=_StubDetector([close_car]),
        segmenter=_StubSegmenter(),
        analyzer=ADASAnalyzer.from_config(cfg.adas.to_dict()),
        visualizer=HUDVisualizer.from_config(cfg.adas.to_dict()),
        event_logger=EventLogger(out_csv),
    )

    seen_frames: list[int] = []
    summary = pipeline.run_source(
        source=in_video,
        output_video_path=out_video,
        show_display=False,
        frame_callback=lambda _img, analysis, _fps: seen_frames.append(analysis.frame_idx),
    )

    assert summary.frames_processed == 4
    assert seen_frames == [0, 1, 2, 3]
    assert summary.avg_fps > 0.0
    assert summary.class_counts["car"] == 4
    assert summary.alert_counts["FCW"] == 4
    assert out_video.exists() and out_video.stat().st_size > 0
    assert out_csv.exists()


def test_cli_parser_and_dry_run_on_image_folder(tmp_path: Path) -> None:
    parser = build_parser()
    parsed = parser.parse_args(["run", "--source", "0", "--no-display", "--max-frames", "10"])
    assert parsed.source == "0"
    assert parsed.no_display is True
    assert parsed.max_frames == 10

    img_dir = tmp_path / "frames"
    img_dir.mkdir()
    for idx in range(2):
        cv2.imwrite(
            str(img_dir / f"f_{idx:02d}.jpg"),
            np.full((180, 320, 3), 100, dtype=np.uint8),
        )

    out_vid = tmp_path / "cli_out.mp4"
    out_csv = tmp_path / "cli_events.csv"
    summary = run_cli(
        [
            "run",
            "--source",
            str(img_dir),
            "--output",
            str(out_vid),
            "--log-csv",
            str(out_csv),
            "--no-display",
            "--allow-missing-weights",
        ]
    )
    assert summary.frames_processed == 2
    assert out_vid.exists()
    assert out_csv.exists()


def test_pipeline_from_configs_requires_weights_by_default(tmp_path: Path) -> None:
    cfg_dict = load_project_configs().to_dict()
    cfg_dict["paths"]["detector_weights"] = str(tmp_path / "missing_det.pt")
    cfg_dict["paths"]["segmenter_weights"] = str(tmp_path / "missing_seg.pt")
    from utils.config import ConfigNode

    with pytest.raises(FileNotFoundError, match="Detector weights not found"):
        ADASPipeline.from_configs(ConfigNode(cfg_dict), require_weights=True)
