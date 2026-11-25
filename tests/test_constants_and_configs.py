"""Unit tests for Milestone 1: constants, types, YAML config loading, and geometry."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import ConfigNode, load_project_configs, load_yaml, validate_configs
from src.utils.constants import (
    BDD_DET_CATEGORY_MAP,
    DET_CLASS_TO_ID,
    DET_CLASSES,
    SEG_CHANNEL_TO_ID,
    SEG_CHANNELS,
)
from src.utils.geometry import (
    bbox_area,
    bbox_bottom_center,
    bbox_iou,
    clip_xyxy,
    create_trapezoid_mask,
    xyxy_to_yolo,
    yolo_to_xyxy,
)
from src.utils.types import ADASAlert, Detection, FrameAnalysis


def test_det_classes_order_and_mapping() -> None:
    expected = (
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "rider",
        "traffic_light",
        "traffic_sign",
    )
    assert DET_CLASSES == expected
    assert len(DET_CLASSES) == 9
    for idx, name in enumerate(DET_CLASSES):
        assert DET_CLASS_TO_ID[name] == idx
    assert BDD_DET_CATEGORY_MAP["motor"] == "motorcycle"
    assert BDD_DET_CATEGORY_MAP["person"] == "pedestrian"
    assert BDD_DET_CATEGORY_MAP["traffic sign"] == "traffic_sign"


def test_seg_channels_order_and_mapping() -> None:
    expected = ("drivable_direct", "drivable_alternative", "lane_marking")
    assert SEG_CHANNELS == expected
    assert len(SEG_CHANNELS) == 3
    for idx, name in enumerate(SEG_CHANNELS):
        assert SEG_CHANNEL_TO_ID[name] == idx


def test_load_all_project_configs() -> None:
    cfg = load_project_configs()
    assert cfg.project.seed == 42
    assert cfg.paths.bdd_root
    assert cfg["paths"]["detector_weights"] == "weights/detector_best.pt"
    assert cfg.detector.model.architecture == "yolo11s.pt"
    assert cfg.segmenter.model.architecture == "DeepLabV3Plus"
    assert list(cfg.segmenter.model.input_size_hw) == [384, 640]
    assert cfg.adas.ldw.hysteresis_frames == 3
    assert isinstance(cfg.to_dict(), dict)


def test_bdd_yolo_yaml_matches_det_classes() -> None:
    bdd_yolo = load_yaml("configs/bdd_yolo.yaml")
    names_map = bdd_yolo.to_dict()["names"]
    loaded = tuple(names_map[i] for i in range(len(DET_CLASSES)))
    assert loaded == DET_CLASSES


def test_env_override_applies_to_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BDD_ROOT", "/tmp/custom_bdd")
    monkeypatch.setenv("DEVICE", "cpu")
    cfg = load_project_configs()
    assert cfg.paths.bdd_root == "/tmp/custom_bdd"
    assert cfg.project.device == "cpu"


def test_validate_configs_rejects_channel_mismatch() -> None:
    cfg_dict = load_project_configs().to_dict()
    cfg_dict["segmenter"]["model"]["thresholds"] = [0.5, 0.5]
    bad_cfg = ConfigNode(cfg_dict)
    with pytest.raises(ValueError, match="thresholds"):
        validate_configs(bad_cfg)


def test_load_yaml_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_yaml(Path("configs/does_not_exist.yaml"))


def test_dataclasses_instantiation() -> None:
    det = Detection(
        bbox_xyxy=(10.0, 20.0, 110.0, 220.0),
        confidence=0.9,
        class_id=0,
        class_name="car",
    )
    alert = ADASAlert(
        alert_type="FCW",
        severity="CRITICAL",
        message="Brake",
        target_bbox=det.bbox_xyxy,
    )
    analysis = FrameAnalysis(frame_idx=1, timestamp_sec=0.033, detections=[det], alerts=[alert])
    assert analysis.ldw_status == "NORMAL"
    assert analysis.fcw_status == "SAFE"
    assert analysis.detections[0].class_name == "car"


def test_geometry_bbox_helpers() -> None:
    clipped = clip_xyxy((-10.0, -5.0, 700.0, 500.0), width=640, height=384)
    assert clipped == (0.0, 0.0, 640.0, 384.0)

    box = (100.0, 50.0, 300.0, 250.0)
    assert bbox_area(box) == pytest.approx(40000.0)
    assert bbox_bottom_center(box) == pytest.approx((200.0, 250.0))
    assert bbox_iou(box, box) == pytest.approx(1.0)
    assert bbox_iou(box, (400.0, 300.0, 500.0, 350.0)) == pytest.approx(0.0)

    yolo = xyxy_to_yolo(box, img_width=640, img_height=480)
    restored = yolo_to_xyxy(yolo, img_width=640, img_height=480)
    assert restored == pytest.approx(box, abs=1e-4)


def test_create_trapezoid_mask() -> None:
    mask = create_trapezoid_mask(
        height=100,
        width=200,
        top_y_ratio=0.5,
        top_width_ratio=0.2,
        bottom_width_ratio=0.6,
    )
    assert mask.shape == (100, 200)
    assert mask.dtype == bool
    assert not mask[20, 100]  # above top_y
    assert mask[99, 100]      # bottom center inside
    assert not mask[99, 0]    # bottom far-left outside
