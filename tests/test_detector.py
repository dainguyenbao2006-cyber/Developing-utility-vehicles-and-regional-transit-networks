"""Unit tests for Milestone 3: YOLO11 detector wrapper and class contract verification."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from models.detector import YOLO11Detector
from utils.constants import DET_CLASSES
from utils.types import Detection


def make_mock_yolo_model(class_names: dict[int, str] | None = None) -> MagicMock:
    """Create a mock YOLO model instance with configured class names."""
    if class_names is None:
        class_names = {idx: name for idx, name in enumerate(DET_CLASSES)}
    mock_model = MagicMock()
    mock_model.names = class_names
    return mock_model


def test_detector_validates_classes_successfully() -> None:
    mock_model = make_mock_yolo_model()
    detector = YOLO11Detector(
        weights_path="dummy_weights.pt",
        model_instance=mock_model,
        validate_classes=True,
    )
    assert detector.model is mock_model


def test_detector_raises_on_class_mismatch() -> None:
    bad_names = {0: "car", 1: "airplane"}  # Mismatched classes
    mock_model = make_mock_yolo_model(class_names=bad_names)
    with pytest.raises(ValueError, match="Model classes mismatch"):
        YOLO11Detector(
            weights_path="dummy_weights.pt",
            model_instance=mock_model,
            validate_classes=True,
        )


def test_detector_raises_on_invalid_names_attribute() -> None:
    mock_model = MagicMock()
    mock_model.names = "not_a_dict"
    with pytest.raises(ValueError, match="valid 'names' attribute"):
        YOLO11Detector(
            weights_path="dummy_weights.pt",
            model_instance=mock_model,
            validate_classes=True,
        )


def test_detector_input_validation() -> None:
    mock_model = make_mock_yolo_model()
    detector = YOLO11Detector(
        weights_path="dummy_weights.pt",
        model_instance=mock_model,
    )

    with pytest.raises(TypeError, match="Expected frame_bgr as np.ndarray"):
        detector.predict("not_an_array")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Expected BGR frame with shape"):
        detector.predict(np.zeros((100, 100), dtype=np.uint8))

    with pytest.raises(ValueError, match="Expected frame dtype uint8"):
        detector.predict(np.zeros((100, 100, 3), dtype=np.float32))


def test_detector_predict_returns_detections() -> None:
    mock_model = make_mock_yolo_model()

    # Mock Ultralytics result structure
    mock_boxes = MagicMock()
    mock_boxes.xyxy = torch.tensor([[10.0, 20.0, 110.0, 120.0], [50.0, 60.0, 80.0, 90.0]])
    mock_boxes.conf = torch.tensor([0.92, 0.75])
    mock_boxes.cls = torch.tensor([0.0, 7.0])  # 0: car, 7: traffic_light
    mock_boxes.__len__.return_value = 2

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_model.predict.return_value = [mock_result]

    detector = YOLO11Detector(
        weights_path="dummy_weights.pt",
        conf_threshold=0.35,
        iou_threshold=0.45,
        img_size=640,
        model_instance=mock_model,
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.predict(frame)

    assert len(detections) == 2
    assert isinstance(detections[0], Detection)
    assert detections[0].class_id == 0
    assert detections[0].class_name == "car"
    assert detections[0].confidence == pytest.approx(0.92, abs=1e-3)
    assert detections[0].bbox_xyxy == (10.0, 20.0, 110.0, 120.0)

    assert detections[1].class_id == 7
    assert detections[1].class_name == "traffic_light"
    assert detections[1].confidence == pytest.approx(0.75, abs=1e-3)


def test_detector_predict_empty_results() -> None:
    mock_model = make_mock_yolo_model()
    mock_result = MagicMock()
    mock_result.boxes = None
    mock_model.predict.return_value = [mock_result]

    detector = YOLO11Detector(
        weights_path="dummy_weights.pt",
        model_instance=mock_model,
    )

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.predict(frame) == []


def test_detector_benchmark_val() -> None:
    mock_model = make_mock_yolo_model()
    mock_metrics = MagicMock()
    mock_metrics.box.map50 = 0.55
    mock_metrics.box.map = 0.35
    mock_metrics.box.mp = 0.60
    mock_metrics.box.mr = 0.50
    mock_metrics.box.maps = np.array([0.75, 0.45, 0.50, 0.35, 0.30, 0.40, 0.25, 0.30, 0.28])

    mock_model.val.return_value = mock_metrics

    detector = YOLO11Detector(
        weights_path="dummy_weights.pt",
        model_instance=mock_model,
    )
    results = detector.benchmark_val("configs/bdd_yolo.yaml")

    assert results["map50"] == pytest.approx(0.55)
    assert results["map50_95"] == pytest.approx(0.35)
    assert results["ap50_car"] == pytest.approx(0.75)
