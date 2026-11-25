"""Unit tests for TV3 YOLO11 detector wrapper, configs, and training helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_detector import (  # noqa: E402
    build_oversampled_train_yaml,
    count_yolo_label_instances,
    format_benchmark_markdown,
    load_detector_config,
    validate_bdd_yolo_yaml,
)
from src.models.detector import YOLO11Detector  # noqa: E402
from src.utils.constants import DET_CLASSES  # noqa: E402


class _MockBoxes:
    """Minimal mock mimicking ultralytics.engine.results.Boxes."""

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.xyxy)


class _MockYOLO:
    """Lightweight mock replacing Ultralytics YOLO in unit tests."""

    def __init__(
        self,
        names: dict[int, str],
        detections: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        self.names = names
        self._detections = detections
        self.last_predict_kwargs: dict[str, object] = {}

    def predict(self, **kwargs: object) -> list[SimpleNamespace]:
        self.last_predict_kwargs = kwargs
        if self._detections is None:
            return [SimpleNamespace(boxes=None)]
        xyxy, conf, cls = self._detections
        return [SimpleNamespace(boxes=_MockBoxes(xyxy=xyxy, conf=conf, cls=cls))]

    def val(self, **kwargs: object) -> SimpleNamespace:
        ap50 = np.linspace(0.5, 0.9, len(DET_CLASSES))
        box = SimpleNamespace(map50=0.72, map=0.48, mp=0.75, mr=0.68, ap50=ap50)
        return SimpleNamespace(box=box)

    def info(self, verbose: bool = True, imgsz: int = 640) -> tuple[int, int, int, float] | None:
        if not verbose:
            return None
        return (120, 9_400_000, 0, 21.5)


def test_detector_configs_are_valid_and_match_det_classes() -> None:
    det_cfg = load_detector_config("configs/detector_yolo11.yaml")
    assert det_cfg["model"]["architecture"] == "yolo11s.pt"
    assert det_cfg["model"]["img_size"] == 640
    assert det_cfg["train"]["seed"] == 42

    bdd_cfg = validate_bdd_yolo_yaml("configs/bdd_yolo.yaml")
    loaded = tuple(bdd_cfg["names"][i] for i in range(len(DET_CLASSES)))
    assert loaded == DET_CLASSES


def test_detector_rejects_mismatched_classes() -> None:
    bad_names = {i: name for i, name in enumerate(reversed(DET_CLASSES))}
    with pytest.raises(ValueError, match="Model classes mismatch"):
        YOLO11Detector(
            weights_path="dummy.pt",
            model_instance=_MockYOLO(names=bad_names),
        )

    incomplete_names = {i: DET_CLASSES[i] for i in range(len(DET_CLASSES) - 1)}
    with pytest.raises(ValueError, match="Model classes mismatch"):
        YOLO11Detector(
            weights_path="dummy.pt",
            model_instance=_MockYOLO(names=incomplete_names),
        )


def test_detector_predict_returns_detections_and_keeps_input_immutable() -> None:
    valid_names = {i: name for i, name in enumerate(DET_CLASSES)}
    xyxy = np.array([[10.0, 20.0, 110.0, 140.0], [200.0, 50.0, 240.0, 180.0]], dtype=np.float32)
    conf = np.array([0.91, 0.64], dtype=np.float32)
    cls_ids = np.array([0, 5], dtype=np.float32)

    mock_model = _MockYOLO(names=valid_names, detections=(xyxy, conf, cls_ids))
    detector = YOLO11Detector.from_config(
        config_path="configs/detector_yolo11.yaml",
        weights_path="dummy.pt",
        model_instance=mock_model,
    )

    frame = np.full((384, 640, 3), 127, dtype=np.uint8)
    frame_copy = frame.copy()

    detections = detector.predict(frame)
    assert np.array_equal(frame, frame_copy)
    assert len(detections) == 2

    assert detections[0].class_id == 0
    assert detections[0].class_name == "car"
    assert detections[0].bbox_xyxy == pytest.approx((10.0, 20.0, 110.0, 140.0))
    assert detections[0].confidence == pytest.approx(0.91)

    assert detections[1].class_id == 5
    assert detections[1].class_name == "pedestrian"


def test_detector_predict_validates_frame_and_handles_empty() -> None:
    valid_names = {i: name for i, name in enumerate(DET_CLASSES)}
    detector = YOLO11Detector(
        weights_path="dummy.pt",
        model_instance=_MockYOLO(names=valid_names, detections=None),
    )

    empty_frame = np.zeros((128, 128, 3), dtype=np.uint8)
    assert detector.predict(empty_frame) == []

    with pytest.raises(ValueError, match="dtype uint8"):
        detector.predict(np.zeros((128, 128, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="shape"):
        detector.predict(np.zeros((128, 128), dtype=np.uint8))


def test_detector_benchmark_val_and_latency() -> None:
    valid_names = {i: name for i, name in enumerate(DET_CLASSES)}
    detector = YOLO11Detector(
        weights_path="dummy.pt",
        img_size=64,
        model_instance=_MockYOLO(names=valid_names, detections=None),
    )
    metrics = detector.benchmark_val("configs/bdd_yolo.yaml")
    assert metrics["map50"] == pytest.approx(0.72)
    assert metrics["map50_95"] == pytest.approx(0.48)
    assert metrics["params_m"] == pytest.approx(9.4)
    assert metrics["gflops"] == pytest.approx(21.5)
    assert "ap50_motorcycle" in metrics
    assert "ap50_traffic_sign" in metrics

    latency_ms = detector.measure_latency(img_size=64, warmup=1, iters=3)
    assert latency_ms >= 0.0


def test_oversample_rare_classes_and_benchmark_markdown(tmp_path: Path) -> None:
    ds_root = tmp_path / "bdd100k_yolo"
    (ds_root / "images" / "train").mkdir(parents=True)
    (ds_root / "labels" / "train").mkdir(parents=True)

    # Image 1: only car (class 0)
    (ds_root / "images" / "train" / "img1.jpg").write_bytes(b"fake")
    (ds_root / "labels" / "train" / "img1.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    # Image 2: contains motorcycle (class 3) and traffic_sign (class 8)
    (ds_root / "images" / "train" / "img2.jpg").write_bytes(b"fake")
    (ds_root / "labels" / "train" / "img2.txt").write_text(
        "3 0.4 0.6 0.1 0.1\n8 0.8 0.2 0.05 0.05\n", encoding="utf-8"
    )

    counts = count_yolo_label_instances(ds_root / "labels" / "train")
    assert counts["car"] == 1
    assert counts["motorcycle"] == 1
    assert counts["traffic_sign"] == 1

    data_yaml = tmp_path / "bdd.yaml"
    with data_yaml.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "path": str(ds_root),
                "train": "images/train",
                "val": "images/val",
                "names": {i: name for i, name in enumerate(DET_CLASSES)},
            },
            handle,
            sort_keys=False,
        )

    derived_yaml = build_oversampled_train_yaml(
        data_yaml=data_yaml,
        rare_classes=["motorcycle", "traffic_sign"],
        oversample_factor=3,
        output_dir=tmp_path / "out",
        seed=42,
    )
    with derived_yaml.open("r", encoding="utf-8") as handle:
        derived_cfg = yaml.safe_load(handle)

    manifest_lines = Path(derived_cfg["train"]).read_text(encoding="utf-8").strip().splitlines()
    # 1 copy of img1 + 3 copies of img2 = 4 entries
    assert len(manifest_lines) == 4
    assert sum("img2.jpg" in line for line in manifest_lines) == 3

    md = format_benchmark_markdown(
        [
            {
                "model": "yolo11s.pt",
                "img_size": 640,
                "params_m": 9.4,
                "gflops": 21.5,
                "map50": 0.65,
                "map50_95": 0.42,
                "ap50_car": 0.80,
                "ap50_motorcycle": 0.45,
                "ap50_pedestrian": 0.62,
                "ap50_traffic_sign": 0.51,
                "latency_ms": 12.34,
            }
        ]
    )
    assert "yolo11s.pt" in md
    assert "0.6500" in md
