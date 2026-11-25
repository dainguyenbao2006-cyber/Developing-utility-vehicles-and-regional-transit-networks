"""YOLO11 Object Detector wrapper adhering to project data contracts and class constraints."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

os.environ.setdefault("YOLO_CONFIG_DIR", tempfile.gettempdir())

from ultralytics import YOLO  # noqa: E402

from src.utils.constants import DET_CLASSES  # noqa: E402
from src.utils.types import Detection  # noqa: E402

logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a Tensor or array-like object to a NumPy array."""
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _normalize_names(raw_names: Any) -> tuple[str, ...] | None:
    """Normalize Ultralytics model.names (dict or sequence) into an ordered tuple."""
    if isinstance(raw_names, dict):
        if len(raw_names) != len(DET_CLASSES):
            return tuple(str(raw_names[k]) for k in sorted(raw_names))
        return tuple(str(raw_names[i]) for i in range(len(DET_CLASSES)) if i in raw_names)
    if isinstance(raw_names, (list, tuple)):
        return tuple(str(name) for name in raw_names)
    return None


class YOLO11Detector:
    """Wrapper for Ultralytics YOLO11 object detection with strict class validation."""

    def __init__(
        self,
        weights_path: str | Path,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: str | None = None,
        validate_classes: bool = True,
        model_instance: Any | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.img_size = int(img_size)
        self.device = device

        if model_instance is not None:
            self.model = model_instance
        else:
            if not self.weights_path.exists() and self.weights_path.suffix != ".pt":
                raise FileNotFoundError(f"Model weights file not found: {self.weights_path}")
            logger.info("Loading YOLO model from: %s", self.weights_path)
            self.model = YOLO(str(self.weights_path))

        if validate_classes:
            self._validate_model_classes()

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "configs/detector_yolo11.yaml",
        weights_path: str | Path = "weights/detector_best.pt",
        device: str | None = None,
        validate_classes: bool = True,
        model_instance: Any | None = None,
    ) -> YOLO11Detector:
        """Instantiate YOLO11Detector from a detector YAML configuration file."""
        cfg_file = Path(config_path)
        if not cfg_file.exists():
            raise FileNotFoundError(f"Detector config not found: {cfg_file}")

        with cfg_file.open("r", encoding="utf-8") as handle:
            raw_cfg = yaml.safe_load(handle) or {}

        model_cfg = raw_cfg.get("model", {})
        return cls(
            weights_path=weights_path,
            conf_threshold=float(model_cfg.get("conf_threshold", 0.35)),
            iou_threshold=float(model_cfg.get("iou_threshold", 0.45)),
            img_size=int(model_cfg.get("img_size", 640)),
            device=device,
            validate_classes=validate_classes,
            model_instance=model_instance,
        )

    def _validate_model_classes(self) -> None:
        """Verify that model.names strictly matches DET_CLASSES in order and content."""
        raw_names = getattr(self.model, "names", None)
        loaded_names = _normalize_names(raw_names)
        if loaded_names is None:
            raise ValueError("Loaded model does not have a valid 'names' mapping or sequence.")

        if loaded_names != DET_CLASSES:
            raise ValueError(
                f"Model classes mismatch: {loaded_names} != {DET_CLASSES}"
            )
        logger.debug("Model classes validated against DET_CLASSES: %s", loaded_names)

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run object detection on a single BGR uint8 frame (H, W, 3).

        Returns Detection objects in original frame xyxy pixel coordinates.
        """
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError(f"Expected frame_bgr as np.ndarray, got {type(frame_bgr).__name__}")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                f"Expected BGR frame with shape (H, W, 3), got shape {frame_bgr.shape}"
            )
        if frame_bgr.dtype != np.uint8:
            raise ValueError(f"Expected frame dtype uint8, got {frame_bgr.dtype}")

        predict_kwargs: dict[str, Any] = {
            "source": frame_bgr,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "imgsz": self.img_size,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        boxes_xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        cls_ids = _to_numpy(boxes.cls).astype(int)

        detections: list[Detection] = []
        for box, conf, cls_id in zip(boxes_xyxy, confs, cls_ids, strict=False):
            if 0 <= cls_id < len(DET_CLASSES):
                x1, y1, x2, y2 = (float(v) for v in box[:4])
                detections.append(
                    Detection(
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=DET_CLASSES[cls_id],
                    )
                )

        return detections

    def get_model_complexity(self, img_size: int | None = None) -> dict[str, float]:
        """Return parameter count (M) and GFLOPs when exposed by Ultralytics model.info()."""
        info_fn = getattr(self.model, "info", None)
        if not callable(info_fn):
            return {}

        target_imgsz = int(img_size or self.img_size)
        try:
            info = info_fn(verbose=True, imgsz=target_imgsz)
        except TypeError:
            info = info_fn(verbose=True)

        if isinstance(info, tuple) and len(info) >= 4:
            # Ultralytics YOLO.info(verbose=True) returns (layers, params, gradients, gflops)
            params = float(info[1])
            flops = float(info[3])
            return {
                "params_m": params / 1e6,
                "gflops": flops,
            }
        return {}

    def benchmark_val(
        self,
        data_yaml: str | Path,
        img_size: int | None = None,
        batch_size: int = 16,
    ) -> dict[str, float]:
        """Evaluate detector on a validation split and return standardized metrics."""
        imgsz = int(img_size or self.img_size)
        logger.info("Running validation benchmark on data=%s (imgsz=%d)", data_yaml, imgsz)

        val_kwargs: dict[str, Any] = {
            "data": str(data_yaml),
            "imgsz": imgsz,
            "batch": int(batch_size),
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "verbose": False,
        }
        if self.device is not None:
            val_kwargs["device"] = self.device

        metrics = self.model.val(**val_kwargs)
        results: dict[str, float] = {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "mp": float(metrics.box.mp),
            "mr": float(metrics.box.mr),
        }

        per_class_ap50 = getattr(metrics.box, "ap50", None)
        if per_class_ap50 is None:
            per_class_ap50 = getattr(metrics.box, "maps", None)

        if per_class_ap50 is not None:
            for idx, class_name in enumerate(DET_CLASSES):
                if idx < len(per_class_ap50):
                    results[f"ap50_{class_name}"] = float(per_class_ap50[idx])

        results.update(self.get_model_complexity())
        return results

    def measure_latency(
        self,
        img_size: int | None = None,
        warmup: int = 10,
        iters: int = 50,
    ) -> float:
        """Measure average single-frame inference latency in milliseconds."""
        if iters <= 0:
            raise ValueError(f"iters must be positive, got {iters}")
        if warmup < 0:
            raise ValueError(f"warmup must be non-negative, got {warmup}")

        side = int(img_size or self.img_size)
        dummy_frame = np.zeros((side, side, 3), dtype=np.uint8)

        for _ in range(warmup):
            self.predict(dummy_frame)

        start = time.perf_counter()
        for _ in range(iters):
            self.predict(dummy_frame)
        elapsed_sec = time.perf_counter() - start

        avg_ms = (elapsed_sec / iters) * 1000.0
        logger.info("Measured average latency: %.2f ms/frame (imgsz=%d)", avg_ms, side)
        return avg_ms
