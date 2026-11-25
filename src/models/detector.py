"""YOLO11 Object Detector wrapper adhering to project data contracts and class constraints."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# Prevent Ultralytics from warning about non-writable home/config directory
# in sandboxed/restricted environments.
os.environ.setdefault("YOLO_CONFIG_DIR", tempfile.gettempdir())

from ultralytics import YOLO  # noqa: E402

from utils.constants import DET_CLASSES  # noqa: E402
from utils.types import Detection  # noqa: E402

logger = logging.getLogger(__name__)


class YOLO11Detector:
    """Wrapper for Ultralytics YOLO11 object detection with class validation.

    Adheres to project data contracts, bounding box types, and canonical class order.
    """

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
        self.weights_path = Path(weights_path) if isinstance(weights_path, str) else weights_path
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.img_size = int(img_size)
        self.device = device

        if model_instance is not None:
            self.model = model_instance
        else:
            if not self.weights_path.exists() and not str(self.weights_path).endswith(".pt"):
                raise FileNotFoundError(f"Model weights file not found: {self.weights_path}")
            logger.info("Loading YOLO model from: %s", self.weights_path)
            self.model = YOLO(str(self.weights_path))

        if validate_classes:
            self._validate_model_classes()

    def _validate_model_classes(self) -> None:
        """Verify that the model's class names strictly match canonical DET_CLASSES order.

        Raises ValueError if classes do not match (CLAUDE.md rule #22).
        """
        if not hasattr(self.model, "names") or not isinstance(self.model.names, dict):
            raise ValueError("Loaded model does not have a valid 'names' attribute mapping.")

        loaded_names = tuple(self.model.names.get(i) for i in range(len(DET_CLASSES)))
        if loaded_names != DET_CLASSES:
            raise ValueError(
                f"Model classes mismatch! Loaded: {loaded_names} != "
                f"Expected canonical: {DET_CLASSES}"
            )
        logger.debug("Model classes successfully validated against DET_CLASSES: %s", loaded_names)

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Run object detection on a single BGR uint8 video frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Input image in OpenCV BGR format (H, W, 3) with uint8 dtype.

        Returns
        -------
        list[Detection]
            Detected bounding boxes in original pixel coordinates [x1, y1, x2, y2].
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
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)

        detections: list[Detection] = []
        for box, conf, cls_id in zip(boxes_xyxy, confs, cls_ids, strict=False):
            if 0 <= cls_id < len(DET_CLASSES):
                class_name = DET_CLASSES[cls_id]
                x1, y1, x2, y2 = (float(v) for v in box[:4])
                detections.append(
                    Detection(
                        bbox_xyxy=(x1, y1, x2, y2),
                        confidence=float(conf),
                        class_id=int(cls_id),
                        class_name=class_name,
                    )
                )

        return detections

    def benchmark_val(
        self,
        data_yaml: str | Path,
        img_size: int | None = None,
        batch_size: int = 16,
    ) -> dict[str, float]:
        """Evaluate detector on a dataset split and return standardized benchmark metrics."""
        imgsz = img_size or self.img_size
        logger.info("Running validation benchmark on data=%s with imgsz=%d...", data_yaml, imgsz)

        val_kwargs: dict[str, Any] = {
            "data": str(data_yaml),
            "imgsz": imgsz,
            "batch": batch_size,
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

        # Extract per-class AP50 if available
        if hasattr(metrics.box, "maps") and metrics.box.maps is not None:
            for i, class_name in enumerate(DET_CLASSES):
                if i < len(metrics.box.maps):
                    results[f"ap50_{class_name}"] = float(metrics.box.maps[i])

        return results
