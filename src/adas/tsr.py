"""Traffic Sign and Traffic Light Awareness (TSR) purely using NumPy and stdlib."""

from __future__ import annotations

import logging
from typing import Any

from utils.geometry import bbox_area
from utils.types import ADASAlert, Detection

logger = logging.getLogger(__name__)

DEFAULT_TSR_CLASSES = ("traffic_light", "traffic_sign")


class TrafficSignRecognizer:
    """Monitors detected traffic signs and traffic lights in ego field-of-view."""

    def __init__(
        self,
        target_classes: tuple[str, ...] | list[str] = DEFAULT_TSR_CLASSES,
        min_confidence: float = 0.45,
        min_box_area_ratio: float = 0.0004,
    ) -> None:
        self.target_classes = set(target_classes)
        self.min_confidence = float(min_confidence)
        self.min_box_area_ratio = float(min_box_area_ratio)

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> list[ADASAlert]:
        """Detect and announce traffic signals and signs.

        Parameters
        ----------
        detections : list[Detection]
            Detected bounding boxes.
        frame_shape : tuple[int, int]
            (Height, Width) of the frame.

        Returns
        -------
        list[ADASAlert]
            Informative alerts for detected signs and lights.
        """
        height, width = frame_shape
        frame_area = float(max(1, height * width))
        alerts: list[ADASAlert] = []

        for det in detections:
            if det.class_name not in self.target_classes:
                continue
            if det.confidence < self.min_confidence:
                continue

            area_ratio = bbox_area(det.bbox_xyxy) / frame_area
            if area_ratio < self.min_box_area_ratio:
                # Too small / distant to alert
                continue

            if det.class_name == "traffic_light":
                alerts.append(
                    ADASAlert(
                        alert_type="TSR",
                        severity="INFO",
                        message="Tín hiệu: Đèn giao thông phía trước",
                        target_bbox=det.bbox_xyxy,
                    )
                )
            elif det.class_name == "traffic_sign":
                alerts.append(
                    ADASAlert(
                        alert_type="TSR",
                        severity="INFO",
                        message="Biển báo: Có biển báo giao thông",
                        target_bbox=det.bbox_xyxy,
                    )
                )

        return alerts

    @classmethod
    def from_config(cls, adas_cfg: dict[str, Any]) -> TrafficSignRecognizer:
        """Create recognizer instance from config dictionary."""
        tsr_cfg = adas_cfg.get("tsr", adas_cfg)
        return cls(
            target_classes=tsr_cfg.get("target_classes", DEFAULT_TSR_CLASSES),
            min_confidence=tsr_cfg.get("min_confidence", 0.45),
            min_box_area_ratio=tsr_cfg.get("min_box_area_ratio", 0.0004),
        )
