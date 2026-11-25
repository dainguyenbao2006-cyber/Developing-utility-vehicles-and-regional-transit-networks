"""Pedestrian Collision Warning (PCW) purely using NumPy and Python stdlib."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from utils.geometry import bbox_bottom_center
from utils.types import ADASAlert, Detection

logger = logging.getLogger(__name__)

DEFAULT_PCW_CLASSES = ("pedestrian", "rider", "bicycle")


class PedestrianCollisionWarner:
    """Detects vulnerable road users (pedestrians, riders, cyclists) within ego lane."""

    def __init__(
        self,
        target_classes: tuple[str, ...] | list[str] = DEFAULT_PCW_CLASSES,
        danger_y_min_ratio: float = 0.50,
        in_lane_iou_threshold: float = 0.10,
    ) -> None:
        self.target_classes = set(target_classes)
        self.danger_y_min_ratio = float(danger_y_min_ratio)
        self.in_lane_iou_threshold = float(in_lane_iou_threshold)

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
        seg_mask: np.ndarray | None = None,
    ) -> list[ADASAlert]:
        """Check for pedestrians/riders in immediate ego path.

        Parameters
        ----------
        detections : list[Detection]
            Detected bounding boxes.
        frame_shape : tuple[int, int]
            (Height, Width) of the frame.
        seg_mask : np.ndarray | None
            Optional 3-channel segmentation mask (channel 0 is drivable_direct).

        Returns
        -------
        list[ADASAlert]
            Alerts generated for pedestrians at risk.
        """
        height, width = frame_shape
        danger_y_cutoff = height * self.danger_y_min_ratio
        alerts: list[ADASAlert] = []

        direct_mask = (seg_mask[0] > 0.5) if (seg_mask is not None and seg_mask.ndim == 3) else None

        for det in detections:
            if det.class_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = det.bbox_xyxy
            if y2 < danger_y_cutoff:
                # Far away, not an imminent collision hazard
                continue

            mid_x, bottom_y = bbox_bottom_center((x1, y1, x2, y2))
            px = max(0, min(width - 1, int(round(mid_x))))
            py = max(0, min(height - 1, int(round(bottom_y))))

            in_danger_zone = False
            if direct_mask is not None and direct_mask.shape == (height, width):
                # Contact point directly on drivable direct area
                if direct_mask[py, px]:
                    in_danger_zone = True
                else:
                    # Check if bounding box overlaps with drivable_direct
                    box_roi = direct_mask[int(y1):int(y2), int(x1):int(x2)]
                    if box_roi.size > 0 and box_roi.mean() >= self.in_lane_iou_threshold:
                        in_danger_zone = True
            else:
                # Geometric fallback: within center 40% horizontal corridor
                center_x = 0.5 * width
                if abs(mid_x - center_x) <= 0.20 * width:
                    in_danger_zone = True

            if in_danger_zone:
                vn_label = (
                    "NGƯỜI ĐI BỘ"
                    if det.class_name == "pedestrian"
                    else ("NGƯỜI LÁI XE" if det.class_name == "rider" else "XE ĐẠP")
                )
                alerts.append(
                    ADASAlert(
                        alert_type="PCW",
                        severity="CRITICAL",
                        message=f"CẢNH BÁO: {vn_label} TRONG LÀN ĐƯỜNG!",
                        target_bbox=(x1, y1, x2, y2),
                    )
                )

        return alerts

    @classmethod
    def from_config(cls, adas_cfg: dict[str, Any]) -> PedestrianCollisionWarner:
        """Create warner instance from config dictionary."""
        pcw_cfg = adas_cfg.get("pcw", adas_cfg)
        return cls(
            target_classes=pcw_cfg.get("target_classes", DEFAULT_PCW_CLASSES),
            danger_y_min_ratio=pcw_cfg.get("danger_y_min_ratio", 0.50),
            in_lane_iou_threshold=pcw_cfg.get("in_lane_iou_threshold", 0.10),
        )
