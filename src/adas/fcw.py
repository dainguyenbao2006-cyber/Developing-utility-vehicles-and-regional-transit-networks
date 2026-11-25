"""Forward Collision Warning (FCW Proxy) purely using NumPy and Python stdlib."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from utils.geometry import bbox_area, bbox_bottom_center, bbox_iou
from utils.types import ADASAlert, Detection

logger = logging.getLogger(__name__)

DEFAULT_TARGET_CLASSES = ("car", "truck", "bus", "motorcycle")


class ForwardCollisionWarner:
    """Evaluates collision threat with lead vehicles inside the ego trajectory corridor."""

    def __init__(
        self,
        target_classes: tuple[str, ...] | list[str] = DEFAULT_TARGET_CLASSES,
        corridor_top_width_ratio: float = 0.18,
        corridor_bottom_width_ratio: float = 0.45,
        corridor_top_y_ratio: float = 0.45,
        caution_area_ratio: float = 0.04,
        critical_area_ratio: float = 0.10,
        growth_rate_critical: float = 0.25,
        history_window: int = 5,
        match_iou_threshold: float = 0.30,
    ) -> None:
        self.target_classes = set(target_classes)
        self.corridor_top_width_ratio = float(corridor_top_width_ratio)
        self.corridor_bottom_width_ratio = float(corridor_bottom_width_ratio)
        self.corridor_top_y_ratio = float(corridor_top_y_ratio)
        self.caution_area_ratio = float(caution_area_ratio)
        self.critical_area_ratio = float(critical_area_ratio)
        self.growth_rate_critical = float(growth_rate_critical)
        self.history_window = max(2, int(history_window))
        self.match_iou_threshold = float(match_iou_threshold)

        # History of lead vehicle bounding box and area ratio
        self._lead_history: deque[tuple[tuple[float, float, float, float], float]] = deque(
            maxlen=self.history_window
        )
        self._current_status: str = "SAFE"

    def reset(self) -> None:
        """Clear temporal tracking history."""
        self._lead_history.clear()
        self._current_status = "SAFE"

    def is_in_corridor(
        self,
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> bool:
        """Check if bottom-center contact point of bbox lies inside the trapezoid corridor."""
        mid_x, bottom_y = bbox_bottom_center(bbox)
        top_y = height * self.corridor_top_y_ratio
        if bottom_y < top_y or bottom_y > height:
            return False

        # Linear interpolation of corridor width from top_y down to bottom of image (height)
        progress = max(0.0, min(1.0, (bottom_y - top_y) / max(1.0, height - top_y)))
        current_width_ratio = (
            self.corridor_top_width_ratio
            + progress * (self.corridor_bottom_width_ratio - self.corridor_top_width_ratio)
        )
        half_w = 0.5 * width * current_width_ratio
        center_x = 0.5 * width

        return abs(mid_x - center_x) <= half_w

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> tuple[str, Detection | None, list[ADASAlert]]:
        """Evaluate forward collision threats.

        Parameters
        ----------
        detections : list[Detection]
            Detected bounding boxes in current frame.
        frame_shape : tuple[int, int]
            (Height, Width) of the frame.

        Returns
        -------
        tuple[str, Detection | None, list[ADASAlert]]
            - fcw_status: "SAFE" | "CAUTION" | "CRITICAL"
            - lead_vehicle: Detection of the lead vehicle (or None)
            - alerts: List containing an ADASAlert if warning/critical, else empty.
        """
        height, width = frame_shape
        frame_area = float(max(1, height * width))

        # 1. Filter target vehicle classes inside the ego corridor
        corridor_candidates: list[Detection] = []
        for det in detections:
            if det.class_name in self.target_classes:
                if self.is_in_corridor(det.bbox_xyxy, width=width, height=height):
                    corridor_candidates.append(det)

        if not corridor_candidates:
            self._lead_history.clear()
            self._current_status = "SAFE"
            return "SAFE", None, []

        # 2. Select lead vehicle: candidate with largest y2 (closest bottom edge)
        lead_vehicle = max(corridor_candidates, key=lambda d: d.bbox_xyxy[3])
        lead_box = lead_vehicle.bbox_xyxy
        current_area_ratio = bbox_area(lead_box) / frame_area

        # 3. Compute growth rate over history window
        growth_rate = 0.0
        if self._lead_history:
            prev_box, prev_ratio = self._lead_history[0]
            # Verify if it corresponds to the same vehicle (IoU > match_iou_threshold)
            if bbox_iou(lead_box, prev_box) >= self.match_iou_threshold:
                growth_rate = (current_area_ratio - prev_ratio) / max(1e-4, prev_ratio)

        self._lead_history.append((lead_box, current_area_ratio))

        # 4. Determine status
        is_critical = (
            current_area_ratio >= self.critical_area_ratio
            or (
                current_area_ratio >= self.caution_area_ratio
                and growth_rate >= self.growth_rate_critical
            )
        )
        is_caution = current_area_ratio >= self.caution_area_ratio

        if is_critical:
            self._current_status = "CRITICAL"
        elif is_caution:
            self._current_status = "CAUTION"
        else:
            self._current_status = "SAFE"

        # 5. Build alert
        alerts: list[ADASAlert] = []
        if self._current_status == "CRITICAL":
            alerts.append(
                ADASAlert(
                    alert_type="FCW",
                    severity="CRITICAL",
                    message="NGUY CƠ VA CHẠM PHÍA TRƯỚC! HÃY PHANH!",
                    target_bbox=lead_box,
                )
            )
        elif self._current_status == "CAUTION":
            alerts.append(
                ADASAlert(
                    alert_type="FCW",
                    severity="CAUTION",
                    message="Chú ý giữ khoảng cách an toàn với xe trước",
                    target_bbox=lead_box,
                )
            )

        return self._current_status, lead_vehicle, alerts

    @classmethod
    def from_config(cls, adas_cfg: dict[str, Any]) -> ForwardCollisionWarner:
        """Create warner instance directly from adas config dictionary."""
        fcw_cfg = adas_cfg.get("fcw", adas_cfg)
        return cls(
            target_classes=fcw_cfg.get("target_classes", DEFAULT_TARGET_CLASSES),
            corridor_top_width_ratio=fcw_cfg.get("corridor_top_width_ratio", 0.18),
            corridor_bottom_width_ratio=fcw_cfg.get("corridor_bottom_width_ratio", 0.45),
            corridor_top_y_ratio=fcw_cfg.get("corridor_top_y_ratio", 0.45),
            caution_area_ratio=fcw_cfg.get("caution_area_ratio", 0.04),
            critical_area_ratio=fcw_cfg.get("critical_area_ratio", 0.10),
            growth_rate_critical=fcw_cfg.get("growth_rate_critical", 0.25),
            history_window=fcw_cfg.get("history_window", 5),
            match_iou_threshold=fcw_cfg.get("match_iou_threshold", 0.30),
        )
