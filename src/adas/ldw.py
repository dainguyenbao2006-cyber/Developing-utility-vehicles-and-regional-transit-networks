"""Lane Departure Warning (LDW) logic purely using NumPy and Python stdlib."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from utils.types import ADASAlert

logger = logging.getLogger(__name__)


class LaneDepartureWarner:
    """Evaluates vehicle position relative to drivable area and markings with hysteresis."""

    def __init__(
        self,
        camera_offset_ratio: float = 0.0,
        ego_y_ratio: float = 0.88,
        wheel_strip_height_ratio: float = 0.03,
        wheel_half_width_ratio: float = 0.12,
        lane_overlap_threshold: float = 0.15,
        min_direct_coverage: float = 0.35,
        hysteresis_frames: int = 3,
    ) -> None:
        self.camera_offset_ratio = float(camera_offset_ratio)
        self.ego_y_ratio = float(ego_y_ratio)
        self.wheel_strip_height_ratio = float(wheel_strip_height_ratio)
        self.wheel_half_width_ratio = float(wheel_half_width_ratio)
        self.lane_overlap_threshold = float(lane_overlap_threshold)
        self.min_direct_coverage = float(min_direct_coverage)
        self.hysteresis_frames = max(1, int(hysteresis_frames))

        self._history: deque[str] = deque(maxlen=self.hysteresis_frames)
        self._current_status: str = "NORMAL"

    def reset(self) -> None:
        """Reset internal temporal hysteresis state."""
        self._history.clear()
        self._current_status = "NORMAL"

    def update(
        self,
        seg_mask: np.ndarray,
        frame_shape: tuple[int, int] | None = None,
    ) -> tuple[str, list[ADASAlert]]:
        """Evaluate lane departure state for current frame.

        Parameters
        ----------
        seg_mask : np.ndarray
            Binary or probability segmentation mask.
            Expected shape: (3, H, W) where:
                channel 0: drivable_direct
                channel 1: drivable_alternative
                channel 2: lane_marking
        frame_shape : tuple[int, int] | None
            Optional (H, W) of original frame if mask has different shape.

        Returns
        -------
        tuple[str, list[ADASAlert]]
            - ldw_status: "NORMAL" | "WARNING" | "DEPARTED"
            - alerts: List containing an ADASAlert if warning/departed, else empty.
        """
        if not isinstance(seg_mask, np.ndarray):
            raise TypeError(f"Expected seg_mask as np.ndarray, got {type(seg_mask).__name__}")
        if seg_mask.ndim != 3 or seg_mask.shape[0] != 3:
            raise ValueError(f"Expected seg_mask with shape (3, H, W), got {seg_mask.shape}")

        height, width = seg_mask.shape[1], seg_mask.shape[2]
        if height == 0 or width == 0:
            return "NORMAL", []

        direct_mask = seg_mask[0] > 0.5
        alt_mask = seg_mask[1] > 0.5
        lane_mask = seg_mask[2] > 0.5

        # 1. Compute ego reference strip coordinates
        ego_x = int(round(width * (0.5 + self.camera_offset_ratio)))
        ego_y = int(round(height * self.ego_y_ratio))
        strip_half_h = max(1, int(round(height * self.wheel_strip_height_ratio * 0.5)))
        wheel_half_w = max(1, int(round(width * self.wheel_half_width_ratio)))
        wheel_contact_w = max(1, int(round(width * 0.03)))

        y1 = max(0, ego_y - strip_half_h)
        y2 = min(height, ego_y + strip_half_h)

        # Center nose region (between wheels)
        center_x1 = max(0, ego_x - wheel_contact_w)
        center_x2 = min(width, ego_x + wheel_contact_w)

        # Left wheel contact zone
        left_w_x1 = max(0, ego_x - wheel_half_w - wheel_contact_w)
        left_w_x2 = min(width, ego_x - wheel_half_w + wheel_contact_w)

        # Right wheel contact zone
        right_w_x1 = max(0, ego_x + wheel_half_w - wheel_contact_w)
        right_w_x2 = min(width, ego_x + wheel_half_w + wheel_contact_w)

        # 2. Extract metrics in regions
        center_direct_cov = (
            direct_mask[y1:y2, center_x1:center_x2].mean()
            if (y2 > y1 and center_x2 > center_x1)
            else 0.0
        )
        center_alt_cov = (
            alt_mask[y1:y2, center_x1:center_x2].mean()
            if (y2 > y1 and center_x2 > center_x1)
            else 0.0
        )

        left_lane_ratio = (
            lane_mask[y1:y2, left_w_x1:left_w_x2].mean()
            if (y2 > y1 and left_w_x2 > left_w_x1)
            else 0.0
        )
        right_lane_ratio = (
            lane_mask[y1:y2, right_w_x1:right_w_x2].mean()
            if (y2 > y1 and right_w_x2 > right_w_x1)
            else 0.0
        )
        max_lane_overlap = max(left_lane_ratio, right_lane_ratio)

        # 3. Determine instantaneous frame status
        if center_direct_cov < self.min_direct_coverage and (
            center_alt_cov > 0.25 or max_lane_overlap > 0.35
        ):
            instant_status = "DEPARTED"
        elif max_lane_overlap >= self.lane_overlap_threshold:
            instant_status = "WARNING"
        else:
            instant_status = "NORMAL"

        # 4. Temporal hysteresis: require consecutive frames to confirm departure/warning
        self._history.append(instant_status)
        if len(self._history) >= self.hysteresis_frames:
            # If all recent frames match the new status, transition
            if all(s == instant_status for s in self._history):
                self._current_status = instant_status
            elif instant_status == "DEPARTED":
                # Escalate to WARNING if not yet fully confirmed DEPARTED
                if self._current_status == "NORMAL":
                    self._current_status = "WARNING"
        else:
            self._current_status = instant_status

        # 5. Build alert
        alerts: list[ADASAlert] = []
        if self._current_status == "DEPARTED":
            side_msg = "SANG TRÁI" if left_lane_ratio > right_lane_ratio else "SANG PHẢI"
            alerts.append(
                ADASAlert(
                    alert_type="LDW",
                    severity="CRITICAL",
                    message=f"CẢNH BÁO: LỆCH LÀN ĐƯỜNG ({side_msg})",
                )
            )
        elif self._current_status == "WARNING":
            side_msg = "vạch trái" if left_lane_ratio >= right_lane_ratio else "vạch phải"
            alerts.append(
                ADASAlert(
                    alert_type="LDW",
                    severity="WARNING",
                    message=f"Chú ý: Đang đè {side_msg}",
                )
            )

        return self._current_status, alerts

    @classmethod
    def from_config(cls, adas_cfg: dict[str, Any]) -> LaneDepartureWarner:
        """Create warner instance directly from adas config dictionary."""
        ldw_cfg = adas_cfg.get("ldw", adas_cfg)
        return cls(
            camera_offset_ratio=ldw_cfg.get("camera_offset_ratio", 0.0),
            ego_y_ratio=ldw_cfg.get("ego_y_ratio", 0.88),
            wheel_strip_height_ratio=ldw_cfg.get("wheel_strip_height_ratio", 0.03),
            wheel_half_width_ratio=ldw_cfg.get("wheel_half_width_ratio", 0.12),
            lane_overlap_threshold=ldw_cfg.get("lane_overlap_threshold", 0.15),
            min_direct_coverage=ldw_cfg.get("min_direct_coverage", 0.35),
            hysteresis_frames=ldw_cfg.get("hysteresis_frames", 3),
        )
