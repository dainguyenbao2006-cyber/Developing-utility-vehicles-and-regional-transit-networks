from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LDWConfig:
    camera_offset_ratio: float = 0.0
    wheel_half_width_ratio: float = 0.14
    wheel_y_ratio: float = 0.90
    roi_radius_px: int = 12
    lane_overlap_thresh: float = 0.25
    drivable_min_ratio: float = 0.50
    frames_to_trigger: int = 3
    frames_to_clear: int = 5


class LaneDepartureWarning:
    """Bộ phát hiện lệch làn / đè vạch (LDW) thuần NumPy có temporal hysteresis."""

    def __init__(self, config: LDWConfig | None = None) -> None:
        self.cfg = config or LDWConfig()
        self._violation_streak: int = 0
        self._safe_streak: int = 0
        self._latched_state: str = "SAFE"  # "SAFE" | "LDW_LEFT" | "LDW_RIGHT"

    def reset(self) -> None:
        """Reset trạng thái theo thời gian (dùng khi đổi video mới hoặc test độc lập)."""
        self._violation_streak = 0
        self._safe_streak = 0
        self._latched_state = "SAFE"

    @staticmethod
    def _validate_mask(mask: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(mask)
        if arr.ndim != 2:
            raise ValueError(f"{name} phải là mask 2D (H, W), nhận được shape={arr.shape}")
        return arr
        #Thuật toán ước lượng vùng bánh xe & tỷ lệ chạm làn theo không gian (Spatial ROI Ratio)
    def _patch_positive_ratio(self, mask: np.ndarray, cx: int, cy: int) -> float:
        h, w = mask.shape                                              # Lấy chiều cao (h) và chiều rộng (w) của mask
        r = max(1, int(self.cfg.roi_radius_px))                        # Bán kính vùng ROI (tối thiểu 1 pixel)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)                    # Biên trái/phải của ô ROI, kẹp trong [0, w]
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)                    # Biên trên/dưới của ô ROI, kẹp trong [0, h]

        if x1 <= x0 or y1 <= y0:                                       # Nếu vùng cắt rỗng (nằm ngoài biên ảnh)
            return 0.0                                                 # Trả về tỷ lệ 0.0

        roi = mask[y0:y1, x0:x1]                                       # Cắt mảng con 2D kích thước quanh tâm (cx, cy)
        return float(np.mean(roi > 0))                                 # Đổi sang mask boolean (roi > 0) rồi tính trung bình (tỷ lệ 0.0 - 1.0)

    def _infer_raw_state(
        self,
        left_lane_ratio: float,
        right_lane_ratio: float,
        ego_direct_ratio: float,
        ego_alt_ratio: float,
        x_ego: int,
        width: int,
    ) -> str:
        thresh = self.cfg.lane_overlap_thresh
        left_hit = left_lane_ratio >= thresh
        right_hit = right_lane_ratio >= thresh

        # 1. Ưu tiên kiểm tra bánh xe đè lên vạch kẻ đường
        if left_hit and right_hit:
            return "LDW_LEFT" if left_lane_ratio >= right_lane_ratio else "LDW_RIGHT"
        if left_hit:
            return "LDW_LEFT"
        if right_hit:
            return "LDW_RIGHT"

        # 2. Kiểm tra vùng ego rời khỏi làn đường chính (drivable_direct)
        if ego_direct_ratio < self.cfg.drivable_min_ratio:
            if ego_alt_ratio > ego_direct_ratio:
                return "LDW_LEFT" if x_ego < width * 0.5 else "LDW_RIGHT"
            return "LDW_LEFT" if x_ego < width * 0.5 else "LDW_RIGHT"

        return "SAFE"
        #Xác định vị trí vệt bánh xe và đo 4 chỉ số không gian
    def update(
        self,
        drivable_direct: np.ndarray,
        drivable_alternative: np.ndarray,
        lane_marking: np.ndarray,
    ) -> dict[str, float | str | bool]:
        direct = self._validate_mask(drivable_direct, "drivable_direct")
        alt = self._validate_mask(drivable_alternative, "drivable_alternative")
        lane = self._validate_mask(lane_marking, "lane_marking")

        if not (direct.shape == alt.shape == lane.shape):
            raise ValueError(
                f"Kích thước các mask không khớp: direct={direct.shape}, "
                f"alt={alt.shape}, lane={lane.shape}"
            )

        h, w = lane.shape
        x_ego = int(np.clip((w * 0.5) + self.cfg.camera_offset_ratio * (w * 0.5), 0, w - 1))
        y_wheel = int(np.clip(h * self.cfg.wheel_y_ratio, 0, h - 1))
        dx_wheel = int(w * self.cfg.wheel_half_width_ratio)

        x_left = max(0, x_ego - dx_wheel)
        x_right = min(w - 1, x_ego + dx_wheel)

        left_lane_ratio = self._patch_positive_ratio(lane, x_left, y_wheel)
        right_lane_ratio = self._patch_positive_ratio(lane, x_right, y_wheel)
        ego_direct_ratio = self._patch_positive_ratio(direct, x_ego, y_wheel)
        ego_alt_ratio = self._patch_positive_ratio(alt, x_ego, y_wheel)

        raw_state = self._infer_raw_state(
            left_lane_ratio=left_lane_ratio,
            right_lane_ratio=right_lane_ratio,
            ego_direct_ratio=ego_direct_ratio,
            ego_alt_ratio=ego_alt_ratio,
            x_ego=x_ego,
            width=w,
        )

        # Thuật toán lọc trễ hai ngưỡng theo thời gian Temporal Hysteresis
        if raw_state != "SAFE":
            self._violation_streak += 1
            self._safe_streak = 0
            if self._violation_streak >= self.cfg.frames_to_trigger:
                self._latched_state = raw_state
        else:
            self._safe_streak += 1
            self._violation_streak = 0
            if self._safe_streak >= self.cfg.frames_to_clear:
                self._latched_state = "SAFE"

        return {
            "alert": self._latched_state != "SAFE",
            "state": self._latched_state,
            "raw_state": raw_state,
            "left_lane_ratio": left_lane_ratio,
            "right_lane_ratio": right_lane_ratio,
            "ego_direct_ratio": ego_direct_ratio,
            "ego_alt_ratio": ego_alt_ratio,
        }