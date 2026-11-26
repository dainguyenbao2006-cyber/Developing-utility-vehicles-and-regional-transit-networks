"""Lane Departure Warning (LDW) logic purely using NumPy and Python stdlib."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from utils.types import ADASAlert

logger = logging.getLogger(__name__)


class LaneDepartureWarner:
    """Đánh giá vị trí xe so với vùng đường đi được và vạch làn đường, có lọc trễ theo thời gian."""

    def __init__(
        self,
        camera_offset_ratio: float = 0.0,        # Độ lệch ngang của camera so với tâm ảnh (-1.0 đến 1.0)
        ego_y_ratio: float = 0.88,               # Tung độ y của dải tham chiếu bánh xe (tính từ trên xuống)
        wheel_strip_height_ratio: float = 0.03,  # Chiều cao dải tham chiếu ngang (tỷ lệ theo chiều cao ảnh)
        wheel_half_width_ratio: float = 0.12,    # Khoảng cách nửa bề rộng từ tâm xe tới mỗi vệt bánh xe
        lane_overlap_threshold: float = 0.15,    # Tỷ lệ chạm vạch tối thiểu để bật mức WARNING
        min_direct_coverage: float = 0.35,       # Độ phủ tối thiểu của làn chính ở vùng mũi/tâm xe
        hysteresis_frames: int = 3,              # Số frame liên tiếp cần khớp để chốt chuyển trạng thái
    ) -> None:
        self.camera_offset_ratio = float(camera_offset_ratio)
        self.ego_y_ratio = float(ego_y_ratio)
        self.wheel_strip_height_ratio = float(wheel_strip_height_ratio)
        self.wheel_half_width_ratio = float(wheel_half_width_ratio)
        self.lane_overlap_threshold = float(lane_overlap_threshold)
        self.min_direct_coverage = float(min_direct_coverage)
        # Đảm bảo tối thiểu 1 frame để bộ đệm lịch sử hoạt động hợp lệ
        self.hysteresis_frames = max(1, int(hysteresis_frames))

        # Lưu lịch sử trạng thái tức thời qua các frame gần nhất
        self._history: deque[str] = deque(maxlen=self.hysteresis_frames)
        # Trạng thái hiện tại được chốt của LDW: "NORMAL" | "WARNING" | "DEPARTED"
        self._current_status: str = "NORMAL"

    def reset(self) -> None:
        """Xóa lịch sử theo dõi và đưa trạng thái về NORMAL."""
        self._history.clear()
        self._current_status = "NORMAL"

    def update(
        self,
        seg_mask: np.ndarray,
        frame_shape: tuple[int, int] | None = None,
    ) -> tuple[str, list[ADASAlert]]:
        """Đánh giá trạng thái lệch làn / đè vạch cho khung hình hiện tại.

        Parameters
        ----------
        seg_mask : np.ndarray
            Mặt nạ phân vùng dạng xác suất hoặc nhị phân.
            Kỳ vọng shape: (3, H, W) với:
                channel 0: drivable_direct (làn đường trực tiếp phía trước xe)
                channel 1: drivable_alternative (làn đường phụ / làn kế bên)
                channel 2: lane_marking (vạch kẻ phân làn)
        frame_shape : tuple[int, int] | None
            Kích thước (H, W) gốc của khung hình (hiện tham số này nhận vào nhưng chưa dùng nội suy).

        Returns
        -------
        tuple[str, list[ADASAlert]]
- ldw_status: "NORMAL" | "WARNING" | "DEPARTED"
- alerts: Danh sách chứa ADASAlert khi ở trạng thái WARNING hoặc DEPARTED, rỗng nếu NORMAL.
        """
        # Kiểm tra kiểu và kích thước đầu vào của mặt nạ phân vùng
        if not isinstance(seg_mask, np.ndarray):
            raise TypeError(f"Expected seg_mask as np.ndarray, got {type(seg_mask).__name__}")
        if seg_mask.ndim != 3 or seg_mask.shape[0] != 3:
            raise ValueError(f"Expected seg_mask with shape (3, H, W), got {seg_mask.shape}")

        height, width = seg_mask.shape[1], seg_mask.shape[2]
        if height == 0 or width == 0:
            return "NORMAL", []

        # Ngưỡng hóa từng kênh về mặt nạ boolean (True/False)
        direct_mask = seg_mask[0] > 0.5
        alt_mask = seg_mask[1] > 0.5
        lane_mask = seg_mask[2] > 0.5

        # Bước 1: Tính tọa độ dải tham chiếu (tâm xe, vùng mũi xe, vệt bánh trái & phải)
        ego_x = int(round(width * (0.5 + self.camera_offset_ratio)))
        ego_y = int(round(height * self.ego_y_ratio))
        strip_half_h = max(1, int(round(height * self.wheel_strip_height_ratio * 0.5)))
        wheel_half_w = max(1, int(round(width * self.wheel_half_width_ratio)))
        wheel_contact_w = max(1, int(round(width * 0.03)))

        # Giới hạn dải chiều cao kiểm tra quanh ego_y
        y1 = max(0, ego_y - strip_half_h)
        y2 = min(height, ego_y + strip_half_h)

        # Vùng giữa (mũi/tâm xe)
        center_x1 = max(0, ego_x - wheel_contact_w)
        center_x2 = min(width, ego_x + wheel_contact_w)

        # Vùng tiếp xúc bánh xe bên trái
        left_w_x1 = max(0, ego_x - wheel_half_w - wheel_contact_w)
        left_w_x2 = min(width, ego_x - wheel_half_w + wheel_contact_w)

        # Vùng tiếp xúc bánh xe bên phải
        right_w_x1 = max(0, ego_x + wheel_half_w - wheel_contact_w)
        right_w_x2 = min(width, ego_x + wheel_half_w + wheel_contact_w)

        # Bước 2: Tính tỷ lệ điểm ảnh dương trong từng vùng ROI
        # Độ phủ làn chính tại vùng tâm xe
        center_direct_cov = (
            direct_mask[y1:y2, center_x1:center_x2].mean()
            if (y2 > y1 and center_x2 > center_x1)
            else 0.0
        )
        # Độ phủ làn phụ tại vùng tâm xe
        center_alt_cov = (
            alt_mask[y1:y2, center_x1:center_x2].mean()
            if (y2 > y1 and center_x2 > center_x1)
            else 0.0
        )

        # Tỷ lệ chạm vạch của bánh trái và bánh phải
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

        # Bước 3: Xác định trạng thái tức thời trên frame hiện tại
        # DEPARTED: tâm xe hụt độ phủ làn chính (< min_direct_coverage) đồng thời sang làn phụ hoặc đè vạch mạnh
        if center_direct_cov < self.min_direct_coverage and (
            center_alt_cov > 0.25 or max_lane_overlap > 0.35
        ):
            instant_status = "DEPARTED"
        # WARNING: bánh xe chạm/đè vạch đạt ngưỡng lane_overlap_threshold
        elif max_lane_overlap >= self.lane_overlap_threshold:
            instant_status = "WARNING"
        else:
            instant_status = "NORMAL"

        # Bước 4: Bộ lọc trễ theo thời gian (Temporal Hysteresis)
        self._history.append(instant_status)
        if len(self._history) >= self.hysteresis_frames:
            # Chỉ chuyển sang trạng thái mới khi đủ hysteresis_frames liên tiếp cùng kết quả
            if all(s == instant_status for s in self._history):
                self._current_status = instant_status
            elif instant_status == "DEPARTED":
                # Nếu đang NORMAL mà gặp DEPARTED tức thời thì nâng lên WARNING trước để cảnh báo sớm
                if self._current_status == "NORMAL":
                    self._current_status = "WARNING"
        else:
            # Khi chưa đủ số frame lịch sử, bám theo trạng thái tức thời
            self._current_status = instant_status

        # Bước 5: Tạo cảnh báo tương ứng với trạng thái đã chốt
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
        """Khởi tạo LaneDepartureWarner từ dictionary cấu hình."""
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