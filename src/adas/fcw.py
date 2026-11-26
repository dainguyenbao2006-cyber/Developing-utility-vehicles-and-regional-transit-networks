"""Forward Collision Warning (FCW Proxy) purely using NumPy and Python stdlib."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from utils.geometry import bbox_area, bbox_bottom_center, bbox_iou
from utils.types import ADASAlert, Detection

logger = logging.getLogger(__name__)

# Các lớp phương tiện cơ giới cần xét nguy cơ va chạm phía trước
DEFAULT_TARGET_CLASSES = ("car", "truck", "bus", "motorcycle")


class ForwardCollisionWarner:
    """Đánh giá nguy cơ va chạm với xe phía trước nằm trong hành lang di chuyển của xe (ego corridor)."""

    def __init__(
        self,
        target_classes: tuple[str, ...] | list[str] = DEFAULT_TARGET_CLASSES,
        corridor_top_width_ratio: float = 0.18,    # Tỷ lệ bề rộng hành lang ở mép trên (so với chiều rộng ảnh)
        corridor_bottom_width_ratio: float = 0.45, # Tỷ lệ bề rộng hành lang ở mép dưới (so với chiều rộng ảnh)
        corridor_top_y_ratio: float = 0.45,        # Ngưỡng tung độ y bắt đầu hành lang (tính từ trên xuống)
        caution_area_ratio: float = 0.04,          # Tỷ lệ diện tích bbox so với ảnh để bật mức CAUTION
        critical_area_ratio: float = 0.10,         # Tỷ lệ diện tích bbox so với ảnh để bật mức CRITICAL
        growth_rate_critical: float = 0.25,        # Tốc độ tăng diện tích tương đối vượt ngưỡng -> CRITICAL
        history_window: int = 5,                   # Số frame tối đa lưu trong bộ đệm lịch sử xe phía trước
        match_iou_threshold: float = 0.30,         # Ngưỡng IoU tối thiểu để coi là cùng một xe qua các frame
    ) -> None:
        self.target_classes = set(target_classes)
        self.corridor_top_width_ratio = float(corridor_top_width_ratio)
        self.corridor_bottom_width_ratio = float(corridor_bottom_width_ratio)
        self.corridor_top_y_ratio = float(corridor_top_y_ratio)
        self.caution_area_ratio = float(caution_area_ratio)
        self.critical_area_ratio = float(critical_area_ratio)
        self.growth_rate_critical = float(growth_rate_critical)
        # Dùng max(2, ...) để luôn có ít nhất 2 phần tử khi so sánh frame cũ và frame mới
        self.history_window = max(2, int(history_window))
        self.match_iou_threshold = float(match_iou_threshold)

        # Hàng đợi lưu lịch sử (bbox, area_ratio) của xe phía trước; tự bỏ phần tử cũ nhất khi đầy maxlen
        self._lead_history: deque[tuple[tuple[float, float, float, float], float]] = deque(
            maxlen=self.history_window
        )
        # Trạng thái hiện tại của FCW: "SAFE" | "CAUTION" | "CRITICAL"
        self._current_status: str = "SAFE"

    def reset(self) -> None:
        """Xóa lịch sử theo dõi và đưa trạng thái về SAFE (dùng khi đổi video hoặc chạy lại)."""
        self._lead_history.clear()
        self._current_status = "SAFE"

    def is_in_corridor(
        self,
        bbox: tuple[float, float, float, float],
        width: int,
        height: int,
    ) -> bool:
        """Kiểm tra điểm tiếp xúc mặt đường (giữa cạnh đáy bbox) có nằm trong hành lang hình thang không."""
        # Lấy tọa độ tâm cạnh đáy (x_giữa, y_đáy) — đại diện cho vị trí bánh xe chạm mặt đường
        mid_x, bottom_y = bbox_bottom_center(bbox)
        top_y = height * self.corridor_top_y_ratio

        # Loại các vật thể nằm quá cao (trên đường chân trời/hành lang) hoặc vượt ra ngoài đáy ảnh
        if bottom_y < top_y or bottom_y > height:
            return False

        # Nội suy tuyến tính độ mở rộng của hành lang theo chiều sâu (từ top_y xuống đáy ảnh)
        progress = max(0.0, min(1.0, (bottom_y - top_y) / max(1.0, height - top_y)))
        current_width_ratio = (
            self.corridor_top_width_ratio
            + progress * (self.corridor_bottom_width_ratio - self.corridor_top_width_ratio)
        )

        # Nửa bề rộng hành lang tại độ cao bottom_y, lấy trục giữa ảnh làm tâm xe
        half_w = 0.5 * width * current_width_ratio
        center_x = 0.5 * width

        # Nằm trong hành lang nếu khoảng cách ngang tới tâm xe không vượt quá half_w
        return abs(mid_x - center_x) <= half_w

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> tuple[str, Detection | None, list[ADASAlert]]:
        """Cập nhật trạng thái FCW cho từng khung hình.

        Trả về:
        - fcw_status: "SAFE" | "CAUTION" | "CRITICAL"
        - lead_vehicle: Detection của xe được chọn là xe phía trước (hoặc None nếu không có)
        - alerts: Danh sách chứa tối đa 1 ADASAlert khi ở mức CAUTION hoặc CRITICAL
        """
        height, width = frame_shape
        frame_area = float(max(1, height * width))

        # Bước 1: Lọc các phương tiện thuộc lớp mục tiêu và nằm trong hành lang phía trước
        corridor_candidates: list[Detection] = []
        for det in detections:
            if det.class_name in self.target_classes:
                if self.is_in_corridor(det.bbox_xyxy, width=width, height=height):
                    corridor_candidates.append(det)

        # Nếu không có phương tiện nào trong hành lang -> reset lịch sử và trả về SAFE
        if not corridor_candidates:
            self._lead_history.clear()
            self._current_status = "SAFE"
            return "SAFE", None, []

        # Bước 2: Chọn xe phía trước là xe có y2 (cạnh đáy bbox) lớn nhất -> gần camera nhất
        lead_vehicle = max(corridor_candidates, key=lambda d: d.bbox_xyxy[3])
        lead_box = lead_vehicle.bbox_xyxy
        current_area_ratio = bbox_area(lead_box) / frame_area

        # Bước 3: Tính tỷ lệ tăng diện tích tương đối so với bản ghi đầu tiên trong cửa sổ lịch sử
        growth_rate = 0.0
        if self._lead_history:
            prev_box, prev_ratio = self._lead_history[0]
            # Chỉ tính tốc độ phình nếu bbox hiện tại khớp đủ nhiều (IoU) với xe ở lịch sử
            if bbox_iou(lead_box, prev_box) >= self.match_iou_threshold:
                growth_rate = (current_area_ratio - prev_ratio) / max(1e-4, prev_ratio)

        # Lưu trạng thái hiện tại vào lịch sử cho các frame sau
        self._lead_history.append((lead_box, current_area_ratio))

        # Bước 4: Phân tầng mức cảnh báo
        # CRITICAL: xe đã chiếm diện tích lớn (>= critical_area_ratio),
        # HOẶC xe đã ở mức CAUTION và phình diện tích nhanh (>= growth_rate_critical)
        is_critical = (
            current_area_ratio >= self.critical_area_ratio
            or (
                current_area_ratio >= self.caution_area_ratio
                and growth_rate >= self.growth_rate_critical
            )
        )
        # CAUTION: xe đạt ngưỡng diện tích chú ý tối thiểu
        is_caution = current_area_ratio >= self.caution_area_ratio

        if is_critical:
            self._current_status = "CRITICAL"
        elif is_caution:
            self._current_status = "CAUTION"
        else:
            self._current_status = "SAFE"

        # Bước 5: Tạo đối tượng cảnh báo ADASAlert nếu có nguy cơ
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
        """Khởi tạo ForwardCollisionWarner từ dictionary cấu hình (ví dụ đọc từ file YAML)."""
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