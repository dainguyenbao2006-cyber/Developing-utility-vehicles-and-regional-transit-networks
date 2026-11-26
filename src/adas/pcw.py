"""Pedestrian Collision Warning (PCW) purely using NumPy and Python stdlib."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from utils.geometry import bbox_bottom_center
from utils.types import ADASAlert, Detection

logger = logging.getLogger(__name__)

# Các lớp người tham gia giao thông dễ bị tổn thương cần cảnh báo va chạm
DEFAULT_PCW_CLASSES = ("pedestrian", "rider", "bicycle")


class PedestrianCollisionWarner:
    """Phát hiện người đi bộ, người lái xe hai bánh và xe đạp nằm trong làn/vùng nguy hiểm phía trước."""

    def __init__(
        self,
        target_classes: tuple[str, ...] | list[str] = DEFAULT_PCW_CLASSES,
        danger_y_min_ratio: float = 0.50,     # Ngưỡng tung độ y tối thiểu để coi là gần (tránh vật thể quá xa ở trên cao)
        in_lane_iou_threshold: float = 0.10,  # Tỷ lệ pixel của bbox nằm trên vùng drivable_direct tối thiểu để tính là vào làn
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
        """Kiểm tra đối tượng mục tiêu có nằm trong vùng nguy hiểm trực tiếp trước đầu xe không.

        Parameters
        ----------
        detections : list[Detection]
            Danh sách các đối tượng phát hiện được trên khung hình hiện tại.
        frame_shape : tuple[int, int]
            Kích thước (Height, Width) của khung hình.
        seg_mask : np.ndarray | None
            Mặt nạ phân vùng 3 kênh tùy chọn (kênh 0 là drivable_direct).

        Returns
        -------
        list[ADASAlert]
            Danh sách cảnh báo PCW tạo ra cho các đối tượng có nguy cơ.
        """
        height, width = frame_shape
        # Mốc y phân cách vùng xa (phía trên) và vùng nguy hiểm gần (phía dưới)
        danger_y_cutoff = height * self.danger_y_min_ratio
        alerts: list[ADASAlert] = []

        # Lấy mặt nạ làn đường trực tiếp (kênh 0) nếu có mask phân vùng 3 chiều
        direct_mask = (seg_mask[0] > 0.5) if (seg_mask is not None and seg_mask.ndim == 3) else None

        for det in detections:
            # Bỏ qua nếu không thuộc nhóm đối tượng PCW (pedestrian, rider, bicycle)
            if det.class_name not in self.target_classes:
                continue

            x1, y1, x2, y2 = det.bbox_xyxy
            # Nếu cạnh đáy khung bao nằm cao hơn danger_y_cutoff -> ở xa, chưa coi là nguy cơ cận kề
            if y2 < danger_y_cutoff:
                continue

            # Lấy điểm tiếp xúc mặt đường ở giữa cạnh đáy khung bao và kẹp gọn trong biên ảnh
            mid_x, bottom_y = bbox_bottom_center((x1, y1, x2, y2))
            px = max(0, min(width - 1, int(round(mid_x))))
            py = max(0, min(height - 1, int(round(bottom_y))))

            in_danger_zone = False
            if direct_mask is not None and direct_mask.shape == (height, width):
                # Trường hợp 1: Điểm tiếp xúc đáy khung bao nằm ngay trên làn đường trực tiếp
                if direct_mask[py, px]:
                    in_danger_zone = True
                else:
                    # Trường hợp 2: Kiểm tra mật độ phủ của drivable_direct bên trong hình chữ nhật bbox
                    box_roi = direct_mask[int(y1) : int(y2), int(x1) : int(x2)]
                    if box_roi.size > 0 and box_roi.mean() >= self.in_lane_iou_threshold:
                        in_danger_zone = True
            else:
                # Trường hợp dự phòng khi không có seg_mask khớp kích thước:
                # Xét vùng hành lang giữa rộng 40% chiều rộng ảnh (mỗi bên 20% tính từ tâm)
                center_x = 0.5 * width
                if abs(mid_x - center_x) <= 0.20 * width:
                    in_danger_zone = True

            # Nếu nằm trong vùng nguy hiểm -> quy đổi tên tiếng Việt và tạo cảnh báo CRITICAL
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
        """Khởi tạo PedestrianCollisionWarner từ dictionary cấu hình."""
        pcw_cfg = adas_cfg.get("pcw", adas_cfg)
        return cls(
            target_classes=pcw_cfg.get("target_classes", DEFAULT_PCW_CLASSES),
            danger_y_min_ratio=pcw_cfg.get("danger_y_min_ratio", 0.50),
            in_lane_iou_threshold=pcw_cfg.get("in_lane_iou_threshold", 0.10),
        )