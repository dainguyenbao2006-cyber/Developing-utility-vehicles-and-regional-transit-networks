"""Traffic Sign and Traffic Light Awareness (TSR) purely using NumPy and stdlib."""

from __future__ import annotations

import logging
from typing import Any

from utils.geometry import bbox_area
from utils.types import ADASAlert, Detection

logger = logging.getLogger(__name__)

# Các lớp tín hiệu & biển báo giao thông được giám sát bởi mô-đun TSR
DEFAULT_TSR_CLASSES = ("traffic_light", "traffic_sign")


class TrafficSignRecognizer:
    """Theo dõi đèn tín hiệu và biển báo giao thông phát hiện được trong góc nhìn phía trước của xe."""

    def __init__(
        self,
        target_classes: tuple[str, ...] | list[str] = DEFAULT_TSR_CLASSES,
        min_confidence: float = 0.45,       # Ngưỡng độ tin cậy tối thiểu của bounding box từ Detector
        min_box_area_ratio: float = 0.0004, # Tỷ lệ diện tích bbox / diện tích khung hình tối thiểu (lọc biển/đèn quá xa)
    ) -> None:
        self.target_classes = set(target_classes)
        self.min_confidence = float(min_confidence)
        self.min_box_area_ratio = float(min_box_area_ratio)

    def update(
        self,
        detections: list[Detection],
        frame_shape: tuple[int, int],
    ) -> list[ADASAlert]:
        """Kiểm tra các đối tượng đèn/biển báo đạt ngưỡng và tạo cảnh báo thông tin (INFO).

        Parameters
        ----------
        detections : list[Detection]
            Danh sách các đối tượng phát hiện được trên khung hình hiện tại.
        frame_shape : tuple[int, int]
            Kích thước (Height, Width) của khung hình.

        Returns
        -------
        list[ADASAlert]
            Danh sách cảnh báo mức INFO cho đèn giao thông và biển báo hợp lệ.
        """
        height, width = frame_shape
        frame_area = float(max(1, height * width))
        alerts: list[ADASAlert] = []

        for det in detections:
            # Bước 1: Chỉ giữ lại các đối tượng thuộc lớp mục tiêu (traffic_light / traffic_sign)
            if det.class_name not in self.target_classes:
                continue
            # Bước 2: Lọc bỏ các dự đoán có độ tin cậy thấp hơn ngưỡng
            if det.confidence < self.min_confidence:
                continue

            # Bước 3: Tính tỷ lệ diện tích khung bao so với toàn khung hình; bỏ qua nếu quá nhỏ/quá xa
            area_ratio = bbox_area(det.bbox_xyxy) / frame_area
            if area_ratio < self.min_box_area_ratio:
                continue

            # Bước 4: Sinh cảnh báo mức INFO tương ứng theo loại tín hiệu
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
        """Khởi tạo TrafficSignRecognizer từ dictionary cấu hình (ví dụ đọc từ file YAML)."""
        tsr_cfg = adas_cfg.get("tsr", adas_cfg)
        return cls(
            target_classes=tsr_cfg.get("target_classes", DEFAULT_TSR_CLASSES),
            min_confidence=tsr_cfg.get("min_confidence", 0.45),
            min_box_area_ratio=tsr_cfg.get("min_box_area_ratio", 0.0004),
        )