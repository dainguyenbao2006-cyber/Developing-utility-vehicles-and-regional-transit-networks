"""ADAS Alert Manager, priority sorter, cooldown throttle, and unified analyzer."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from adas.fcw import ForwardCollisionWarner
from adas.ldw import LaneDepartureWarner
from adas.pcw import PedestrianCollisionWarner
from adas.tsr import TrafficSignRecognizer  # Lưu ý: có thêm mô-đun TSR (nhận diện biển báo/tín hiệu)
from utils.constants import SEVERITY_LEVELS
from utils.types import ADASAlert, Detection, FrameAnalysis

logger = logging.getLogger(__name__)


class ADASAlertManager:
    """Lọc trùng cảnh báo theo thời gian (cooldown) và sắp xếp thứ tự ưu tiên."""

    def __init__(self, cooldown_sec: float = 1.0) -> None:
        # Thời gian giãn cách tối thiểu (giây) giữa 2 lần phát cùng một loại cảnh báo
        self.cooldown_sec = float(cooldown_sec)
        # Lưu mốc thời gian phát cảnh báo gần nhất theo từng loại: {alert_type: timestamp_sec}
        self._last_alert_time: dict[str, float] = {}

    def reset(self) -> None:
        """Xóa toàn bộ lịch sử mốc thời gian cảnh báo."""
        self._last_alert_time.clear()

    def filter_with_cooldown(
        self,
        alerts: list[ADASAlert],
        current_time_sec: float,
    ) -> list[ADASAlert]:
        """Loại bỏ các cảnh báo lặp lại cùng loại nếu chưa qua hết thời gian cooldown."""
        accepted: list[ADASAlert] = []
        for alert in alerts:
            # Ưu tiên an toàn: mức CRITICAL luôn được cho qua ngay lập tức, không bị chặn bởi cooldown
            if alert.severity == "CRITICAL":
                accepted.append(alert)
                self._last_alert_time[alert.alert_type] = current_time_sec
                continue

            # Với các mức nhẹ hơn (WARNING / CAUTION / INFO), kiểm tra khoảng cách thời gian từ lần báo trước
            last_time = self._last_alert_time.get(alert.alert_type, -999.0)
            if (current_time_sec - last_time) >= self.cooldown_sec:
                accepted.append(alert)
                self._last_alert_time[alert.alert_type] = current_time_sec

        return accepted

    @staticmethod
    def sort_by_priority(alerts: list[ADASAlert]) -> list[ADASAlert]:
        """Sắp xếp danh sách cảnh báo giảm dần theo mức độ nghiêm trọng (CRITICAL > WARNING > INFO > SAFE)."""
        return sorted(
            alerts,
            key=lambda a: SEVERITY_LEVELS.get(a.severity.upper(), 0),
            reverse=True,
        )


class ADASAnalyzer:
    """Bộ điều phối trung tâm chạy đồng thời LDW, FCW, PCW và TSR trên mỗi khung hình."""

    def __init__(
        self,
        ldw: LaneDepartureWarner,
        fcw: ForwardCollisionWarner,
        pcw: PedestrianCollisionWarner,
        tsr: TrafficSignRecognizer,
        alert_manager: ADASAlertManager | None = None,
    ) -> None:
        self.ldw = ldw
        self.fcw = fcw
        self.pcw = pcw
        self.tsr = tsr
        self.alert_manager = alert_manager or ADASAlertManager()

    def reset(self) -> None:
        """Đặt lại trạng thái nội bộ của các mô-đun có lưu lịch sử theo thời gian."""
        self.ldw.reset()
        self.fcw.reset()
        self.alert_manager.reset()

    def analyze_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        detections: list[Detection],
        seg_mask: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> FrameAnalysis:
        """Phân tích trọn vẹn một khung hình và đóng gói kết quả thành FrameAnalysis."""
        # Bước 1: Chạy từng mô-đun con để lấy trạng thái và danh sách cảnh báo thô
        ldw_status, ldw_alerts = self.ldw.update(seg_mask=seg_mask, frame_shape=frame_shape)
        fcw_status, _, fcw_alerts = self.fcw.update(detections=detections, frame_shape=frame_shape)
        pcw_alerts = self.pcw.update(
            detections=detections, frame_shape=frame_shape, seg_mask=seg_mask
        )
        tsr_alerts = self.tsr.update(detections=detections, frame_shape=frame_shape)

        # Bước 2: Gộp tất cả cảnh báo từ 4 mô-đun thành