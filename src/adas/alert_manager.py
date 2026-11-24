from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class AlertManagerConfig:
    cooldown_sec: float = 1.0                     # Thời gian giãn cách (giây) giữa 2 lần ghi Event Log cùng loại
    escalate_on_severity_up: bool = True          # Cho phép ghi log ngay nếu severity tăng (WARNING -> CRITICAL)
    priority_order: tuple[str, ...] = (           # Thứ tự ưu tiên từ cao đến thấp giữa các loại cảnh báo
        "FCW",
        "PCW",
        "LDW_LEFT",
        "LDW_RIGHT",
    )


@dataclass
class AlertEvent:
    frame_idx: int
    timestamp_sec: float
    alert_type: str                               # "FCW" | "PCW" | "LDW_LEFT" | "LDW_RIGHT"
    severity: str                                 # "WARNING" | "CRITICAL"
    target_label: str = ""
    metric_value: float = 0.0
    bbox: list[float] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp_sec": round(self.timestamp_sec, 4),
            "alert_type": self.alert_type,
            "severity": self.severity,
            "target_label": self.target_label,
            "metric_value": round(self.metric_value, 4),
            "bbox": self.bbox,
            "message": self.message,
        }


class AlertManager:
    """Điều phối ưu tiên cảnh báo, áp dụng cooldown và sinh Event Log chuẩn CSV."""

    _SEVERITY_SCORES = np.array([0, 1, 2], dtype=np.float64)  # NONE=0, WARNING=1, CRITICAL=2

    def __init__(self, config: AlertManagerConfig | None = None) -> None:
        self.cfg = config or AlertManagerConfig()
        # Lưu thời điểm lần cuối phát sinh event cho từng alert_type
        self._last_emit_time: dict[str, float] = {}
        # Lưu severity lần cuối phát sinh event cho từng alert_type
        self._last_emit_severity_idx: dict[str, int] = {}
        # Lưu toàn bộ lịch sử sự kiện đã phát sinh trong phiên chạy
        self._event_log: list[AlertEvent] = []

        # Gán điểm ưu tiên loại cảnh báo (cảnh báo đứng đầu danh sách có điểm cao nhất)
        n_types = len(self.cfg.priority_order)
        self._type_priority_map: dict[str, float] = {
            alert_type: float(n_types - idx)
            for idx, alert_type in enumerate(self.cfg.priority_order)
        }

    def reset(self) -> None:
        """Xóa trạng thái cooldown và log khi bắt đầu video/phiên mới."""
        self._last_emit_time.clear()
        self._last_emit_severity_idx.clear()
        self._event_log.clear()

    @property
    def event_log(self) -> list[AlertEvent]:
        return list(self._event_log)

    def get_log_rows(self) -> list[dict]:
        """Trả về danh sách dict chuẩn để ghi ra file CSV hoặc DataFrame."""
        return [ev.to_dict() for ev in self._event_log]

    @staticmethod
    def _severity_to_idx(severity: str) -> int:
        s = severity.upper()
        if s == "CRITICAL":
            return 2
        if s == "WARNING":
            return 1
        return 0

    def _compute_rank_score(
        self,
        alert_type: str,
        severity_idx: int,
        metric_value: float,
    ) -> float:
        """Tính điểm xếp hạng thuần NumPy để chọn cảnh báo ưu tiên cao nhất."""
        type_score = self._type_priority_map.get(alert_type, 0.0)
        sev_score = float(self._SEVERITY_SCORES[np.clip(severity_idx, 0, 2)])
        metric_clamped = float(np.clip(metric_value, 0.0, 10.0))

        # Dùng vector trọng số thuần NumPy: ưu tiên severity & loại cảnh báo, tie-break bằng metric
        weights = np.array([100.0, 10.0, 1.0], dtype=np.float64)
        features = np.array([sev_score, type_score, metric_clamped], dtype=np.float64)
        return float(np.dot(weights, features))

    def _collect_candidates(
        self,
        ldw_out: dict | None,
        fcw_out: dict | None,
        pcw_out: dict | None,
    ) -> list[dict]:
        """Chuẩn hóa output từ LDW, FCW, PCW thành danh sách ứng viên cảnh báo."""
        candidates: list[dict] = []

        if ldw_out and ldw_out.get("alert"):
            state = str(ldw_out.get("state", "LDW_LEFT"))
            ratio = float(
                max(
                    float(ldw_out.get("left_lane_ratio", 0.0)),
                    float(ldw_out.get("right_lane_ratio", 0.0)),
                    1.0 - float(ldw_out.get("ego_direct_ratio", 1.0)),
                )
            )
            candidates.append(
                {
                    "alert_type": state,
                    "severity": "WARNING",
                    "target_label": "lane",
                    "metric_value": ratio,
                    "bbox": [],
                    "message": f"Lệch / đè vạch làn đường ({state})",
                }
            )

        if fcw_out and fcw_out.get("alert"):
            sev = str(fcw_out.get("severity", "WARNING")).upper()
            target = fcw_out.get("target") or {}
            growth = float(target.get("growth_rate", 0.0))
            area = float(target.get("area_ratio", 0.0))
            metric = max(growth, area)
            label = str(target.get("label", "vehicle"))
            bbox = list(target.get("bbox", []))
            candidates.append(
                {
                    "alert_type": "FCW",
                    "severity": sev if sev in ("WARNING", "CRITICAL") else "WARNING",
                    "target_label": label,
                    "metric_value": metric,
                    "bbox": bbox,
                    "message": f"Nguy cơ va chạm phía trước với {label} ({sev})",
                }
            )

        if pcw_out and pcw_out.get("alert"):
            sev = str(pcw_out.get("severity", "WARNING")).upper()
            target = pcw_out.get("target") or {}
            height_r = float(target.get("height_ratio", 0.0))
            overlap_r = float(target.get("overlap_ratio", 0.0))
            metric = max(height_r, overlap_r)
            label = str(target.get("label", "pedestrian"))
            bbox = list(target.get("bbox", []))
            candidates.append(
                {
                    "alert_type": "PCW",
                    "severity": sev if sev in ("WARNING", "CRITICAL") else "WARNING",
                    "target_label": label,
                    "metric_value": metric,
                    "bbox": bbox,
                    "message": f"Cảnh báo người đi bộ / xe thô sơ: {label} ({sev})",
                }
            )

        return candidates

    def _should_emit_event(
        self,
        alert_type: str,
        severity_idx: int,
        timestamp_sec: float,
    ) -> bool:
        """Kiểm tra cooldown trước khi ghi sự kiện mới vào Event Log."""
        if alert_type not in self._last_emit_time:
            return True

        elapsed = float(timestamp_sec) - float(self._last_emit_time[alert_type])
        if elapsed < 0.0:
            return False

        prev_sev_idx = self._last_emit_severity_idx.get(alert_type, 0)
        if self.cfg.escalate_on_severity_up and severity_idx > prev_sev_idx:
            return True                               # Mức độ nguy hiểm tăng lên -> ghi log ngay không chờ hết cooldown

        return elapsed >= self.cfg.cooldown_sec

    def update(
        self,
        frame_idx: int,
        timestamp_sec: float,
        ldw_out: dict | None = None,
        fcw_out: dict | None = None,
        pcw_out: dict | None = None,
    ) -> dict:
        raw_candidates = self._collect_candidates(ldw_out, fcw_out, pcw_out)

        ranked: list[dict] = []
        for cand in raw_candidates:
            sev_idx = self._severity_to_idx(cand["severity"])
            score = self._compute_rank_score(
                alert_type=cand["alert_type"],
                severity_idx=sev_idx,
                metric_value=cand["metric_value"],
            )
            item = dict(cand)
            item["score"] = score
            item["severity_idx"] = sev_idx
            ranked.append(item)

        # Sắp xếp giảm dần theo điểm rank score
        ranked.sort(key=lambda x: x["score"], reverse=True)

        emitted_events: list[AlertEvent] = []
        for item in ranked:
            atype = item["alert_type"]
            sev_idx = int(item["severity_idx"])
            if self._should_emit_event(atype, sev_idx, float(timestamp_sec)):
                ev = AlertEvent(
                    frame_idx=int(frame_idx),
                    timestamp_sec=float(timestamp_sec),
                    alert_type=atype,
                    severity=item["severity"],
                    target_label=item["target_label"],
                    metric_value=float(item["metric_value"]),
                    bbox=item["bbox"],
                    message=item["message"],
                )
                self._event_log.append(ev)
                emitted_events.append(ev)
                self._last_emit_time[atype] = float(timestamp_sec)
                self._last_emit_severity_idx[atype] = sev_idx

        primary_alert = ranked[0] if len(ranked) > 0 else None

        return {
            "has_alert": primary_alert is not None,
            "primary_alert": primary_alert,
            "active_alerts": ranked,
            "emitted_events": [ev.to_dict() for ev in emitted_events],
            "total_logged_events": len(self._event_log),
        }