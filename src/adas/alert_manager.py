"""ADAS Alert Manager, priority sorter, cooldown throttle, and unified analyzer."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from adas.fcw import ForwardCollisionWarner
from adas.ldw import LaneDepartureWarner
from adas.pcw import PedestrianCollisionWarner
from adas.tsr import TrafficSignRecognizer
from utils.constants import SEVERITY_LEVELS
from utils.types import ADASAlert, Detection, FrameAnalysis

logger = logging.getLogger(__name__)


class ADASAlertManager:
    """Filters, prioritizes, and manages cooldown throttling for ADAS alerts."""

    def __init__(self, cooldown_sec: float = 1.0) -> None:
        self.cooldown_sec = float(cooldown_sec)
        self._last_alert_time: dict[str, float] = {}

    def reset(self) -> None:
        """Clear alert timestamps."""
        self._last_alert_time.clear()

    def filter_with_cooldown(
        self,
        alerts: list[ADASAlert],
        current_time_sec: float,
    ) -> list[ADASAlert]:
        """Drop repeated alerts of the same type if within cooldown window."""
        accepted: list[ADASAlert] = []
        for alert in alerts:
            # Always pass CRITICAL alerts through immediately
            if alert.severity == "CRITICAL":
                accepted.append(alert)
                self._last_alert_time[alert.alert_type] = current_time_sec
                continue

            last_time = self._last_alert_time.get(alert.alert_type, -999.0)
            if (current_time_sec - last_time) >= self.cooldown_sec:
                accepted.append(alert)
                self._last_alert_time[alert.alert_type] = current_time_sec

        return accepted

    @staticmethod
    def sort_by_priority(alerts: list[ADASAlert]) -> list[ADASAlert]:
        """Sort alerts in descending order of severity (CRITICAL > WARNING > INFO > SAFE)."""
        return sorted(
            alerts,
            key=lambda a: SEVERITY_LEVELS.get(a.severity.upper(), 0),
            reverse=True,
        )


class ADASAnalyzer:
    """Unified coordinator executing LDW, FCW, PCW, and TSR rules on each frame."""

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
        """Reset state across all rule components."""
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
        """Evaluate a complete frame and return aggregated FrameAnalysis."""
        # 1. Run individual ADAS components
        ldw_status, ldw_alerts = self.ldw.update(seg_mask=seg_mask, frame_shape=frame_shape)
        fcw_status, _, fcw_alerts = self.fcw.update(detections=detections, frame_shape=frame_shape)
        pcw_alerts = self.pcw.update(
            detections=detections, frame_shape=frame_shape, seg_mask=seg_mask
        )
        tsr_alerts = self.tsr.update(detections=detections, frame_shape=frame_shape)

        # 2. Consolidate alerts
        raw_alerts = ldw_alerts + fcw_alerts + pcw_alerts + tsr_alerts

        # 3. Apply cooldown and sort by severity
        filtered_alerts = self.alert_manager.filter_with_cooldown(
            alerts=raw_alerts,
            current_time_sec=timestamp_sec,
        )
        prioritized_alerts = self.alert_manager.sort_by_priority(filtered_alerts)

        return FrameAnalysis(
            frame_idx=int(frame_idx),
            timestamp_sec=float(timestamp_sec),
            detections=detections,
            ldw_status=ldw_status,
            fcw_status=fcw_status,
            alerts=prioritized_alerts,
        )

    @classmethod
    def from_config(cls, adas_cfg: dict[str, Any]) -> ADASAnalyzer:
        """Instantiate complete ADASAnalyzer from adas config dictionary."""
        rules = adas_cfg.get("adas", adas_cfg)
        ldw = LaneDepartureWarner.from_config(rules)
        fcw = ForwardCollisionWarner.from_config(rules)
        pcw = PedestrianCollisionWarner.from_config(rules)
        tsr = TrafficSignRecognizer.from_config(rules)

        cooldown = float(rules.get("alert_manager", {}).get("cooldown_sec", 1.0))
        alert_mgr = ADASAlertManager(cooldown_sec=cooldown)

        return cls(ldw=ldw, fcw=fcw, pcw=pcw, tsr=tsr, alert_manager=alert_mgr)
