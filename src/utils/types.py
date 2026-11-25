"""Shared data contracts exchanged between detection, segmentation, ADAS, and HUD."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Detection:
    """Single object detection in original frame pixel coordinates (xyxy)."""

    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True)
class ADASAlert:
    """Single ADAS event emitted for a frame."""

    alert_type: str  # "LDW" | "FCW" | "PCW" | "TSR"
    severity: str    # "SAFE" | "INFO" | "WARNING" | "CAUTION" | "CRITICAL"
    message: str
    target_bbox: tuple[float, float, float, float] | None = None


@dataclass
class FrameAnalysis:
    """Aggregated ADAS evaluation output for one video frame."""

    frame_idx: int
    timestamp_sec: float
    detections: list[Detection] = field(default_factory=list)
    ldw_status: str = "NORMAL"
    fcw_status: str = "SAFE"
    alerts: list[ADASAlert] = field(default_factory=list)
