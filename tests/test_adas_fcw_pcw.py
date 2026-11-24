from __future__ import annotations

import numpy as np

from src.adas.alert_manager import AlertManager, AlertManagerConfig
from src.adas.fcw import FCWConfig, ForwardCollisionWarning
from src.adas.pcw import PCWConfig, PedestrianCollisionWarning


def test_fcw_detects_approaching_vehicle_in_corridor() -> None:
    cfg = FCWConfig(
        growth_thresh=0.05,
        critical_growth_thresh=0.15,
        warning_area_thresh=0.20,
        critical_area_thresh=0.50,
    )
    fcw = ForwardCollisionWarning(cfg)
    img_shape = (720, 1280)

    # Xe ở giữa hành lang (cx ~ 640, cy = y2 nằm trong [324, 684]) phình to dần theo thời gian
    boxes_and_times = [
        ([590.0, 420.0, 690.0, 500.0], 0.0),   # area = 8,000 / 921,600 ~ 0.0087
        ([560.0, 400.0, 720.0, 530.0], 0.1),   # area = 20,800 / 921,600 ~ 0.0226
        ([520.0, 380.0, 760.0, 570.0], 0.2),   # area = 45,600 / 921,600 ~ 0.0495
    ]

    last_out = None
    for bbox, ts in boxes_and_times:
        dets = [{"label": "car", "confidence": 0.9, "bbox": bbox}]
        last_out = fcw.update(detections=dets, img_shape=img_shape, timestamp=ts)

    assert last_out is not None
    assert last_out["alert"] is True
    assert last_out["severity"] in ("WARNING", "CRITICAL")
    assert last_out["target"] is not None
    assert last_out["target"]["growth_rate"] > 0.0


def test_fcw_ignores_vehicle_outside_corridor() -> None:
    fcw = ForwardCollisionWarning(FCWConfig())
    img_shape = (720, 1280)

    # Xe nằm hẳn sang bên trái ngoài Ego Corridor (cx = 150)
    dets = [{"label": "car", "confidence": 0.95, "bbox": [100.0, 450.0, 200.0, 560.0]}]
    out = fcw.update(detections=dets, img_shape=img_shape, timestamp=0.0)

    assert out["alert"] is False
    assert out["severity"] == "NONE"
    assert len(out["corridor_vehicles"]) == 0


def test_pcw_detects_pedestrian_in_danger_zone() -> None:
    pcw = PedestrianCollisionWarning(PCWConfig())
    img_shape = (720, 1280)
    drivable_direct = np.zeros(img_shape, dtype=np.uint8)
    drivable_direct[350:720, 450:830] = 1

    # Người đi bộ đứng ngay trước đầu xe (chiều cao ~ 220/720 ~ 0.305 -> CRITICAL)
    dets = [{"label": "pedestrian", "confidence": 0.88, "bbox": [600.0, 420.0, 670.0, 640.0]}]
    out = pcw.update(detections=dets, img_shape=img_shape, drivable_direct=drivable_direct)

    assert out["alert"] is True
    assert out["severity"] == "CRITICAL"
    assert len(out["hazards"]) == 1
    assert out["hazards"][0]["label"] == "pedestrian"


def test_alert_manager_priority_cooldown_and_escalation() -> None:
    mgr = AlertManager(AlertManagerConfig(cooldown_sec=1.0, escalate_on_severity_up=True))

    ldw_out = {"alert": True, "state": "LDW_LEFT", "left_lane_ratio": 0.5, "right_lane_ratio": 0.0, "ego_direct_ratio": 0.8}
    fcw_warn = {
        "alert": True,
        "severity": "WARNING",
        "target": {"label": "car", "growth_rate": 0.12, "area_ratio": 0.05, "bbox": [550.0, 400.0, 730.0, 560.0]},
    }
    fcw_crit = {
        "alert": True,
        "severity": "CRITICAL",
        "target": {"label": "car", "growth_rate": 0.28, "area_ratio": 0.16, "bbox": [480.0, 350.0, 800.0, 650.0]},
    }

    # t = 0.0s: Có cả LDW và FCW(WARNING) -> FCW phải được chọn làm primary_alert và ghi 1 event
    r0 = mgr.update(frame_idx=0, timestamp_sec=0.0, ldw_out=ldw_out, fcw_out=fcw_warn)
    assert r0["has_alert"] is True
    assert r0["primary_alert"]["alert_type"] == "FCW"
    assert r0["total_logged_events"] == 1

    # t = 0.3s: Vẫn là FCW(WARNING), chưa qua 1.0s cooldown -> không ghi thêm event mới
    r1 = mgr.update(frame_idx=9, timestamp_sec=0.3, ldw_out=ldw_out, fcw_out=fcw_warn)
    assert r1["total_logged_events"] == 1

    # t = 0.5s: Leo thang lên FCW(CRITICAL) -> ghi ngay dù chưa hết 1.0s cooldown
    r2 = mgr.update(frame_idx=15, timestamp_sec=0.5, ldw_out=ldw_out, fcw_out=fcw_crit)
    assert r2["primary_alert"]["severity"] == "CRITICAL"
    assert r2["total_logged_events"] == 2

    rows = mgr.get_log_rows()
    assert len(rows) == 2
    assert {"frame_idx", "timestamp_sec", "alert_type", "severity", "metric_value"}.issubset(rows[0].keys())