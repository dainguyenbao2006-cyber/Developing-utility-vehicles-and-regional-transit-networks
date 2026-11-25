"""Unit tests for Forward Collision Warning (FCW), PCW, TSR, and ADASAnalyzer."""

from __future__ import annotations

import numpy as np

from adas.alert_manager import ADASAlertManager, ADASAnalyzer
from adas.fcw import ForwardCollisionWarner
from adas.ldw import LaneDepartureWarner
from adas.pcw import PedestrianCollisionWarner
from adas.tsr import TrafficSignRecognizer
from utils.types import ADASAlert, Detection, FrameAnalysis


def test_fcw_vehicle_outside_corridor_is_safe() -> None:
    warner = ForwardCollisionWarner()
    # Vehicle in adjacent lane on far left (x1=10, x2=50 on 640px image)
    det = Detection(
        bbox_xyxy=(10.0, 300.0, 50.0, 400.0),
        confidence=0.9,
        class_id=0,
        class_name="car",
    )

    status, lead, alerts = warner.update([det], frame_shape=(480, 640))
    assert status == "SAFE"
    assert lead is None
    assert alerts == []


def test_fcw_vehicle_inside_corridor_caution_and_critical() -> None:
    frame_h, frame_w = 480, 640

    warner = ForwardCollisionWarner(
        caution_area_ratio=0.04,  # area >= 12288
        critical_area_ratio=0.10,  # area >= 30720
    )

    # 1. Distant car in corridor: area = 40 * 50 = 2000 (< 0.04) -> SAFE
    car_safe = Detection(
        bbox_xyxy=(300.0, 250.0, 340.0, 300.0), confidence=0.9, class_id=0, class_name="car"
    )
    status, lead, alerts = warner.update([car_safe], frame_shape=(frame_h, frame_w))
    assert status == "SAFE"
    assert lead == car_safe
    assert alerts == []

    # 2. Moderate car in corridor: area = 120 * 120 = 14400 (> 0.04 frame area) -> CAUTION
    car_caution = Detection(
        bbox_xyxy=(260.0, 300.0, 380.0, 420.0), confidence=0.9, class_id=0, class_name="car"
    )
    status, lead, alerts = warner.update([car_caution], frame_shape=(frame_h, frame_w))
    assert status == "CAUTION"
    assert lead == car_caution
    assert len(alerts) == 1
    assert alerts[0].severity == "CAUTION"

    # 3. Very close car in corridor: area = 200 * 180 = 36000 (> 0.10 frame area) -> CRITICAL
    car_critical = Detection(
        bbox_xyxy=(220.0, 290.0, 420.0, 470.0), confidence=0.9, class_id=0, class_name="car"
    )
    status, lead, alerts = warner.update([car_critical], frame_shape=(frame_h, frame_w))
    assert status == "CRITICAL"
    assert alerts[0].severity == "CRITICAL"
    assert "HÃY PHANH" in alerts[0].message


def test_fcw_rapid_approach_escalates_to_critical() -> None:
    frame_h, frame_w = 480, 640
    warner = ForwardCollisionWarner(
        caution_area_ratio=0.03,
        critical_area_ratio=0.15,
        growth_rate_critical=0.20,
        history_window=3,
    )

    # Frame 1: Car area = 100 * 100 = 10000
    box1 = (270.0, 300.0, 370.0, 400.0)
    det1 = Detection(bbox_xyxy=box1, confidence=0.9, class_id=0, class_name="car")
    warner.update([det1], frame_shape=(frame_h, frame_w))

    # Frame 2: Car area expands to 125 * 125 = 15625 (> 50% growth rate)
    box2 = (257.5, 287.5, 382.5, 412.5)
    det2 = Detection(bbox_xyxy=box2, confidence=0.9, class_id=0, class_name="car")
    status, _, alerts = warner.update([det2], frame_shape=(frame_h, frame_w))

    assert status == "CRITICAL"
    assert len(alerts) == 1
    assert alerts[0].severity == "CRITICAL"


def test_pcw_pedestrian_in_lane() -> None:
    warner = PedestrianCollisionWarner(danger_y_min_ratio=0.50)
    frame_h, frame_w = 480, 640

    # 1. Distant pedestrian at top: y2 = 180 (< 0.5 * 480 = 240) -> No alert
    ped_far = Detection(
        bbox_xyxy=(310.0, 140.0, 330.0, 180.0), confidence=0.85, class_id=5, class_name="pedestrian"
    )
    alerts_far = warner.update([ped_far], frame_shape=(frame_h, frame_w))
    assert alerts_far == []

    # 2. Imminent pedestrian in lane path: y2 = 380, center_x = 320 -> CRITICAL
    ped_close = Detection(
        bbox_xyxy=(300.0, 300.0, 340.0, 380.0), confidence=0.85, class_id=5, class_name="pedestrian"
    )
    alerts_close = warner.update([ped_close], frame_shape=(frame_h, frame_w))
    assert len(alerts_close) == 1
    assert alerts_close[0].alert_type == "PCW"
    assert alerts_close[0].severity == "CRITICAL"
    assert "NGƯỜI ĐI BỘ" in alerts_close[0].message


def test_tsr_traffic_sign_and_light() -> None:
    recognizer = TrafficSignRecognizer(min_confidence=0.40, min_box_area_ratio=0.0001)
    frame_h, frame_w = 480, 640

    tl = Detection(
        bbox_xyxy=(400.0, 50.0, 430.0, 100.0),
        confidence=0.8,
        class_id=7,
        class_name="traffic_light",
    )
    ts = Detection(
        bbox_xyxy=(500.0, 100.0, 540.0, 140.0),
        confidence=0.75,
        class_id=8,
        class_name="traffic_sign",
    )

    alerts = recognizer.update([tl, ts], frame_shape=(frame_h, frame_w))
    assert len(alerts) == 2
    assert any(a.alert_type == "TSR" and "Đèn giao thông" in a.message for a in alerts)
    assert any(a.alert_type == "TSR" and "biển báo" in a.message for a in alerts)


def test_alert_manager_priority_and_cooldown() -> None:
    manager = ADASAlertManager(cooldown_sec=1.0)

    alert_info = ADASAlert(alert_type="TSR", severity="INFO", message="Sign")
    alert_warn = ADASAlert(alert_type="LDW", severity="WARNING", message="Lane")
    alert_crit = ADASAlert(alert_type="FCW", severity="CRITICAL", message="Brake")

    # Priority sorting
    sorted_alerts = manager.sort_by_priority([alert_info, alert_crit, alert_warn])
    assert sorted_alerts[0].severity == "CRITICAL"
    assert sorted_alerts[1].severity == "WARNING"
    assert sorted_alerts[2].severity == "INFO"

    # Cooldown throttling: repeated info alert within 0.5s is dropped
    filtered1 = manager.filter_with_cooldown([alert_info], current_time_sec=10.0)
    assert len(filtered1) == 1

    filtered2 = manager.filter_with_cooldown([alert_info], current_time_sec=10.5)
    assert len(filtered2) == 0  # throttled

    filtered3 = manager.filter_with_cooldown([alert_info], current_time_sec=11.2)
    assert len(filtered3) == 1  # allowed after 1.2s


def test_adas_analyzer_end_to_end() -> None:
    ldw = LaneDepartureWarner()
    fcw = ForwardCollisionWarner()
    pcw = PedestrianCollisionWarner()
    tsr = TrafficSignRecognizer()

    analyzer = ADASAnalyzer(ldw=ldw, fcw=fcw, pcw=pcw, tsr=tsr)

    seg_mask = np.zeros((3, 100, 200), dtype=np.uint8)
    seg_mask[0, 50:, 50:150] = 1  # Centered direct lane

    analysis = analyzer.analyze_frame(
        frame_idx=1,
        timestamp_sec=0.033,
        detections=[],
        seg_mask=seg_mask,
        frame_shape=(100, 200),
    )

    assert isinstance(analysis, FrameAnalysis)
    assert analysis.frame_idx == 1
    assert analysis.ldw_status == "NORMAL"
    assert analysis.fcw_status == "SAFE"
    assert analysis.alerts == []
