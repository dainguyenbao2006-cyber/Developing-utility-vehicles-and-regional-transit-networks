from __future__ import annotations

import numpy as np
import pytest

from src.adas.ldw import LaneDepartureWarning, LDWConfig


def _make_masks(
    h: int = 720,
    w: int = 1280,
    left_lane_center: tuple[int, int] | None = None,
    right_lane_center: tuple[int, int] | None = None,
    direct_box: tuple[int, int, int, int] | None = None,
    alt_box: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tạo bộ 3 mask giả lập (drivable_direct, drivable_alternative, lane_marking)."""
    direct = np.zeros((h, w), dtype=np.uint8)
    alt = np.zeros((h, w), dtype=np.uint8)
    lane = np.zeros((h, w), dtype=np.uint8)

    if direct_box is not None:
        x1, y1, x2, y2 = direct_box
        direct[y1:y2, x1:x2] = 1
    else:
        # Mặc định tạo vùng làn chính ở giữa/đáy khung hình
        direct[int(h * 0.45) : h, int(w * 0.35) : int(w * 0.65)] = 1

    if alt_box is not None:
        x1, y1, x2, y2 = alt_box
        alt[y1:y2, x1:x2] = 1

    if left_lane_center is not None:
        cx, cy = left_lane_center
        lane[max(0, cy - 16) : min(h, cy + 17), max(0, cx - 16) : min(w, cx + 17)] = 1

    if right_lane_center is not None:
        cx, cy = right_lane_center
        lane[max(0, cy - 16) : min(h, cy + 17), max(0, cx - 16) : min(w, cx + 17)] = 1

    return direct, alt, lane


def test_ldw_safe_when_centered() -> None:
    cfg = LDWConfig(frames_to_trigger=3, frames_to_clear=4)
    ldw = LaneDepartureWarning(cfg)
    direct, alt, lane = _make_masks(h=720, w=1280)

    for _ in range(5):
        out = ldw.update(direct, alt, lane)

    assert out["alert"] is False
    assert out["state"] == "SAFE"


def test_ldw_triggers_left_with_hysteresis() -> None:
    cfg = LDWConfig(
        wheel_half_width_ratio=0.14,
        wheel_y_ratio=0.90,
        lane_overlap_thresh=0.25,
        frames_to_trigger=3,
        frames_to_clear=4,
    )
    ldw = LaneDepartureWarning(cfg)

    # Ở ảnh 1280x720: x_ego = 640, y_wheel = 648, dx = 179 -> x_left = 461
    direct, alt, lane = _make_masks(h=720, w=1280, left_lane_center=(461, 648))

    out1 = ldw.update(direct, alt, lane)
    out2 = ldw.update(direct, alt, lane)
    assert out1["alert"] is False
    assert out2["alert"] is False

    out3 = ldw.update(direct, alt, lane)
    assert out3["alert"] is True
    assert out3["state"] == "LDW_LEFT"


def test_ldw_ignores_single_frame_noise() -> None:
    cfg = LDWConfig(frames_to_trigger=3, frames_to_clear=4)
    ldw = LaneDepartureWarning(cfg)

    safe_direct, safe_alt, safe_lane = _make_masks(h=720, w=1280)
    noisy_direct, noisy_alt, noisy_lane = _make_masks(
        h=720, w=1280, right_lane_center=(819, 648)
    )

    # Xen kẽ nhiễu 1 frame rồi về an toàn
    for _ in range(3):
        ldw.update(safe_direct, safe_alt, safe_lane)
        out_noise = ldw.update(noisy_direct, noisy_alt, noisy_lane)
        assert out_noise["alert"] is False
        out_safe = ldw.update(safe_direct, safe_alt, safe_lane)
        assert out_safe["state"] == "SAFE"


def test_ldw_clears_only_after_clear_streak() -> None:
    cfg = LDWConfig(frames_to_trigger=2, frames_to_clear=3)
    ldw = LaneDepartureWarning(cfg)

    hit_direct, hit_alt, hit_lane = _make_masks(h=720, w=1280, right_lane_center=(819, 648))
    safe_direct, safe_alt, safe_lane = _make_masks(h=720, w=1280)

    # Kích hoạt LDW_RIGHT sau 2 frame
    ldw.update(hit_direct, hit_alt, hit_lane)
    active = ldw.update(hit_direct, hit_alt, hit_lane)
    assert active["alert"] is True
    assert active["state"] == "LDW_RIGHT"

    # 2 frame an toàn đầu tiên vẫn giữ cảnh báo (2 < 3)
    assert ldw.update(safe_direct, safe_alt, safe_lane)["state"] == "LDW_RIGHT"
    assert ldw.update(safe_direct, safe_alt, safe_lane)["state"] == "LDW_RIGHT"

    # Frame an toàn thứ 3 mới tắt cảnh báo
    cleared = ldw.update(safe_direct, safe_alt, safe_lane)
    assert cleared["alert"] is False
    assert cleared["state"] == "SAFE"


def test_ldw_invalid_mask_shape_raises() -> None:
    ldw = LaneDepartureWarning()
    bad_mask = np.zeros((10, 10, 3), dtype=np.uint8)
    good_mask = np.zeros((10, 10), dtype=np.uint8)

    with pytest.raises(ValueError):
        ldw.update(bad_mask, good_mask, good_mask)