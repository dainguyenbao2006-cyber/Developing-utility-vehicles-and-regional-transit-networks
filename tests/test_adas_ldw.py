"""Unit tests for Lane Departure Warning (LDW) logic."""

from __future__ import annotations

import numpy as np
import pytest

from adas.ldw import LaneDepartureWarner


def create_synthetic_road_mask(
    height: int = 100,
    width: int = 200,
    ego_lane_bounds: tuple[int, int] = (60, 140),
    lane_width: int = 4,
) -> np.ndarray:
    """Create a 3-channel synthetic road mask (direct, alternative, lane_marking)."""
    mask = np.zeros((3, height, width), dtype=np.uint8)
    left_x, right_x = ego_lane_bounds

    # Channel 0: drivable_direct
    mask[0, 40:, left_x:right_x] = 1

    # Channel 1: drivable_alternative (adjacent lanes on left and right)
    mask[1, 40:, :left_x] = 1
    mask[1, 40:, right_x:] = 1

    # Channel 2: lane_marking
    mask[2, 40:, left_x - lane_width : left_x] = 1
    mask[2, 40:, right_x : right_x + lane_width] = 1

    return mask


def test_ldw_normal_driving_centered() -> None:
    warner = LaneDepartureWarner(
        ego_y_ratio=0.88,
        wheel_half_width_ratio=0.12,  # 24 px half width on 200 px image (100 +/- 24 = 76 to 124)
        hysteresis_frames=1,
    )
    mask = create_synthetic_road_mask(height=100, width=200, ego_lane_bounds=(60, 140))

    status, alerts = warner.update(mask)
    assert status == "NORMAL"
    assert alerts == []


def test_ldw_warning_when_touching_lane_marking() -> None:
    warner = LaneDepartureWarner(
        ego_y_ratio=0.88,
        wheel_half_width_ratio=0.12,
        lane_overlap_threshold=0.10,
        hysteresis_frames=1,
    )
    # Shift ego lane so left wheel touches left boundary:
    # Ego center is at 100. Left wheel zone is at ~76. Set lane boundary at 76.
    mask = create_synthetic_road_mask(height=100, width=200, ego_lane_bounds=(76, 160))

    status, alerts = warner.update(mask)
    assert status == "WARNING"
    assert len(alerts) == 1
    assert alerts[0].severity == "WARNING"
    assert "Đang đè" in alerts[0].message


def test_ldw_departed_when_drifting_into_alternative_lane() -> None:
    warner = LaneDepartureWarner(
        ego_y_ratio=0.88,
        min_direct_coverage=0.35,
        hysteresis_frames=1,
    )
    # Lane is far to the right (140 to 190), so center nose at 100 is in alternative lane
    mask = create_synthetic_road_mask(height=100, width=200, ego_lane_bounds=(140, 190))

    status, alerts = warner.update(mask)
    assert status == "DEPARTED"
    assert len(alerts) == 1
    assert alerts[0].severity == "CRITICAL"
    assert "LỆCH LÀN ĐƯỜNG" in alerts[0].message


def test_ldw_temporal_hysteresis_suppresses_single_frame_flicker() -> None:
    warner = LaneDepartureWarner(
        ego_y_ratio=0.88,
        wheel_half_width_ratio=0.12,
        lane_overlap_threshold=0.10,
        hysteresis_frames=3,
    )
    normal_mask = create_synthetic_road_mask(height=100, width=200, ego_lane_bounds=(60, 140))
    warning_mask = create_synthetic_road_mask(height=100, width=200, ego_lane_bounds=(76, 160))

    # Frame 1: Normal
    status, _ = warner.update(normal_mask)
    assert status == "NORMAL"

    # Frame 2: Warning spike (should be buffered and not yet transition if starting from normal)
    warner.reset()
    status1, _ = warner.update(normal_mask)
    assert status1 == "NORMAL"

    status2, _ = warner.update(warning_mask)  # Frame 2
    # Frame 3: Sustained warning
    status3, _ = warner.update(warning_mask)
    assert status3 == "WARNING"


def test_ldw_camera_offset_adjusts_ego_center() -> None:
    # Camera offset = -0.1 moves center from 100 to 80 on a 200px width
    warner = LaneDepartureWarner(
        camera_offset_ratio=-0.10,
        ego_y_ratio=0.88,
        wheel_half_width_ratio=0.10,
        hysteresis_frames=1,
    )
    # Ego lane centered at 80 (50 to 110)
    mask = create_synthetic_road_mask(height=100, width=200, ego_lane_bounds=(50, 110))
    status, _ = warner.update(mask)
    assert status == "NORMAL"


def test_ldw_input_validation() -> None:
    warner = LaneDepartureWarner()

    with pytest.raises(TypeError, match="Expected seg_mask as np.ndarray"):
        warner.update("invalid")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Expected seg_mask with shape"):
        warner.update(np.zeros((2, 10, 10), dtype=np.uint8))
