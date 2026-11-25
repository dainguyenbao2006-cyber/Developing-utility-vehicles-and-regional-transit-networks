"""Hằng số dùng chung toàn dự án — nguồn duy nhất (docs/02 §2).

Không viết lại tên lớp / chỉ số kênh ở module khác; luôn import từ đây.
"""
from __future__ import annotations

DET_CLASSES: tuple[str, ...] = (
    "car",            # 0
    "truck",          # 1
    "bus",            # 2
    "motorcycle",     # 3
    "bicycle",        # 4
    "pedestrian",     # 5
    "rider",          # 6
    "traffic_light",  # 7
    "traffic_sign",   # 8
)

SEG_CHANNELS: tuple[str, ...] = (
    "drivable_direct",       # Kênh 0: làn xe đang đi (ego lane)
    "drivable_alternative",  # Kênh 1: làn xe lân cận
    "lane_marking",          # Kênh 2: vạch kẻ đường
)

DET_CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(DET_CLASSES)}
SEG_CHANNEL_TO_ID: dict[str, int] = {name: i for i, name in enumerate(SEG_CHANNELS)}
NUM_DET_CLASSES: int = len(DET_CLASSES)
NUM_SEG_CHANNELS: int = len(SEG_CHANNELS)
