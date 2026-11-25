"""Project-wide constants for detection classes, segmentation channels, and ADAS."""

from __future__ import annotations

DET_CLASSES: tuple[str, ...] = (
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "rider",
    "traffic_light",
    "traffic_sign",
)

DET_CLASS_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(DET_CLASSES)}

SEG_CHANNELS: tuple[str, ...] = (
    "drivable_direct",
    "drivable_alternative",
    "lane_marking",
)

SEG_CHANNEL_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(SEG_CHANNELS)}

# Raw BDD100K detection category strings -> canonical DET_CLASSES names.
BDD_DET_CATEGORY_MAP: dict[str, str] = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motor": "motorcycle",
    "motorcycle": "motorcycle",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "person": "pedestrian",
    "pedestrian": "pedestrian",
    "rider": "rider",
    "traffic light": "traffic_light",
    "traffic_light": "traffic_light",
    "traffic sign": "traffic_sign",
    "traffic_sign": "traffic_sign",
}

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

SEVERITY_LEVELS: dict[str, int] = {
    "SAFE": 0,
    "NORMAL": 0,
    "INFO": 1,
    "WARNING": 2,
    "CAUTION": 2,
    "DEPARTED": 3,
    "CRITICAL": 3,
}
