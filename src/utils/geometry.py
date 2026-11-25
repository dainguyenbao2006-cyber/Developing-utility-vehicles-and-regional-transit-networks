"""Pure NumPy + stdlib geometric utilities for bounding boxes and ROI masks."""

from __future__ import annotations

import numpy as np


def clip_xyxy(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Clamp an xyxy bounding box to image boundaries [0, width] x [0, height]."""
    x1, y1, x2, y2 = bbox
    cx1 = max(0.0, min(float(x1), float(width)))
    cy1 = max(0.0, min(float(y1), float(height)))
    cx2 = max(0.0, min(float(x2), float(width)))
    cy2 = max(0.0, min(float(y2), float(height)))
    return (min(cx1, cx2), min(cy1, cy2), max(cx1, cx2), max(cy1, cy2))


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    """Compute area of an xyxy bounding box."""
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_bottom_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the bottom-center (x_mid, y2) contact point of an xyxy bounding box."""
    x1, _, x2, y2 = bbox
    return ((x1 + x2) * 0.5, y2)


def bbox_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union between two xyxy bounding boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area <= 0.0:
        return 0.0

    union_area = bbox_area(box_a) + bbox_area(box_b) - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def xyxy_to_yolo(
    bbox: tuple[float, float, float, float],
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    """Convert pixel xyxy box to normalized YOLO (x_center, y_center, width, height)."""
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"Image dimensions must be positive, got ({img_width}, {img_height})")

    x1, y1, x2, y2 = clip_xyxy(bbox, img_width, img_height)
    cx = ((x1 + x2) * 0.5) / img_width
    cy = ((y1 + y2) * 0.5) / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height
    return (cx, cy, w, h)


def yolo_to_xyxy(
    yolo_box: tuple[float, float, float, float],
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    """Convert normalized YOLO (cx, cy, w, h) box to pixel xyxy coordinates."""
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"Image dimensions must be positive, got ({img_width}, {img_height})")

    cx, cy, w, h = yolo_box
    x1 = (cx - w * 0.5) * img_width
    y1 = (cy - h * 0.5) * img_height
    x2 = (cx + w * 0.5) * img_width
    y2 = (cy + h * 0.5) * img_height
    return clip_xyxy((x1, y1, x2, y2), img_width, img_height)


def create_trapezoid_mask(
    height: int,
    width: int,
    top_y_ratio: float,
    top_width_ratio: float,
    bottom_width_ratio: float,
    center_x_ratio: float = 0.5,
) -> np.ndarray:
    """Create a boolean (H, W) trapezoid ROI mask centered horizontally near center_x_ratio."""
    if height <= 0 or width <= 0:
        raise ValueError(f"Mask dimensions must be positive, got ({height}, {width})")

    top_y = max(0, min(height - 1, int(round(height * top_y_ratio))))
    bottom_y = height - 1
    if bottom_y <= top_y:
        return np.zeros((height, width), dtype=bool)

    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]

    progress = np.clip((ys - top_y) / float(bottom_y - top_y), 0.0, 1.0)
    half_width_px = 0.5 * width * (
        top_width_ratio + progress * (bottom_width_ratio - top_width_ratio)
    )
    center_x_px = width * center_x_ratio

    in_y = ys >= top_y
    in_x = np.abs(xs - center_x_px) <= half_width_px
    return in_y & in_x
