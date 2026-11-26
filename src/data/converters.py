"""Data conversion utilities for BDD100K detection/segmentation and subset splits."""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from utils.constants import BDD_DET_CATEGORY_MAP, DET_CLASS_TO_ID, SEG_CHANNELS
from utils.geometry import xyxy_to_yolo

logger = logging.getLogger(__name__)


def convert_bdd_frame_labels_to_yolo(
    labels: list[dict[str, Any]],
    img_width: int,
    img_height: int,
) -> list[str]:
    """Convert raw BDD100K frame label dicts into normalized YOLO label lines."""
    lines: list[str] = []
    for label in labels:
        if not isinstance(label, dict):
            continue
        raw_category = label.get("category")
        if not isinstance(raw_category, str):
            continue

        canonical = BDD_DET_CATEGORY_MAP.get(raw_category.strip().lower())
        if canonical is None:
            continue

        box2d = label.get("box2d")
        if not isinstance(box2d, dict):
            continue

        try:
            bbox_xyxy = (
                float(box2d["x1"]),
                float(box2d["y1"]),
                float(box2d["x2"]),
                float(box2d["y2"]),
            )
        except (KeyError, TypeError, ValueError):
            continue

        cx, cy, w, h = xyxy_to_yolo(bbox_xyxy, img_width=img_width, img_height=img_height)
        if w <= 0.0 or h <= 0.0:
            continue

        class_id = DET_CLASS_TO_ID[canonical]
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def _extract_bdd_frames(data: Any) -> list[dict[str, Any]]:
    """Extract frame list from BDD100K JSON (either list or dict['frames'])."""
    if isinstance(data, dict) and isinstance(data.get("frames"), list):
        return data["frames"]
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported BDD100K JSON format: expected list or dict['frames'].")


def convert_bdd_json_to_yolo_dir(
    json_path: Path,
    images_dir: Path,
    labels_out_dir: Path,
    allowed_stems: set[str] | None = None,
    default_size_wh: tuple[int, int] = (1280, 720),
) -> dict[str, int]:
    """Convert a BDD100K detection JSON file into per-image YOLO .txt label files."""
    if not json_path.exists():
        raise FileNotFoundError(f"BDD100K detection JSON not found: {json_path}")

    labels_out_dir.mkdir(parents=True, exist_ok=True)
    with json_path.open("r", encoding="utf-8") as handle:
        raw_data = json.load(handle)

    frames = _extract_bdd_frames(raw_data)
    stats = {"frames_written": 0, "boxes_written": 0, "skipped_missing_images": 0}

    for frame in frames:
        if not isinstance(frame, dict):
            continue
        image_name = frame.get("name")
        if not isinstance(image_name, str) or not image_name:
            continue

        stem = Path(image_name).stem
        if allowed_stems is not None and stem not in allowed_stems:
            continue

        img_path = images_dir / image_name
        if img_path.exists():
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        elif not images_dir.exists():
            img_w, img_h = default_size_wh
        else:
            # Skip frames whose image file is not present in images_dir.
            stats["skipped_missing_images"] += 1
            continue

        raw_labels = frame.get("labels") or []
        yolo_lines = convert_bdd_frame_labels_to_yolo(raw_labels, img_width=img_w, img_height=img_h)
        out_txt = labels_out_dir / f"{stem}.txt"
        out_txt.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        stats["frames_written"] += 1
        stats["boxes_written"] += len(yolo_lines)

    logger.info("Converted BDD JSON %s -> %s (%s)", json_path, labels_out_dir, stats)
    return stats


def encode_bdd_masks_to_3ch(
    drivable_mask: np.ndarray | None,
    lane_mask: np.ndarray | None,
    direct_value: int = 0,
    alternative_value: int = 1,
    lane_bg_value: int = 255,
) -> np.ndarray:
    """Encode raw 2D BDD100K drivable and lane masks into binary (3, H, W) uint8 array.

    Channel order matches SEG_CHANNELS:
        0: drivable_direct
        1: drivable_alternative
        2: lane_marking
    """
    ref_mask = drivable_mask if drivable_mask is not None else lane_mask
    if ref_mask is None:
        raise ValueError("At least one of drivable_mask or lane_mask must be provided.")
    if ref_mask.ndim != 2:
        raise ValueError(f"Expected 2D label mask (H, W), got shape {ref_mask.shape}")

    height, width = ref_mask.shape
    out = np.zeros((len(SEG_CHANNELS), height, width), dtype=np.uint8)

    if drivable_mask is not None:
        if drivable_mask.shape != (height, width):
            raise ValueError(
                f"drivable_mask shape {drivable_mask.shape} != expected {(height, width)}"
            )
        out[0] = (drivable_mask == direct_value).astype(np.uint8)
        out[1] = (drivable_mask == alternative_value).astype(np.uint8)

    if lane_mask is not None:
        if lane_mask.shape != (height, width):
            raise ValueError(f"lane_mask shape {lane_mask.shape} != expected {(height, width)}")
        out[2] = (lane_mask != lane_bg_value).astype(np.uint8)

    return out


def extract_cityscapes_road_mask(label_mask: np.ndarray, road_id: int = 7) -> np.ndarray:
    """Extract binary (H, W) uint8 road mask from a Cityscapes labelIds/trainIds mask."""
    if label_mask.ndim != 2:
        raise ValueError(f"Expected 2D Cityscapes mask (H, W), got shape {label_mask.shape}")
    return (label_mask == road_id).astype(np.uint8)


def create_subset_split(
    items: Sequence[str],
    max_samples: int | None,
    seed: int,
) -> list[str]:
    """Deterministically sample up to max_samples items using a fixed random seed."""
    ordered = sorted(items)
    if max_samples is None or max_samples >= len(ordered):
        return ordered
    if max_samples < 0:
        raise ValueError(f"max_samples must be non-negative, got {max_samples}")

    rng = random.Random(seed)
    sampled = rng.sample(ordered, k=max_samples)
    return sorted(sampled)


def write_split_file(stems: Sequence[str], out_path: Path) -> Path:
    """Write image stems (one per line) to a split file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(stems) + ("\n" if stems else "")
    out_path.write_text(content, encoding="utf-8")
    return out_path


def read_split_file(split_path: Path) -> list[str]:
    """Read image stems from a split file, ignoring blank lines."""
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    return [
        line.strip()
        for line in split_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sample_bdd_poly2d(poly: dict[str, Any], steps_per_curve: int = 8) -> np.ndarray | None:
    """Sample BDD100K poly2d vertices (supporting 'L' lines and 'C' cubic Bezier segments)."""
    raw_verts = poly.get("vertices")
    if not isinstance(raw_verts, list) or len(raw_verts) < 2:
        return None

    pts = np.asarray(raw_verts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2:
        return None

    types = str(poly.get("types", "L" * len(pts)))
    closed = bool(poly.get("closed", False))
    if len(types) != len(pts) or "C" not in types:
        return np.round(pts).astype(np.int32)

    sampled: list[np.ndarray] = [pts[0]]
    idx = 1
    n = len(pts)
    ts = np.linspace(0.0, 1.0, steps_per_curve + 1, dtype=np.float32)[1:, None]

    while idx < n:
        if idx + 2 < n and types[idx : idx + 3] == "CCC":
            p0, p1, p2, p3 = sampled[-1], pts[idx], pts[idx + 1], pts[idx + 2]
            one_minus_t = 1.0 - ts
            curve = (
                (one_minus_t**3) * p0
                + 3.0 * (one_minus_t**2) * ts * p1
                + 3.0 * one_minus_t * (ts**2) * p2
                + (ts**3) * p3
            )
            sampled.extend(list(curve))
            idx += 3
        else:
            sampled.append(pts[idx])
            idx += 1

    if closed and idx == n - 2 and types[-2:] == "CC":
        p0, p1, p2, p3 = sampled[-1], pts[-2], pts[-1], pts[0]
        one_minus_t = 1.0 - ts
        curve = (
            (one_minus_t**3) * p0
            + 3.0 * (one_minus_t**2) * ts * p1
            + 3.0 * one_minus_t * (ts**2) * p2
            + (ts**3) * p3
        )
        sampled.extend(list(curve))

    return np.round(np.asarray(sampled, dtype=np.float32)).astype(np.int32)


def rasterize_bdd_frame_seg_masks(
    labels: list[dict[str, Any]],
    img_width: int = 1280,
    img_height: int = 720,
    lane_thickness: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize BDD100K JSON `drivable area` and `lane` poly2d labels into 2D ID masks.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - drivable_mask: uint8 (H, W) with 0=direct, 1=alternative, 2=background
        - lane_mask: uint8 (H, W) with 1=lane_marking, 255=background
    """
    drivable_mask = np.full((img_height, img_width), 2, dtype=np.uint8)
    lane_mask = np.full((img_height, img_width), 255, dtype=np.uint8)

    for label in labels:
        if not isinstance(label, dict):
            continue
        cat = str(label.get("category", "")).strip().lower()
        polys = label.get("poly2d")
        if not isinstance(polys, list):
            continue

        if cat in {"drivable area", "area/drivable", "area/alternative"}:
            attrs = label.get("attributes") or {}
            area_type = str(attrs.get("areaType", "direct")).strip().lower()
            if cat == "area/alternative" or area_type == "alternative":
                pix_id = 1
            elif area_type == "direct" or cat == "area/drivable":
                pix_id = 0
            else:
                continue

            for poly in polys:
                if not isinstance(poly, dict):
                    continue
                pts = _sample_bdd_poly2d(poly)
                if pts is not None and len(pts) >= 3:
                    cv2.fillPoly(drivable_mask, [pts.reshape(-1, 1, 2)], color=pix_id)

        elif cat == "lane" or cat.startswith("lane/"):
            for poly in polys:
                if not isinstance(poly, dict):
                    continue
                pts = _sample_bdd_poly2d(poly)
                if pts is not None and len(pts) >= 2:
                    closed = bool(poly.get("closed", False))
                    cv2.polylines(
                        lane_mask,
                        [pts.reshape(-1, 1, 2)],
                        isClosed=closed,
                        color=1,
                        thickness=lane_thickness,
                    )

    return drivable_mask, lane_mask


def rasterize_bdd_json_seg_to_dirs(
    json_path: Path,
    images_dir: Path,
    drivable_out_dir: Path,
    lane_out_dir: Path,
    allowed_stems: set[str] | None = None,
    default_size_wh: tuple[int, int] = (1280, 720),
) -> dict[str, int]:
    """Generate PNG drivable and lane masks from BDD100K JSON poly2d annotations."""
    if not json_path.exists():
        raise FileNotFoundError(f"BDD100K JSON not found: {json_path}")

    drivable_out_dir.mkdir(parents=True, exist_ok=True)
    lane_out_dir.mkdir(parents=True, exist_ok=True)

    with json_path.open("r", encoding="utf-8") as handle:
        frames = _extract_bdd_frames(json.load(handle))

    written = 0
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        image_name = frame.get("name")
        if not isinstance(image_name, str) or not image_name:
            continue

        stem = Path(image_name).stem
        if allowed_stems is not None and stem not in allowed_stems:
            continue

        drv_path = drivable_out_dir / f"{stem}.png"
        ln_path = lane_out_dir / f"{stem}.png"
        if drv_path.exists() and ln_path.exists():
            written += 1
            continue

        img_path = images_dir / image_name
        if img_path.exists():
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        else:
            img_w, img_h = default_size_wh

        drv_mask, ln_mask = rasterize_bdd_frame_seg_masks(
            labels=frame.get("labels") or [],
            img_width=img_w,
            img_height=img_h,
        )
        cv2.imwrite(str(drv_path), drv_mask)
        cv2.imwrite(str(ln_path), ln_mask)
        written += 1

    logger.info(
        "Rasterized %d segmentation mask pairs from %s -> (%s, %s)",
        written,
        json_path,
        drivable_out_dir,
        lane_out_dir,
    )
    return {"masks_written": written}
