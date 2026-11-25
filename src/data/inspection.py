"""Kiểm tra dữ liệu BDD100K thực tế trước khi chuyển đổi (docs/03, CLAUDE.md #7).

Xác minh: cấu trúc thư mục, tên category GỐC trong JSON, giá trị pixel trong mask
drivable / lane, phân bố lớp và điều kiện (weather / timeofday / scene).
Mọi điểm lệch so với docs/03 được ghi vào ``report["warnings"]``.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.utils.constants import DET_CLASS_TO_ID, DET_CLASSES

from . import bdd_layout
from .converters import BDD_IGNORED_CATEGORIES, BDD_IMAGE_SIZE, DEFAULT_DRIVABLE_IDS, normalize_category

EXPECTED_DRIVABLE_VALUES = {DEFAULT_DRIVABLE_IDS.direct, DEFAULT_DRIVABLE_IDS.alternative, DEFAULT_DRIVABLE_IDS.background}
EXPECTED_LANE_BACKGROUNDS = {0, 255}
SMALL_AREA, MEDIUM_AREA = 32**2, 96**2  # ngưỡng box nhỏ/vừa theo chuẩn COCO


def inspect_split(root: Path, split: str, check_images: bool = True, sample_masks: int = 200) -> dict:
    """Kiểm tra 1 split BDD100K, trả về báo cáo dạng dict (kèm ``warnings`` và ``structure_ok``)."""
    report: dict = {"split": split, "warnings": []}
    warn = report["warnings"].append

    det_path = bdd_layout.det_json(root, split)
    frames = bdd_layout.load_frames(det_path)
    report["det_json"] = str(det_path) if det_path else None
    report["num_frames"] = len(frames)

    img_dir = bdd_layout.images_dir(root, split)
    report["images_dir"] = str(img_dir) if img_dir else None
    if check_images and img_dir is not None:
        on_disk = {p.name for p in img_dir.iterdir() if p.suffix.lower() == ".jpg"}
        names = {f["name"] for f in frames}
        report["images_on_disk"] = len(on_disk)
        report["frames_missing_image"] = len(names - on_disk)
        report["images_without_label"] = len(on_disk - names)

    raw_categories: Counter = Counter()
    cls_counter: Counter = Counter()
    dropped: Counter = Counter()
    size_counter: Counter = Counter()
    weather: Counter = Counter()
    timeofday: Counter = Counter()
    scene: Counter = Counter()
    boxes_per_img: list[int] = []
    invalid = 0
    empty_frames = 0

    for fr in frames:
        attrs = fr.get("attributes") or {}
        weather[attrs.get("weather", "undefined")] += 1
        timeofday[attrs.get("timeofday", "undefined")] += 1
        scene[attrs.get("scene", "undefined")] += 1
        n = 0
        for lab in fr.get("labels") or []:
            raw_categories[lab.get("category", "<none>")] += 1
            b = lab.get("box2d")
            if not b:
                continue
            cat = normalize_category(lab.get("category", ""))
            w, h = float(b["x2"]) - float(b["x1"]), float(b["y2"]) - float(b["y1"])
            if w <= 1 or h <= 1:
                invalid += 1
                continue
            if cat not in DET_CLASS_TO_ID:
                dropped[cat] += 1
                continue
            cls_counter[cat] += 1
            area = w * h
            size_counter["small" if area < SMALL_AREA else "medium" if area < MEDIUM_AREA else "large"] += 1
            n += 1
        boxes_per_img.append(n)
        if n == 0:
            empty_frames += 1

    report["raw_det_categories"] = dict(raw_categories.most_common())
    unknown = sorted(c for c in dropped if c not in BDD_IGNORED_CATEGORIES)
    if unknown:
        warn(f"Category lạ không thuộc 9 lớp và không nằm trong danh sách bỏ qua: {unknown}")
    missing_cls = [c for c in DET_CLASSES if c not in cls_counter]
    if frames and missing_cls:
        warn(f"Không thấy lớp chuẩn trong JSON: {missing_cls} — kiểm tra tên class / alias")
    report["class_counts"] = dict(cls_counter.most_common())
    report["dropped_class_counts"] = dict(dropped.most_common())
    report["invalid_boxes"] = invalid
    report["box_size"] = dict(size_counter)
    report["frames_without_box"] = empty_frames
    if boxes_per_img:
        arr = np.asarray(boxes_per_img)
        report["boxes_per_image"] = {"mean": round(float(arr.mean()), 2), "max": int(arr.max()), "median": float(np.median(arr))}
    report["weather"] = dict(weather.most_common())
    report["timeofday"] = dict(timeofday.most_common())
    report["scene"] = dict(scene.most_common())

    # --- Segmentation coverage ---
    mask_dir = bdd_layout.drivable_mask_dir(root, split)
    report["drivable_mask_dir"] = str(mask_dir) if mask_dir else None
    if mask_dir is not None:
        masks = sorted(mask_dir.glob("*.png"))
        report["num_drivable_masks"] = len(masks)
        if masks and sample_masks > 0:
            rng = np.random.default_rng(42)
            pick = rng.choice(len(masks), size=min(sample_masks, len(masks)), replace=False)
            ratios = np.zeros(3)
            values: set[int] = set()
            shapes: Counter = Counter()
            channels: Counter = Counter()
            for i in pick:
                m = cv2.imread(str(masks[i]), cv2.IMREAD_UNCHANGED)
                if m is None:
                    continue
                channels[1 if m.ndim == 2 else m.shape[2]] += 1
                if m.ndim == 3:
                    m = m[..., 0]
                shapes[f"{m.shape[1]}x{m.shape[0]}"] += 1
                values |= {int(v) for v in np.unique(m)}
                ratios += np.bincount(m.ravel(), minlength=256)[:3] / m.size
            ratios /= len(pick)
            report["drivable_pixel_values"] = sorted(values)
            report["drivable_mask_shapes"] = dict(shapes)
            report["drivable_mask_channels"] = dict(channels)
            if not values <= EXPECTED_DRIVABLE_VALUES:
                warn(f"Mask drivable có giá trị lạ {sorted(values - EXPECTED_DRIVABLE_VALUES)} "
                     f"(kỳ vọng 0=direct, 1=alternative, 2=background) — có thể đang dùng mask màu thay vì mask id")
            exp_shape = f"{BDD_IMAGE_SIZE[0]}x{BDD_IMAGE_SIZE[1]}"
            if set(shapes) - {exp_shape}:
                warn(f"Mask drivable có kích thước khác {exp_shape}: {dict(shapes)}")
            report["drivable_pixel_ratio"] = {
                "direct": round(float(ratios[0]), 4),
                "alternative": round(float(ratios[1]), 4),
                "background": round(float(ratios[2]), 4),
            }

    lane_dir = bdd_layout.lane_mask_dir(root, split)
    report["lane_mask_dir"] = str(lane_dir) if lane_dir else None
    if lane_dir is not None:
        lmasks = sorted(lane_dir.glob("*.png"))
        report["num_lane_masks"] = len(lmasks)
        bg_counter: Counter = Counter()
        lvals: set[int] = set()
        rng = np.random.default_rng(42)
        for i in rng.choice(len(lmasks), size=min(sample_masks, len(lmasks)), replace=False) if lmasks else []:
            m = cv2.imread(str(lmasks[i]), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            lvals |= {int(v) for v in np.unique(m)}
            bg_counter[int(np.bincount(m.ravel(), minlength=256).argmax())] += 1
        report["lane_pixel_values"] = sorted(lvals)
        report["lane_background_value"] = dict(bg_counter)
        if bg_counter and set(bg_counter) - EXPECTED_LANE_BACKGROUNDS:
            warn(f"Nền mask lane không phải 0/255: {dict(bg_counter)}")

    lane_path = bdd_layout.lane_json(root, split)
    report["lane_json"] = str(lane_path) if lane_path else None
    if lane_path is not None:
        lane_frames = bdd_layout.load_frames(lane_path)
        with_lane = sum(
            any(str(lb.get("category", "")).lower().startswith("lane") for lb in (f.get("labels") or []))
            for f in lane_frames
        )
        lane_cats: Counter = Counter(
            lb.get("category", "<none>") for f in lane_frames for lb in (f.get("labels") or [])
        )
        report["lane_frames"] = len(lane_frames)
        report["lane_frames_with_lane"] = with_lane
        report["raw_lane_categories"] = dict(lane_cats.most_common())
        if lane_frames and not with_lane:
            warn("JSON lane không có category nào bắt đầu bằng 'lane' — kiểm tra format")

    for key, label in (("det_json", "JSON detection"), ("images_dir", "thư mục ảnh"), ("drivable_mask_dir", "mask drivable")):
        if not report.get(key):
            warn(f"Thiếu {label} cho split '{split}'")
    if not report.get("lane_json") and not report.get("lane_mask_dir"):
        warn(f"Thiếu nhãn lane (JSON hoặc mask) cho split '{split}'")
    if report.get("frames_missing_image"):
        warn(f"{report['frames_missing_image']} frame trong JSON không có ảnh trên đĩa")
    report["structure_ok"] = not any(w.startswith("Thiếu") for w in report["warnings"])
    return report
