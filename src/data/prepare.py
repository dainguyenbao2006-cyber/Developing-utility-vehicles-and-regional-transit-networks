"""Chạy chuyển đổi BDD100K -> dữ liệu train (gọi từ ``scripts/prepare_data.py``).

Đầu ra (``processed_root`` mặc định ``data/processed``, đã nằm trong .gitignore)::

    bdd100k_det/images/<split>/*.jpg  labels/<split>/*.txt  data.yaml   # TV3 (YOLO11)
    bdd100k_seg/images/<split>/*.jpg  masks/<split>/*.png               # TV4 (DeepLabV3+)

Ảnh được đặt bằng hardlink (không cần quyền admin trên NTFS, không tốn dung
lượng); nếu hardlink lỗi (khác ổ đĩa...) thì tự copy. Không bắt buộc symlink.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.utils.constants import DET_CLASSES, SEG_CHANNELS

from . import bdd_layout
from .converters import (
    BDD_IMAGE_SIZE,
    DEFAULT_DRIVABLE_IDS,
    DrivableIds,
    build_mask,
    frame_to_yolo,
    save_mask,
    yolo_lines,
)

logger = logging.getLogger(__name__)

LINK_MODES = ("auto", "hardlink", "copy", "symlink", "none")


@dataclass
class PrepareOptions:
    """Tham số chung cho bước chuyển đổi."""

    bdd_root: Path
    out_root: Path
    splits: tuple[str, ...] = ("train", "val")
    subsets: dict[str, set[str]] = field(default_factory=dict)  # split -> tên ảnh được chọn
    limit: int | None = None
    seed: int = 42
    link_mode: str = "auto"
    skip_empty: bool = False
    lane_thickness: int = 8
    lane_background: int | None = None
    mask_size: tuple[int, int] | None = None  # (W, H)
    require_lane: bool = False
    drivable_ids: DrivableIds = DEFAULT_DRIVABLE_IDS


def _progress(items: Iterable, total: int, desc: str) -> Iterator:
    try:
        from tqdm import tqdm

        yield from tqdm(items, total=total, desc=desc)
    except ImportError:  # tqdm có trong requirements; dự phòng khi chạy môi trường tối giản
        step = max(1, total // 10)
        for i, x in enumerate(items):
            if i % step == 0:
                logger.info("[%s] %d/%d", desc, i, total)
            yield x


def place_image(src: Path, dst: Path, mode: str = "auto") -> None:
    """Đặt ảnh vào thư mục đích theo ``mode`` (auto = hardlink, lỗi thì copy)."""
    if mode not in LINK_MODES:
        raise ValueError(f"link mode không hợp lệ: {mode}")
    if mode == "none" or dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    else:  # auto
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def _limit_names(names: Iterable[str], limit: int | None, seed: int) -> list[str]:
    names = sorted(names)
    if limit is None or limit >= len(names):
        return names
    rng = np.random.default_rng(seed)
    return sorted(names[i] for i in rng.choice(len(names), size=limit, replace=False))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def write_data_yaml(out: Path, splits: Iterable[str]) -> Path:
    """Sinh ``data.yaml`` chuẩn Ultralytics với ``names`` lấy từ ``DET_CLASSES``."""
    splits = list(splits)
    lines = [f"path: {out.resolve().as_posix()}"]
    for sp in ("train", "val", "test"):
        if sp in splits:
            lines.append(f"{sp}: images/{sp}")
    lines.append(f"nc: {len(DET_CLASSES)}")
    lines.append("names:")
    lines += [f"  {i}: {name}" for i, name in enumerate(DET_CLASSES)]
    path = out / "data.yaml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def prepare_detection(opt: PrepareOptions) -> Path:
    """BDD detection JSON -> ``<out_root>/bdd100k_det`` (YOLO)."""
    out = opt.out_root / "bdd100k_det"
    stats: dict = {}
    for split in opt.splits:
        frames = bdd_layout.load_frames(bdd_layout.det_json(opt.bdd_root, split))
        if not frames:
            logger.warning("[det] Không có nhãn detection cho split '%s', bỏ qua", split)
            continue
        img_dir = bdd_layout.images_dir(opt.bdd_root, split)
        if split in opt.subsets:
            frames = [f for f in frames if f["name"] in opt.subsets[split]]
        keep = set(_limit_names((f["name"] for f in frames), opt.limit, opt.seed))
        frames = [f for f in frames if f["name"] in keep]

        lbl_out = out / "labels" / split
        lbl_out.mkdir(parents=True, exist_ok=True)
        n_img = n_box = n_missing = 0
        for fr in _progress(frames, len(frames), f"det/{split}"):
            src = img_dir / fr["name"] if img_dir else None
            if src is None or not src.exists():
                n_missing += 1
                continue
            boxes = frame_to_yolo(fr, *BDD_IMAGE_SIZE)
            if not boxes and opt.skip_empty:
                continue
            (lbl_out / f"{Path(fr['name']).stem}.txt").write_text(yolo_lines(boxes), encoding="utf-8")
            place_image(src, out / "images" / split / fr["name"], opt.link_mode)
            n_img += 1
            n_box += len(boxes)
        stats[split] = {"images": n_img, "boxes": n_box, "missing_images": n_missing}
        logger.info("[det/%s] %d ảnh, %d box, thiếu ảnh: %d", split, n_img, n_box, n_missing)

    if stats:
        write_data_yaml(out, stats)
        (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def _scale_frame(frame: dict | None, sx: float, sy: float) -> dict | None:
    """Scale toạ độ poly2d khi xuất mask nhỏ hơn 1280x720."""
    if frame is None or (sx == 1 and sy == 1):
        return frame
    labels = []
    for lab in frame.get("labels") or []:
        polys = [
            {**p, "vertices": [[x * sx, y * sy] for x, y in p.get("vertices", [])]} for p in (lab.get("poly2d") or [])
        ]
        labels.append({**lab, "poly2d": polys})
    return {**frame, "labels": labels}


def _read_gray(path: Path | None) -> np.ndarray | None:
    return None if path is None else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def prepare_segmentation(opt: PrepareOptions) -> Path:
    """Mask drivable + lane -> ``<out_root>/bdd100k_seg`` (mask 3 kênh)."""
    out = opt.out_root / "bdd100k_seg"
    stats: dict = {}
    for split in opt.splits:
        root = opt.bdd_root
        img_dir = bdd_layout.images_dir(root, split)
        drv_dir = bdd_layout.drivable_mask_dir(root, split)
        lane_dir = bdd_layout.lane_mask_dir(root, split)
        drv_polys = {} if drv_dir else bdd_layout.index_by_name(bdd_layout.load_frames(bdd_layout.drivable_json(root, split)))
        lane_polys = {} if lane_dir else bdd_layout.index_by_name(bdd_layout.load_frames(bdd_layout.lane_json(root, split)))

        names = bdd_layout.drivable_names(root, split)
        lanes = bdd_layout.lane_names(root, split)
        names = names & lanes if opt.require_lane else names | lanes
        if split in opt.subsets:
            names &= opt.subsets[split]
        selected = _limit_names(names, opt.limit, opt.seed)
        if not selected:
            logger.warning("[seg] Không có nhãn segmentation cho split '%s', bỏ qua", split)
            continue

        m_out = out / "masks" / split
        m_out.mkdir(parents=True, exist_ok=True)
        w, h = opt.mask_size or BDD_IMAGE_SIZE
        sx, sy = w / BDD_IMAGE_SIZE[0], h / BDD_IMAGE_SIZE[1]
        thick = max(1, round(opt.lane_thickness * (sx + sy) / 2))
        n_ok = n_missing = 0
        ratio = np.zeros(len(SEG_CHANNELS))
        for name in _progress(selected, len(selected), f"seg/{split}"):
            src = img_dir / name if img_dir else None
            if src is None or not src.exists():
                n_missing += 1
                continue
            mask = build_mask(
                w,
                h,
                drivable_png=_read_gray(bdd_layout.drivable_mask_path(drv_dir, name) if drv_dir else None),
                drivable_ids=opt.drivable_ids,
                drivable_frame=_scale_frame(drv_polys.get(name), sx, sy),
                lane_png=_read_gray(bdd_layout.lane_mask_path(lane_dir, name) if lane_dir else None),
                lane_background=opt.lane_background,
                lane_frame=_scale_frame(lane_polys.get(name), sx, sy),
                lane_thickness=thick,
            )
            save_mask(str(m_out / f"{Path(name).stem}.png"), mask)
            place_image(src, out / "images" / split / name, opt.link_mode)
            ratio += (mask > 0).reshape(-1, mask.shape[2]).mean(0)
            n_ok += 1
        if n_ok:
            ratio /= n_ok
        stats[split] = {
            "images": n_ok,
            "missing_images": n_missing,
            "mask_size_wh": [w, h],
            "positive_pixel_ratio": {c: round(float(r), 4) for c, r in zip(SEG_CHANNELS, ratio)},
        }
        logger.info("[seg/%s] %d mask, thiếu ảnh: %d, tỉ lệ pixel dương: %s",
                    split, n_ok, n_missing, stats[split]["positive_pixel_ratio"])
    if stats:
        (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out
