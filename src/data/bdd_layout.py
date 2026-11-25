"""Đường dẫn dữ liệu: đọc ``configs/paths.yaml`` và dò layout thư mục BDD100K.

Không hard-code đường dẫn dataset (CLAUDE.md #8): gốc dữ liệu lấy từ
``cfg.paths.bdd_root`` / ``cfg.paths.cityscapes_root``; seed lấy từ ``project.seed``.
Layout BÊN TRONG bdd_root vẫn phải xác minh bằng ``scripts/inspect_bdd.py``.

Layout chính thức (bản giải nén từ bdd-data.berkeley.edu)::

    <bdd_root>/images/100k/{train,val,test}/*.jpg
    <bdd_root>/labels/det_20/det_{train,val}.json
    <bdd_root>/labels/lane/polygons/lane_{train,val}.json
    <bdd_root>/labels/drivable/masks/{train,val}/*.png
    <bdd_root>/labels/lane/masks/{train,val}/*.png        (nếu có)

Bản cũ (2018) gộp hết vào ``labels/bdd100k_labels_images_{train,val}.json``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATHS_CONFIG = Path("configs/paths.yaml")
DEFAULT_PROCESSED_ROOT = Path("data/processed")
DEFAULT_SPLITS_DIR = Path("data/splits")


@dataclass(frozen=True)
class DataPaths:
    """Đường dẫn + seed cho pipeline dữ liệu."""

    bdd_root: Path | None
    cityscapes_root: Path | None
    processed_root: Path
    splits_dir: Path
    seed: int


def load_data_paths(config: Path | None = DEFAULT_PATHS_CONFIG) -> DataPaths:
    """Đọc ``paths.yaml``. File trống / thiếu key -> None hoặc mặc định.

    Key dùng: ``project.seed``, ``paths.bdd_root``, ``paths.cityscapes_root``,
    ``paths.processed_root`` (tuỳ chọn, mặc định ``data/processed``),
    ``paths.splits_dir`` (tuỳ chọn, mặc định ``data/splits``).
    TODO: chuyển sang ``src.utils.config`` khi TV1 hoàn thiện loader chung.
    """
    raw: dict = {}
    if config is not None and Path(config).exists():
        raw = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    paths = raw.get("paths") or {}
    project = raw.get("project") or {}

    def _p(key: str) -> Path | None:
        return Path(paths[key]) if paths.get(key) else None

    return DataPaths(
        bdd_root=_p("bdd_root"),
        cityscapes_root=_p("cityscapes_root"),
        processed_root=_p("processed_root") or DEFAULT_PROCESSED_ROOT,
        splits_dir=_p("splits_dir") or DEFAULT_SPLITS_DIR,
        seed=int(project.get("seed", 42)),
    )


def _first_existing(cands: list[Path]) -> Path | None:
    for c in cands:
        if c.exists():
            return c
    return None


def images_dir(root: Path, split: str) -> Path | None:
    """Thư mục ảnh của split."""
    return _first_existing([root / "images" / "100k" / split, root / "images" / split, root / "100k" / split])


def det_json(root: Path, split: str) -> Path | None:
    """File JSON detection của split."""
    return _first_existing(
        [
            root / "labels" / "det_20" / f"det_{split}.json",
            root / "labels" / f"det_{split}.json",
            root / "labels" / f"bdd100k_labels_images_{split}.json",
        ]
    )


def lane_json(root: Path, split: str) -> Path | None:
    """File JSON lane (polyline) của split."""
    return _first_existing(
        [
            root / "labels" / "lane" / "polygons" / f"lane_{split}.json",
            root / "labels" / "lane" / f"lane_{split}.json",
            root / "labels" / f"lane_{split}.json",
            root / "labels" / f"bdd100k_labels_images_{split}.json",
        ]
    )


def lane_mask_dir(root: Path, split: str) -> Path | None:
    """Mask lane dạng ảnh (``lane_maps``) — nếu có thì ưu tiên hơn polyline JSON."""
    return _first_existing(
        [
            root / "labels" / "lane" / "masks" / split,
            root / "lane_maps" / "labels" / split,
            root / "lane_maps" / split,
        ]
    )


def lane_mask_path(mask_dir: Path, image_name: str) -> Path | None:
    """Mask lane của 1 ảnh (``<stem>.png`` hoặc ``<stem>_lane_id.png``)."""
    stem = Path(image_name).stem
    return _first_existing([mask_dir / f"{stem}.png", mask_dir / f"{stem}_lane_id.png"])


def drivable_mask_dir(root: Path, split: str) -> Path | None:
    """Thư mục mask drivable id PNG."""
    return _first_existing(
        [
            root / "labels" / "drivable" / "masks" / split,
            root / "drivable_maps" / "labels" / split,
            root / "labels" / "drivable" / split,
        ]
    )


def drivable_json(root: Path, split: str) -> Path | None:
    """File JSON drivable (polygon) của split."""
    return _first_existing(
        [
            root / "labels" / "drivable" / "polygons" / f"drivable_{split}.json",
            root / "labels" / f"drivable_{split}.json",
            root / "labels" / f"bdd100k_labels_images_{split}.json",
        ]
    )


def drivable_mask_path(mask_dir: Path, image_name: str) -> Path | None:
    """Mask drivable có thể tên ``<stem>.png`` hoặc ``<stem>_drivable_id.png``."""
    stem = Path(image_name).stem
    return _first_existing([mask_dir / f"{stem}.png", mask_dir / f"{stem}_drivable_id.png"])


def load_frames(path: Path | None) -> list[dict[str, Any]]:
    """Đọc JSON Scalabel. Hỗ trợ cả dạng list và dạng {"frames": [...]}."""
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("frames", [])
    return list(data)


def index_by_name(frames: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Danh sách frame -> dict theo tên ảnh."""
    return {f["name"]: f for f in frames if "name" in f}


def _png_to_jpg_names(mask_dir: Path, suffix: str) -> set[str]:
    return {p.name.replace(suffix, "").replace(".png", ".jpg") for p in mask_dir.glob("*.png")}


def drivable_names(root: Path, split: str) -> set[str]:
    """Tên ảnh có nhãn drivable (từ mask PNG, nếu không có thì từ JSON polygon)."""
    mdir = drivable_mask_dir(root, split)
    if mdir is not None:
        return _png_to_jpg_names(mdir, "_drivable_id")
    return {f["name"] for f in load_frames(drivable_json(root, split))}


def lane_names(root: Path, split: str) -> set[str]:
    """Tên ảnh có nhãn lane (từ mask PNG, nếu không có thì từ JSON polyline)."""
    ldir = lane_mask_dir(root, split)
    if ldir is not None:
        return _png_to_jpg_names(ldir, "_lane_id")
    return {f["name"] for f in load_frames(lane_json(root, split))}
