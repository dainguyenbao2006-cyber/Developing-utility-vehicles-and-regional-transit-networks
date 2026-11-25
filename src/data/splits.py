"""Chia subset cố định (``project.seed`` = 42) cho TV3 & TV4 (docs/03 §6).

* Chỉ lấy ảnh có ĐỦ nhãn (detection + drivable + lane) -> cùng 1 tập val cho 2 model.
* Phân tầng theo ``weather|timeofday`` để tập nhỏ vẫn đủ ngày/đêm/mưa.
* Chọn trên danh sách ĐÃ SẮP XẾP với RNG có seed -> lặp lại y hệt trên mọi máy.

File split: ``data/splits/[<tag>_]<split>.txt`` — dòng đầu là comment ``#``,
mỗi dòng sau là 1 tên ảnh (vd ``0000f77c-6257be58.jpg``).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from . import bdd_layout

DEFAULT_SEED = 42
REQUIRE_CHOICES = ("det", "drivable", "lane")


def split_file(splits_dir: Path, split: str, tag: str | None = None) -> Path:
    """Đường dẫn file split: ``<splits_dir>/[<tag>_]<split>.txt``."""
    return Path(splits_dir) / (f"{tag}_{split}.txt" if tag else f"{split}.txt")


def read_split(path: Path) -> list[str]:
    """Đọc file split -> danh sách tên ảnh (bỏ dòng trống / comment)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def write_split(path: Path, names: list[str], seed: int, require: tuple[str, ...]) -> None:
    """Ghi file split kèm header ghi seed để truy vết."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# seed={seed} n={len(names)} require={','.join(require)}\n"
    path.write_text(header + "\n".join(names) + "\n", encoding="utf-8")


def labeled_names(root: Path, split: str, require: tuple[str, ...] = REQUIRE_CHOICES) -> list[str]:
    """Tên ảnh có trên đĩa và có đủ các loại nhãn trong ``require``."""
    img_dir = bdd_layout.images_dir(root, split)
    if img_dir is None:
        raise FileNotFoundError(f"Không thấy thư mục ảnh cho split '{split}' trong {root}")
    names = {p.name for p in img_dir.glob("*.jpg")}
    if "det" in require:
        names &= {f["name"] for f in bdd_layout.load_frames(bdd_layout.det_json(root, split))}
    if "drivable" in require:
        names &= bdd_layout.drivable_names(root, split)
    if "lane" in require:
        names &= bdd_layout.lane_names(root, split)
    return sorted(names)


def strata_of(root: Path, split: str) -> dict[str, str]:
    """name -> ``"weather|timeofday"`` lấy từ attributes của JSON detection."""
    out: dict[str, str] = {}
    for f in bdd_layout.load_frames(bdd_layout.det_json(root, split)):
        a = f.get("attributes") or {}
        out[f["name"]] = f"{a.get('weather', 'undefined')}|{a.get('timeofday', 'undefined')}"
    return out


def random_pick(names: list[str], size: int | None, seed: int) -> list[str]:
    """Chọn ngẫu nhiên ``size`` phần tử (có seed, không phụ thuộc thứ tự đầu vào)."""
    names = sorted(names)
    if size is None or size >= len(names):
        return names
    rng = np.random.default_rng(seed)
    return sorted(names[i] for i in rng.choice(len(names), size=size, replace=False))


def stratified_pick(names: list[str], strata: dict[str, str], size: int | None, seed: int) -> list[str]:
    """Chọn ``size`` ảnh chia theo tỉ lệ từng nhóm (largest remainder), có seed."""
    names = sorted(names)
    if size is None or size >= len(names):
        return names
    groups: dict[str, list[str]] = defaultdict(list)
    for n in names:
        groups[strata.get(n, "undefined|undefined")].append(n)
    keys = sorted(groups)
    exact = {k: size * len(groups[k]) / len(names) for k in keys}
    alloc = {k: int(exact[k]) for k in keys}
    rest = size - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (-(exact[k] - alloc[k]), k))[:rest]:
        alloc[k] += 1
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for k in keys:
        g = groups[k]
        if alloc[k]:
            chosen += [g[i] for i in rng.choice(len(g), size=alloc[k], replace=False)]
    return sorted(chosen)


def make_subset(
    root: Path,
    split: str,
    size: int | None,
    seed: int = DEFAULT_SEED,
    require: tuple[str, ...] = REQUIRE_CHOICES,
    stratify: bool = True,
) -> tuple[list[str], int]:
    """Trả về (danh sách được chọn, số ảnh đủ điều kiện)."""
    pool = labeled_names(root, split, require)
    if stratify:
        chosen = stratified_pick(pool, strata_of(root, split), size, seed)
    else:
        chosen = random_pick(pool, size, seed)
    return chosen, len(pool)
