#!/usr/bin/env python3
"""Chuẩn bị dữ liệu BDD100K cho TV3 (YOLO11) và TV4 (DeepLabV3+).

Lệnh con:
  subset        Chia subset cố định (seed = project.seed) -> data/splits/[<tag>_]{train,val}.txt
  detection     Detection JSON -> YOLO .txt 9 lớp + data.yaml
  segmentation  Mask drivable/lane -> mask 3 kênh (drivable_direct, drivable_alternative, lane_marking)
  all           detection + segmentation

Đường dẫn & seed lấy từ ``configs/paths.yaml`` (paths.bdd_root, project.seed...);
tham số dòng lệnh ghi đè config. Logic nằm trong ``src/data/``.

Quy trình chuẩn (chạy ``scripts/inspect_bdd.py`` TRƯỚC để xác nhận định dạng):
    python scripts/prepare_data.py subset                     # dev: 5000 train / 1000 val
    python scripts/prepare_data.py all --use-splits           # chuyển đổi đúng các ảnh trong split
    python scripts/prepare_data.py subset --tag main --train-size 20000 --val-size 2000
    python scripts/prepare_data.py all --use-splits --tag main
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import splits as split_lib  # noqa: E402
from src.data.bdd_layout import DEFAULT_PATHS_CONFIG, load_data_paths  # noqa: E402
from src.data.converters import DrivableIds  # noqa: E402
from src.data.prepare import LINK_MODES, PrepareOptions, prepare_detection, prepare_segmentation  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["subset", "detection", "segmentation", "all"])
    ap.add_argument("--config", type=Path, default=DEFAULT_PATHS_CONFIG, help="paths.yaml")
    ap.add_argument("--bdd-root", type=Path, help="Ghi đè paths.bdd_root")
    ap.add_argument("--out", type=Path, help="Ghi đè paths.processed_root (mặc định data/processed)")
    ap.add_argument("--splits-dir", type=Path, help="Ghi đè paths.splits_dir (mặc định data/splits)")
    ap.add_argument("--seed", type=int, help="Ghi đè project.seed (mặc định 42)")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--tag", help="Tiền tố file split, vd 'main' -> main_train.txt")

    g = ap.add_argument_group("subset")
    g.add_argument("--train-size", type=int, default=5000, help="Mặc định 5000 (bộ dev, docs/03 §6)")
    g.add_argument("--val-size", type=int, default=1000, help="Mặc định 1000 (bộ dev, docs/03 §6)")
    g.add_argument("--require", nargs="+", default=list(split_lib.REQUIRE_CHOICES), choices=split_lib.REQUIRE_CHOICES)
    g.add_argument("--no-stratify", action="store_true", help="Không phân tầng weather × timeofday")

    g = ap.add_argument_group("chuyển đổi")
    g.add_argument("--use-splits", action="store_true", help="Chỉ chuyển các ảnh có trong file split")
    g.add_argument("--limit", type=int, help="Chỉ lấy N ảnh / split (ngẫu nhiên theo seed) — để thử nhanh")
    g.add_argument("--link-mode", choices=LINK_MODES, default="auto", help="auto = hardlink, lỗi thì copy")
    g.add_argument("--skip-empty", action="store_true", help="Detection: bỏ ảnh không có box")
    g.add_argument("--mask-size", type=int, nargs=2, metavar=("W", "H"), help="Kích thước mask xuất ra")
    g.add_argument("--lane-thickness", type=int, default=8, help="Độ dày vạch khi vẽ từ JSON (px @1280x720)")
    g.add_argument("--lane-background", type=int, choices=[0, 255], help="Nền mask lane PNG (mặc định tự nhận diện)")
    g.add_argument("--drivable-ids", type=int, nargs=3, metavar=("DIRECT", "ALT", "BG"), default=[0, 1, 2],
                   help="Giá trị pixel mask drivable — lấy theo kết quả inspect_bdd.py")
    g.add_argument("--require-lane", action="store_true", help="Segmentation: chỉ lấy ảnh có nhãn lane")
    return ap


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    cfg = load_data_paths(args.config)
    bdd_root = args.bdd_root or cfg.bdd_root
    if bdd_root is None:
        raise SystemExit("Chưa có bdd_root: điền paths.bdd_root trong configs/paths.yaml hoặc truyền --bdd-root")
    seed = cfg.seed if args.seed is None else args.seed
    splits_dir = args.splits_dir or cfg.splits_dir

    if args.task == "subset":
        sizes = {"train": args.train_size, "val": args.val_size}
        for split in args.splits:
            chosen, pool = split_lib.make_subset(
                bdd_root, split, sizes.get(split), seed, tuple(args.require), not args.no_stratify
            )
            path = split_lib.split_file(splits_dir, split, args.tag)
            split_lib.write_split(path, chosen, seed, tuple(args.require))
            print(f"[subset] {split}: {len(chosen)}/{pool} ảnh đủ nhãn -> {path}")
        return 0

    subsets: dict[str, set[str]] = {}
    if args.use_splits:
        for split in args.splits:
            path = split_lib.split_file(splits_dir, split, args.tag)
            if not path.exists():
                raise SystemExit(f"Không thấy {path} — chạy 'prepare_data.py subset' trước")
            subsets[split] = set(split_lib.read_split(path))

    opt = PrepareOptions(
        bdd_root=bdd_root,
        out_root=args.out or cfg.processed_root,
        splits=tuple(args.splits),
        subsets=subsets,
        limit=args.limit,
        seed=seed,
        link_mode=args.link_mode,
        skip_empty=args.skip_empty,
        lane_thickness=args.lane_thickness,
        lane_background=args.lane_background,
        mask_size=tuple(args.mask_size) if args.mask_size else None,
        require_lane=args.require_lane,
        drivable_ids=DrivableIds(*args.drivable_ids),
    )
    if args.task in ("detection", "all"):
        print("->", prepare_detection(opt))
    if args.task in ("segmentation", "all"):
        print("->", prepare_segmentation(opt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
