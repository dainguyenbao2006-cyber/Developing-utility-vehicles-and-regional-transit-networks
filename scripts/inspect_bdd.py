#!/usr/bin/env python3
"""Kiểm tra (EDA) bộ BDD100K trước khi chuyển đổi.

Báo cáo:
  * Cấu trúc thư mục: đủ / thiếu thư mục ảnh, JSON detection, mask drivable, JSON lane
  * Tên class GỐC trong JSON (detection + lane) — đối chiếu với 9 lớp chuẩn
  * Giá trị pixel trong mask drivable (kỳ vọng {0, 1, 2}) và kích thước mask
  * Số frame trong JSON, số ảnh có/thiếu trên đĩa
  * Phân bố lớp detection (cả lớp bị bỏ), box lỗi, box nhỏ/vừa/lớn (chuẩn COCO)
  * Phân bố thời tiết / thời điểm / bối cảnh (để chọn video demo ngày/đêm/mưa)
  * Độ phủ nhãn segmentation (drivable mask, lane)

Logic nằm ở ``src/data/inspection.py``; script này chỉ đọc tham số và in báo cáo.

Ví dụ:
    python scripts/inspect_bdd.py --split val                    # bdd_root lấy từ configs/paths.yaml
    python scripts/inspect_bdd.py --split train val --out outputs/bdd_inspect.json
    python scripts/inspect_bdd.py --bdd-root D:/datasets/bdd100k --split val   # ghi đè config

Mã thoát 1 nếu thiếu thành phần cấu trúc -> dùng được trong CI.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.bdd_layout import DEFAULT_PATHS_CONFIG, load_data_paths  # noqa: E402
from src.data.inspection import inspect_split  # noqa: E402


def print_report(r: dict) -> None:
    print(f"\n===== BDD100K / {r['split']} =====")
    for k in ("det_json", "images_dir", "num_frames", "images_on_disk", "frames_missing_image", "images_without_label"):
        if k in r:
            print(f"{k:24s}: {r[k]}")
    print(f"{'structure_ok':24s}: {r['structure_ok']}")
    print("\n-- Tên class gốc (detection JSON) --")
    print(" ", r.get("raw_det_categories"))
    print("\n-- Detection --")
    total = sum(r["class_counts"].values()) or 1
    for c, n in r["class_counts"].items():
        print(f"  {c:16s} {n:>9d}  {100 * n / total:5.1f}%")
    if r["dropped_class_counts"]:
        print(f"  (bỏ qua) {r['dropped_class_counts']}")
    print(f"  invalid_boxes       : {r['invalid_boxes']}")
    print(f"  box_size            : {r['box_size']}")
    print(f"  boxes_per_image     : {r.get('boxes_per_image')}")
    print(f"  frames_without_box  : {r['frames_without_box']}")
    print("\n-- Điều kiện --")
    for k in ("weather", "timeofday", "scene"):
        print(f"  {k:10s}: {r[k]}")
    print("\n-- Segmentation --")
    for k in ("drivable_mask_dir", "num_drivable_masks", "drivable_pixel_values", "drivable_mask_shapes",
              "drivable_mask_channels", "drivable_pixel_ratio", "lane_mask_dir", "num_lane_masks",
              "lane_pixel_values", "lane_background_value", "lane_json", "lane_frames", "lane_frames_with_lane",
              "raw_lane_categories"):
        if k in r:
            print(f"  {k:22s}: {r[k]}")


def _print_warnings(r: dict) -> None:
    print("\n-- Cảnh báo --")
    for w in r["warnings"] or ["(không có)"]:
        print("  !", w)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=DEFAULT_PATHS_CONFIG, help="paths.yaml (mặc định configs/paths.yaml)")
    ap.add_argument("--bdd-root", type=Path, help="Ghi đè paths.bdd_root trong config")
    ap.add_argument("--split", nargs="+", default=["val"], help="train / val (có thể nhiều)")
    ap.add_argument("--no-image-check", action="store_true", help="Bỏ bước đối chiếu ảnh trên đĩa")
    ap.add_argument("--sample-masks", type=int, default=200, help="Số mask drivable/lane lấy mẫu để kiểm tra pixel")
    ap.add_argument("--out", type=Path, help="Ghi báo cáo JSON ra file")
    args = ap.parse_args(argv)
    bdd_root = args.bdd_root or load_data_paths(args.config).bdd_root
    if bdd_root is None:
        ap.error("Chưa có bdd_root: điền paths.bdd_root trong configs/paths.yaml hoặc truyền --bdd-root")

    reports = []
    for sp in args.split:
        r = inspect_split(bdd_root, sp, not args.no_image_check, args.sample_masks)
        print_report(r)
        _print_warnings(r)
        reports.append(r)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nĐã ghi báo cáo: {args.out}")
    return 0 if all(r["structure_ok"] for r in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
