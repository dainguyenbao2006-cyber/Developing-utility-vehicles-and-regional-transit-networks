#!/usr/bin/env python3
"""Tải / giải nén / xác minh BDD100K về đúng cấu trúc thư mục của dự án.

BDD100K yêu cầu ĐĂNG NHẬP ở https://bdd-data.berkeley.edu (không có link tải
trực tiếp công khai), nên script hỗ trợ 3 cách:

  A) Tải thủ công (khuyên dùng) — đăng nhập trang trên, tải các file zip bên dưới
     vào ``data/raw/``, rồi chạy:

        python data/dowload_data_bdd100k.py --zips-dir data/raw

  B) Mirror nội bộ của nhóm (Google Drive công khai / server riêng...) — đặt các zip
     cùng tên lên 1 URL gốc rồi chạy:

        python data/dowload_data_bdd100k.py --base-url https://<mirror>/bdd100k

  C) Kaggle (kagglehub, cần ~/.kaggle/kaggle.json) — nhóm tự chọn 1 bộ BDD100K trên
     Kaggle có đủ det_20 + drivable + lane, rồi:

        python data/dowload_data_bdd100k.py --kaggle <owner>/<dataset>

     Bản Kaggle do bên thứ ba đăng lại, cấu trúc có thể khác -> script copy về
     data/bdd100k nếu nhận ra layout chuẩn, nếu không thì in đường dẫn để xử lý tay.
     BẮT BUỘC chạy scripts/inspect_bdd.py sau đó.

Các file zip cần (tên chuẩn trên trang BDD100K):

    100k_images_train.zip               ~5.3 GB  (ảnh train 70k)
    100k_images_val.zip                 ~0.8 GB  (ảnh val 10k)
    bdd100k_det_20_labels_trainval.zip            (Detection 2020)
    bdd100k_drivable_labels_trainval.zip          (Drivable maps)
    bdd100k_lane_labels_trainval.zip              (Lane marking)

Chỉ cần val để thử nhanh: ``--parts images_val det drivable lane``.

Kết quả (sau đó chạy ``scripts/inspect_bdd.py`` để kiểm tra chi tiết)::

    data/bdd100k/images/100k/{train,val}/*.jpg
    data/bdd100k/labels/det_20/det_{train,val}.json
    data/bdd100k/labels/drivable/masks/{train,val}/*.png
    data/bdd100k/labels/lane/polygons/lane_{train,val}.json
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

PARTS: dict[str, str] = {
    "images_train": "100k_images_train.zip",
    "images_val": "100k_images_val.zip",
    "det": "bdd100k_det_20_labels_trainval.zip",
    "drivable": "bdd100k_drivable_labels_trainval.zip",
    "lane": "bdd100k_lane_labels_trainval.zip",
}

# Đường dẫn kỳ vọng (tương đối với <out>/bdd100k) cho từng phần
EXPECTED: dict[str, list[str]] = {
    "images_train": ["images/100k/train"],
    "images_val": ["images/100k/val"],
    "det": ["labels/det_20/det_train.json", "labels/det_20/det_val.json"],
    "drivable": ["labels/drivable/masks/train", "labels/drivable/masks/val"],
    "lane": ["labels/lane/polygons/lane_train.json", "labels/lane/polygons/lane_val.json"],
}


def download(url: str, dst: Path, chunk: int = 1 << 20) -> None:
    """Tải có hỗ trợ tải tiếp (resume) nếu server hỗ trợ Range."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    done = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req) as r:
        if done and r.status != 206:  # server không hỗ trợ resume -> tải lại từ đầu
            done = 0
        total = int(r.headers.get("Content-Length", 0)) + done
        with open(tmp, "ab" if done else "wb") as f:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    print(f"\r  {dst.name}: {done / 1e9:.2f}/{total / 1e9:.2f} GB", end="", flush=True)
    print()
    tmp.rename(dst)


def extract(zip_path: Path, out_root: Path) -> None:
    """Giải nén vào ``out_root``. Zip BDD đã chứa sẵn thư mục ``bdd100k/`` ở gốc."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        top = {n.split("/", 1)[0] for n in names if n.strip("/")}
        target = out_root if top == {"bdd100k"} else out_root / "bdd100k"
        print(f"  giải nén {zip_path.name} ({len(names)} file) -> {target}")
        zf.extractall(target)


def normalize_layout(bdd: Path) -> None:
    """Một số bản zip ảnh giải nén ra ``bdd100k/images/100k/train`` lồng thêm 1 cấp
    hoặc ra ``bdd100k/100k/train`` — đưa về đúng ``images/100k/<split>``."""
    for split in ("train", "val", "test"):
        want = bdd / "images" / "100k" / split
        if want.exists() and any(want.glob("*.jpg")):
            continue
        for alt in (bdd / "100k" / split, want / split):
            if not alt.exists():
                continue
            tmp = bdd / f"_tmp_{split}"
            shutil.move(str(alt), str(tmp))  # tách ra trước (alt có thể nằm trong want)
            if want.exists():
                shutil.rmtree(want)
            want.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp), str(want))
            print(f"  sửa cấu trúc: {alt} -> {want}")
            break


def from_kaggle(handle: str, bdd: Path) -> None:
    """Tải bằng kagglehub rồi tìm thư mục có layout chuẩn (images/100k + labels) để copy về ``bdd``."""
    import kagglehub  # có trong requirements.txt

    src = Path(kagglehub.dataset_download(handle))
    print(f"  kagglehub tải về: {src}")
    for cand in [src, *sorted(p for p in src.rglob("*") if p.is_dir())]:
        if (cand / "images" / "100k").is_dir() and (cand / "labels").is_dir():
            if cand.resolve() != bdd.resolve():
                print(f"  copy {cand} -> {bdd} (có thể lâu)")
                shutil.copytree(cand, bdd, dirs_exist_ok=True)
            return
    print("  !! Không nhận ra layout chuẩn (images/100k + labels) trong bản Kaggle này.")
    print(f"     Xem thư mục {src}, sắp xếp lại về {bdd} theo docs rồi chạy --verify-only.")


def verify(bdd: Path, parts: list[str]) -> bool:
    ok = True
    print("\nKiểm tra cấu trúc:")
    for part in parts:
        for rel in EXPECTED[part]:
            p = bdd / rel
            if p.is_dir():
                n = sum(1 for _ in p.iterdir())
                good = n > 0
                info = f"{n} file"
            else:
                good = p.exists()
                info = f"{p.stat().st_size / 1e6:.1f} MB" if good else "THIẾU"
            ok &= good
            print(f"  [{'OK' if good else '!!'}] {rel:45s} {info}")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("data"), help="Thư mục chứa bdd100k/ (mặc định data/)")
    ap.add_argument("--zips-dir", type=Path, default=Path("data/raw"), help="Nơi chứa/tải các file zip")
    ap.add_argument("--base-url", help="URL gốc của mirror chứa các zip (tuỳ chọn)")
    ap.add_argument("--kaggle", metavar="HANDLE", help="Tải bằng kagglehub, vd owner/bdd100k")
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=list(PARTS))
    ap.add_argument("--delete-zips", action="store_true", help="Xoá zip sau khi giải nén để tiết kiệm ổ")
    ap.add_argument("--verify-only", action="store_true", help="Chỉ kiểm tra cấu trúc, không tải/giải nén")
    args = ap.parse_args(argv)

    bdd = args.out / "bdd100k"
    if args.kaggle and not args.verify_only:
        from_kaggle(args.kaggle, bdd)
    elif not args.verify_only:
        missing = []
        for part in args.parts:
            zp = args.zips_dir / PARTS[part]
            if not zp.exists() and args.base_url:
                url = f"{args.base_url.rstrip('/')}/{PARTS[part]}"
                print(f"Tải {url}")
                download(url, zp)
            if not zp.exists():
                missing.append(PARTS[part])
                continue
            extract(zp, args.out)
            if args.delete_zips:
                zp.unlink()
        if missing:
            print("\nThiếu các file zip sau trong", args.zips_dir)
            for m in missing:
                print("   -", m)
            print("-> Đăng nhập https://bdd-data.berkeley.edu, tải về thư mục trên rồi chạy lại,")
            print("   hoặc dùng --base-url trỏ tới mirror của nhóm.")
        normalize_layout(bdd)

    ok = verify(bdd, args.parts)
    print("\nHoàn tất." if ok else "\nCấu trúc CHƯA đủ — xem các dòng [!!] ở trên.")
    print("Bước tiếp: python scripts/inspect_bdd.py --bdd-root", bdd, "--split val")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
