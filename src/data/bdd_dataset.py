"""Dataset + DataLoader BDD100K cho TV3 (detection) và TV4 (segmentation).

Đọc dữ liệu ĐÃ CHUẨN BỊ bởi ``scripts/prepare_data.py``::

    <root>/images/<split>/<stem>.jpg
    <root>/labels/<split>/<stem>.txt     # YOLO, cho detection
    <root>/masks/<split>/<stem>.png      # 3 kênh {0,255}, cho segmentation

Ảnh đọc ra là **BGR uint8** (quy ước nội bộ). Transforms (``src/data/transforms.py``)
nhận/trả dict ``image, mask, boxes, labels`` và tự đổi BGR -> RGB + chuẩn hoá.
Train dùng ``build_train_transforms``; val dùng ``build_eval_transforms`` — cùng
pipeline với ``preprocess_inference``.

Lưu ý TV3: YOLO11 (ultralytics) có dataloader riêng, chỉ cần trỏ vào ``data.yaml``.
``DetectionDataset`` ở đây dùng cho benchmark / evaluate / pipeline end-to-end.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .converters import load_mask, parse_yolo_line, yolo_to_xyxy
from .splits import read_split
from .transforms import _resize_mask, normalize_image

IMG_EXTS = (".jpg", ".jpeg", ".png")


def list_images(img_dir: Path) -> list[Path]:
    """Danh sách ảnh (jpg/png) trong thư mục, đã sắp xếp."""
    img_dir = Path(img_dir)
    if not img_dir.exists():
        return []
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)


def read_bgr(path: Path) -> np.ndarray:
    """Đọc ảnh -> BGR uint8 (H, W, 3)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _subset_stems(path: str | Path | None) -> set[str] | None:
    return None if path is None else {Path(n).stem for n in read_split(Path(path))}


class DetectionDataset:
    """BDD detection (YOLO đã chuyển). Trả dict: image, boxes (N,4) xyxy pixel, labels (N,), meta."""

    def __init__(self, root: str | Path, split: str = "train", transforms: Callable | None = None,
                 subset_file: str | Path | None = None):
        self.root = Path(root)
        self.img_dir = self.root / "images" / split
        self.lbl_dir = self.root / "labels" / split
        subset = _subset_stems(subset_file)
        self.items = [p for p in list_images(self.img_dir) if subset is None or p.stem in subset]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path = self.items[idx]
        image = read_bgr(img_path)
        h, w = image.shape[:2]
        boxes, labels = [], []
        lbl_path = self.lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                if line.strip():
                    yb = parse_yolo_line(line)
                    boxes.append(yolo_to_xyxy(yb, w, h))
                    labels.append(yb.class_id)
        bx = np.asarray(boxes, np.float32).reshape(-1, 4)
        lb = np.asarray(labels, np.int64)
        if self.transforms is not None:
            out = self.transforms(image=image, boxes=bx, labels=lb)
            image, bx, lb = out["image"], out["boxes"], out["labels"]
        return {"image": image, "boxes": bx, "labels": lb, "meta": {"path": str(img_path), "orig_size": (h, w)}}


class SegmentationDataset:
    """BDD segmentation (mask 3 kênh). Trả dict: image, mask (H,W,3) uint8 {0,1}, meta."""

    def __init__(self, root: str | Path, split: str = "train", transforms: Callable | None = None,
                 subset_file: str | Path | None = None):
        self.root = Path(root)
        self.img_dir = self.root / "images" / split
        self.mask_dir = self.root / "masks" / split
        subset = _subset_stems(subset_file)
        self.items = [
            p for p in list_images(self.img_dir)
            if (self.mask_dir / f"{p.stem}.png").exists() and (subset is None or p.stem in subset)
        ]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_path = self.items[idx]
        image = read_bgr(img_path)
        mask = load_mask(str(self.mask_dir / f"{img_path.stem}.png"), binary=True)
        orig = image.shape[:2]
        if mask.shape[:2] != orig:  # mask xuất bằng --mask-size: đưa về cỡ ảnh (Albumentations yêu cầu khớp)
            mask = _resize_mask(mask, orig[1], orig[0])
        if self.transforms is not None:
            out = self.transforms(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return {"image": image, "mask": mask, "meta": {"path": str(img_path), "orig_size": orig}}


# ---------------------------------------------------------------------------
# Torch glue (chỉ import torch khi gọi)
# ---------------------------------------------------------------------------
def _to_chw_float(image: np.ndarray):
    import torch

    if image.dtype == np.uint8:  # chưa qua transforms (BGR) -> đổi RGB + chuẩn hoá giống eval
        image = normalize_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))


def detection_collate(batch: list[dict[str, Any]]):
    import torch

    images = torch.stack([_to_chw_float(b["image"]) for b in batch])
    targets = [
        {"boxes": torch.from_numpy(b["boxes"]), "labels": torch.from_numpy(b["labels"]), "meta": b["meta"]}
        for b in batch
    ]
    return images, targets


def segmentation_collate(batch: list[dict[str, Any]]):
    """Trả images (B,3,H,W) float và masks (B,3,H,W) float — giữ 255 để loss tự bỏ."""
    import torch

    images = torch.stack([_to_chw_float(b["image"]) for b in batch])
    masks = torch.stack([torch.from_numpy(np.ascontiguousarray(b["mask"].transpose(2, 0, 1))) for b in batch]).float()
    return images, masks


def build_dataloader(dataset, batch_size: int = 8, shuffle: bool = True, num_workers: int = 2, **kw):
    """DataLoader với collate phù hợp. Windows: gọi trong ``if __name__ == "__main__":`` khi num_workers > 0."""
    from torch.utils.data import DataLoader

    collate = detection_collate if isinstance(dataset, DetectionDataset) else segmentation_collate
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate, pin_memory=True, drop_last=shuffle, **kw)
