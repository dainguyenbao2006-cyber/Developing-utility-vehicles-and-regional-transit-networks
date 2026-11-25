"""Cityscapes val cho đánh giá Cross-Dataset của TV4 (spec §7).

Đọc THẲNG dữ liệu gốc (không cần bước chuyển đổi)::

    <root>/leftImg8bit/val/<city>/<stem>_leftImg8bit.png     (2048×1024)
    <root>/gtFine/val/<city>/<stem>_gtFine_labelIds.png

Cityscapes chỉ có lớp ``road`` (labelId 7), KHÔNG tách làn hiện tại / làn bên.
Theo spec, khi đánh giá::

    Pred_Road = Pred_Direct OR Pred_Alternative
    IoU(Pred_Road, GT_road)

Kênh ``lane_marking`` không đánh giá được trên Cityscapes (không có nhãn).

Ví dụ (TV4)::

    ds = CityscapesRoadDataset("data/cityscapes", transforms=build_eval_transforms(640, 384))
    meter = RoadIoU()
    for item in ds:
        probs = model(item["image"])            # (3, H, W) sigmoid
        meter.update(probs, item["road"])
    print(meter.compute())
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

CITYSCAPES_ROAD_LABEL_ID = 7


def labelids_to_road(label_ids: np.ndarray) -> np.ndarray:
    """labelIds (H, W) -> road mask uint8 (H, W) {0, 1}."""
    if label_ids.ndim == 3:
        label_ids = label_ids[..., 0]
    return (label_ids == CITYSCAPES_ROAD_LABEL_ID).astype(np.uint8)


def road_from_prediction(pred: np.ndarray, threshold: float | tuple[float, float] = 0.5, channels_first: bool = True) -> np.ndarray:
    """Dự đoán 3 kênh (xác suất sigmoid hoặc nhị phân) -> road = direct OR alternative.

    ``threshold`` có thể là 1 số hoặc (thr_direct, thr_alternative) — TV4 tối ưu ngưỡng từng kênh.
    """
    p = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
    if not channels_first:
        p = np.moveaxis(p, -1, 0)
    t0, t1 = (threshold, threshold) if np.isscalar(threshold) else threshold
    return ((p[0] > t0) | (p[1] > t1)).astype(np.uint8)


class RoadIoU:
    """Cộng dồn intersection/union qua toàn bộ tập (không lấy trung bình từng ảnh)."""

    def __init__(self, threshold: float | tuple[float, float] = 0.5):
        """``threshold``: ngưỡng chung hoặc (direct, alternative) — lấy từ thresholds của checkpoint TV4."""
        self.threshold = threshold
        self.inter = 0
        self.union = 0

    def update(self, pred: np.ndarray, road_gt: np.ndarray, channels_first: bool = True) -> None:
        """Cộng dồn 1 ảnh: ``pred`` 3 kênh (xác suất hoặc nhị phân), ``road_gt`` (H, W)."""
        pr = road_from_prediction(pred, self.threshold, channels_first).astype(bool)
        gt = (road_gt.detach().cpu().numpy() if hasattr(road_gt, "detach") else np.asarray(road_gt)).squeeze().astype(bool)
        if pr.shape != gt.shape:
            raise ValueError(f"Kích thước dự đoán {pr.shape} khác ground truth {gt.shape}")
        self.inter += int(np.logical_and(pr, gt).sum())
        self.union += int(np.logical_or(pr, gt).sum())

    def compute(self) -> float:
        """IoU road trên toàn tập; NaN nếu chưa có dữ liệu."""
        return self.inter / self.union if self.union else float("nan")


class CityscapesRoadDataset:
    """Cityscapes val (đọc thẳng dữ liệu gốc). Trả dict: image, road (H, W) {0,1}, meta.

    Nên truyền ``transforms=build_eval_transforms(W, H)`` — cùng pipeline với val BDD.
    """

    def __init__(self, root: str | Path, split: str = "val", transforms: Callable | None = None, limit: int | None = None):
        self.root = Path(root)
        img_root = self.root / "leftImg8bit" / split
        gt_root = self.root / "gtFine" / split
        if not img_root.exists():
            raise FileNotFoundError(f"Không thấy {img_root}")
        self.items: list[tuple[Path, Path]] = []
        for img in sorted(img_root.glob("*/*_leftImg8bit.png")):
            stem = img.name.replace("_leftImg8bit.png", "")
            gt = gt_root / img.parent.name / f"{stem}_gtFine_labelIds.png"
            if gt.exists():
                self.items.append((img, gt))
        if limit is not None:
            self.items = self.items[:limit]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        img_path, gt_path = self.items[idx]
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)  # BGR nội bộ; transforms tự đổi RGB
        lid = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        if image is None or lid is None:
            raise FileNotFoundError(img_path)
        road = labelids_to_road(lid)
        if self.transforms is not None:
            out = self.transforms(image=image, mask=road)
            image, road = out["image"], out["mask"]
        return {"image": image, "road": road, "meta": {"path": str(img_path), "dataset": "cityscapes"}}
