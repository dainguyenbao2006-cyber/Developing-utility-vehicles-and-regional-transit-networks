"""PyTorch Dataset for BDD100K 3-channel drivable area and lane marking segmentation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.converters import encode_bdd_masks_to_3ch, read_split_file
from data.transforms import apply_seg_transforms, get_seg_val_transforms

logger = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def resolve_bdd_seg_dirs(
    bdd_root: Path,
    split: str,
) -> tuple[Path, Path | None, Path | None]:
    """Auto-resolve BDD100K image, drivable mask, and lane mask directories across layouts."""
    img_candidates = [
        bdd_root / "images" / "100k" / split,
        bdd_root / "bdd100k" / "images" / "100k" / split,
        bdd_root / "bdd100k_images_100k" / "bdd100k" / "images" / "100k" / split,
    ]
    img_dir = next((p for p in img_candidates if p.exists()), img_candidates[0])

    drv_candidates = [
        bdd_root / "labels" / "drivable" / "masks" / split,
        bdd_root / "drivable_maps" / "labels" / split,
        bdd_root / "bdd100k" / "labels" / "drivable" / "masks" / split,
        bdd_root / "bdd100k" / "drivable_maps" / "labels" / split,
    ]
    drv_dir = next((p for p in drv_candidates if p.exists()), None)

    lane_candidates = [
        bdd_root / "labels" / "lane" / "masks" / split,
        bdd_root / "lane" / "labels" / split,
        bdd_root / "labels" / "lane" / "colormaps" / split,
        bdd_root / "bdd100k" / "labels" / "lane" / "masks" / split,
    ]
    lane_dir = next((p for p in lane_candidates if p.exists()), None)
    return img_dir, drv_dir, lane_dir


class BDDSegmentationDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset returning (image_tensor (3, H, W), mask_tensor (3, H, W)) for BDD100K."""

    def __init__(
        self,
        images_dir: Path,
        drivable_dir: Path | None,
        lane_dir: Path | None,
        size_hw: tuple[int, int] = (384, 640),
        transform: A.Compose | None = None,
        stems: Sequence[str] | None = None,
        split_file: Path | None = None,
        direct_value: int = 0,
        alternative_value: int = 1,
        lane_bg_value: int = 255,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.drivable_dir = Path(drivable_dir) if drivable_dir is not None else None
        self.lane_dir = Path(lane_dir) if lane_dir is not None else None
        self.transform = transform if transform is not None else get_seg_val_transforms(size_hw)
        self.direct_value = direct_value
        self.alternative_value = alternative_value
        self.lane_bg_value = lane_bg_value

        if self.drivable_dir is None and self.lane_dir is None:
            raise ValueError("At least one of drivable_dir or lane_dir must be provided.")

        if stems is not None:
            target_stems = list(stems)
        elif split_file is not None:
            target_stems = read_split_file(Path(split_file))
        else:
            target_stems = sorted(
                p.stem
                for p in self.images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )

        self.samples: list[tuple[Path, Path | None, Path | None]] = []
        for stem in target_stems:
            img_path = self._resolve_image(stem)
            if img_path is None:
                continue
            drv_path = self._resolve_mask(self.drivable_dir, stem)
            lane_path = self._resolve_mask(self.lane_dir, stem)
            if drv_path is None and lane_path is None:
                continue
            self.samples.append((img_path, drv_path, lane_path))

        logger.info(
            "Initialized BDDSegmentationDataset with %d samples from %s",
            len(self.samples),
            self.images_dir,
        )

    def _resolve_image(self, stem: str) -> Path | None:
        for ext in IMAGE_EXTS:
            candidate = self.images_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _resolve_mask(mask_dir: Path | None, stem: str) -> Path | None:
        if mask_dir is None:
            return None
        for suffix in (".png", "_drivable_id.png", "_train_id.png"):
            candidate = mask_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, drv_path, lane_path = self.samples[index]
        frame_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise OSError(f"Failed to read image: {img_path}")

        drv_mask: np.ndarray | None = None
        if drv_path is not None:
            drv_mask = cv2.imread(str(drv_path), cv2.IMREAD_UNCHANGED)
            if drv_mask is not None and drv_mask.ndim == 3:
                drv_mask = drv_mask[:, :, 0]

        lane_mask: np.ndarray | None = None
        if lane_path is not None:
            lane_mask = cv2.imread(str(lane_path), cv2.IMREAD_UNCHANGED)
            if lane_mask is not None and lane_mask.ndim == 3:
                lane_mask = lane_mask[:, :, 0]

        mask_chw = encode_bdd_masks_to_3ch(
            drivable_mask=drv_mask,
            lane_mask=lane_mask,
            direct_value=self.direct_value,
            alternative_value=self.alternative_value,
            lane_bg_value=self.lane_bg_value,
        )
        image_tensor, mask_tensor = apply_seg_transforms(
            frame_bgr=frame_bgr,
            transform=self.transform,
            mask_chw=mask_chw,
        )
        if mask_tensor is None:
            raise RuntimeError("Segmentation transform unexpectedly returned None mask.")
        return image_tensor, mask_tensor
