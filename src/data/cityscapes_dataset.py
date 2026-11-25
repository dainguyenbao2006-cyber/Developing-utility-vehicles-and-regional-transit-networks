"""PyTorch Dataset for Cityscapes validation cross-dataset road segmentation evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

import albumentations as A
import cv2
import torch
from torch.utils.data import Dataset

from data.converters import extract_cityscapes_road_mask
from data.transforms import apply_seg_transforms, get_seg_val_transforms

logger = logging.getLogger(__name__)


class CityscapesRoadDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Cross-dataset validation dataset returning (image_tensor, road_mask_hw) for Cityscapes."""

    def __init__(
        self,
        cityscapes_root: Path,
        split: str = "val",
        size_hw: tuple[int, int] = (384, 640),
        transform: A.Compose | None = None,
        road_id: int = 7,
        use_train_ids: bool = False,
    ) -> None:
        self.root = Path(cityscapes_root)
        self.images_dir = self.root / "leftImg8bit" / split
        self.gt_dir = self.root / "gtFine" / split
        self.transform = transform if transform is not None else get_seg_val_transforms(size_hw)
        self.road_id = 0 if (use_train_ids and road_id == 7) else road_id
        label_suffix = "_gtFine_labelTrainIds.png" if use_train_ids else "_gtFine_labelIds.png"

        self.samples: list[tuple[Path, Path]] = []
        if self.images_dir.exists() and self.gt_dir.exists():
            for img_path in sorted(self.images_dir.rglob("*_leftImg8bit.png")):
                city = img_path.parent.name
                stem = img_path.name.replace("_leftImg8bit.png", "")
                gt_path = self.gt_dir / city / f"{stem}{label_suffix}"
                if gt_path.exists():
                    self.samples.append((img_path, gt_path))

        logger.info(
            "Initialized CityscapesRoadDataset (%s) with %d samples from %s",
            split,
            len(self.samples),
            self.root,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, gt_path = self.samples[index]
        frame_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise OSError(f"Failed to read Cityscapes image: {img_path}")

        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if gt_mask is None:
            raise OSError(f"Failed to read Cityscapes mask: {gt_path}")

        road_binary = extract_cityscapes_road_mask(gt_mask, road_id=self.road_id)
        image_tensor, mask_tensor = apply_seg_transforms(
            frame_bgr=frame_bgr,
            transform=self.transform,
            mask_chw=road_binary[None, :, :],
        )
        if mask_tensor is None:
            raise RuntimeError("Cityscapes transform unexpectedly returned None mask.")
        return image_tensor, mask_tensor.squeeze(0)
