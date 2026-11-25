"""Shared Albumentations preprocessing and augmentation pipelines for segmentation."""

from __future__ import annotations

import os

# Disable remote PyPI version lookup warnings in offline/sandboxed environments.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from utils.constants import IMAGENET_MEAN, IMAGENET_STD


def get_seg_train_transforms(size_hw: tuple[int, int]) -> A.Compose:
    """Build training augmentation pipeline for 3-channel road/lane segmentation."""
    height, width = size_hw
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.HueSaturationValue(p=0.3),
            A.RandomShadow(p=0.2),
            A.MotionBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(transpose_mask=True),
        ]
    )


def get_seg_val_transforms(size_hw: tuple[int, int]) -> A.Compose:
    """Build deterministic preprocessing pipeline shared between validation and inference."""
    height, width = size_hw
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(transpose_mask=True),
        ]
    )


def apply_seg_transforms(
    frame_bgr: np.ndarray,
    transform: A.Compose,
    mask_chw: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Convert BGR uint8 image (and optional (C, H, W) mask) using an Albumentations Compose."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR image (H, W, 3), got shape {frame_bgr.shape}")

    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if mask_chw is None:
        transformed = transform(image=image_rgb)
        return transformed["image"], None

    if mask_chw.ndim != 3:
        raise ValueError(f"Expected mask in (C, H, W) layout, got shape {mask_chw.shape}")
    mask_hwc = np.transpose(mask_chw, (1, 2, 0))
    transformed = transform(image=image_rgb, mask=mask_hwc)
    mask_tensor = transformed["mask"].float()
    return transformed["image"], mask_tensor


def preprocess_for_segmentation(
    frame_bgr: np.ndarray,
    size_hw: tuple[int, int],
    add_batch_dim: bool = True,
) -> torch.Tensor:
    """Preprocess a BGR uint8 frame for segmentation inference using val transforms."""
    val_tf = get_seg_val_transforms(size_hw)
    image_tensor, _ = apply_seg_transforms(frame_bgr=frame_bgr, transform=val_tf, mask_chw=None)
    if add_batch_dim:
        return image_tensor.unsqueeze(0)
    return image_tensor
