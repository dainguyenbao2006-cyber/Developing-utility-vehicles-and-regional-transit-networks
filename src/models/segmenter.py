"""DeepLabV3+ Semantic Segmenter wrapper, ComboLoss, and checkpoint management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from data.transforms import preprocess_for_segmentation
from utils.constants import SEG_CHANNELS

logger = logging.getLogger(__name__)

CHECKPOINT_REQUIRED_KEYS = (
    "model_state",
    "cfg",
    "epoch",
    "metrics",
    "thresholds",
    "size_hw",
    "channels",
)


class ComboLoss(nn.Module):
    """Combined BCEWithLogits and Dice Loss with channel-wise weighting for multi-label masks."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        channel_weights: tuple[float, float, float] | list[float] | None = None,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.channel_weights = (
            torch.tensor(channel_weights, dtype=torch.float32)
            if channel_weights is not None
            else None
        )
        self.bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute ComboLoss given logits and binary float targets in (B, C, H, W)."""
        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: logits {logits.shape} != targets {targets.shape}"
            )

        # 1. BCE Loss with optional channel weighting
        bce_per_elem = self.bce_loss_fn(logits, targets)  # (B, C, H, W)
        if self.channel_weights is not None:
            w = self.channel_weights.to(logits.device).view(1, -1, 1, 1)
            bce_loss = (bce_per_elem * w).mean()
        else:
            bce_loss = bce_per_elem.mean()

        # 2. Dice Loss per channel
        probs = torch.sigmoid(logits)
        num_channels = logits.shape[1]
        dice_losses: list[torch.Tensor] = []

        for c in range(num_channels):
            prob_c = probs[:, c].reshape(-1)
            target_c = targets[:, c].reshape(-1)
            intersection = (prob_c * target_c).sum()
            dice = (2.0 * intersection + self.smooth) / (
                prob_c.sum() + target_c.sum() + self.smooth
            )
            dice_loss_c = 1.0 - dice
            if self.channel_weights is not None:
                dice_loss_c = dice_loss_c * self.channel_weights[c].to(logits.device)
            dice_losses.append(dice_loss_c)

        total_dice_loss = torch.stack(dice_losses).mean()

        return self.bce_weight * bce_loss + self.dice_weight * total_dice_loss


def compute_channel_ious(
    probs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    thresholds: tuple[float, float, float] | list[float] = (0.5, 0.5, 0.4),
    eps: float = 1e-6,
) -> dict[str, float]:
    """Compute per-channel IoU and mean IoU across 3 channels in SEG_CHANNELS."""
    if isinstance(probs, torch.Tensor):
        p_arr = probs.detach().cpu().numpy()
    else:
        p_arr = np.asarray(probs)

    if isinstance(targets, torch.Tensor):
        t_arr = targets.detach().cpu().numpy()
    else:
        t_arr = np.asarray(targets)

    if p_arr.shape != t_arr.shape:
        raise ValueError(f"Shape mismatch: probs {p_arr.shape} != targets {t_arr.shape}")

    ious: dict[str, float] = {}
    sum_iou = 0.0

    for idx, channel_name in enumerate(SEG_CHANNELS):
        th = thresholds[idx]
        pred_bin = p_arr[:, idx] >= th
        target_bin = t_arr[:, idx] >= 0.5

        intersection = np.logical_and(pred_bin, target_bin).sum()
        union = np.logical_or(pred_bin, target_bin).sum()
        iou = float((intersection + eps) / (union + eps))
        ious[f"iou_{channel_name}"] = iou
        sum_iou += iou

    ious["mean_iou"] = sum_iou / len(SEG_CHANNELS)
    return ious


def save_segmenter_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    cfg: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
    thresholds: tuple[float, float, float] | list[float],
    size_hw: tuple[int, int],
) -> Path:
    """Save segmentation checkpoint strictly complying with CLAUDE.md rule #21."""
    dest = Path(checkpoint_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state": model.state_dict(),
        "cfg": cfg,
        "epoch": int(epoch),
        "metrics": metrics,
        "thresholds": list(thresholds),
        "size_hw": tuple(size_hw),
        "channels": SEG_CHANNELS,
    }
    torch.save(payload, dest)
    logger.info("Saved segmentation checkpoint to: %s (epoch=%d)", dest, epoch)
    return dest


class DeepLabV3PlusSegmenter:
    """Wrapper for smp.DeepLabV3Plus multi-label road and lane segmentation."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        size_hw: tuple[int, int] = (384, 640),
        thresholds: tuple[float, float, float] | list[float] = (0.5, 0.5, 0.4),
        device: str | None = None,
        model_instance: nn.Module | None = None,
    ) -> None:
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.size_hw = tuple(size_hw)
        self.thresholds = list(thresholds)
        self.cfg: dict[str, Any] = {}

        if model_instance is not None:
            self.model = model_instance
        elif weights_path is not None:
            self.model = self._load_from_checkpoint(Path(weights_path))
        else:
            logger.info("Initializing fresh DeepLabV3Plus (encoder=%s)...", encoder_name)
            self.model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=len(SEG_CHANNELS),
            )

        self.model.to(self.device)
        self.model.eval()

    def _load_from_checkpoint(self, checkpoint_path: Path) -> nn.Module:
        """Load and validate checkpoint containing all 7 required keys (CLAUDE.md rule #21)."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info("Loading segmentation checkpoint from: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if not isinstance(checkpoint, dict):
            raise ValueError(f"Expected dict checkpoint, got {type(checkpoint).__name__}")

        missing = [k for k in CHECKPOINT_REQUIRED_KEYS if k not in checkpoint]
        if missing:
            raise ValueError(f"Checkpoint is missing required keys: {missing}")

        loaded_channels = tuple(checkpoint["channels"])
        if loaded_channels != SEG_CHANNELS:
            raise ValueError(
                f"Channel order mismatch! Loaded: {loaded_channels} != Expected: {SEG_CHANNELS}"
            )

        self.cfg = checkpoint.get("cfg", {})
        self.size_hw = tuple(checkpoint.get("size_hw", self.size_hw))
        self.thresholds = list(checkpoint.get("thresholds", self.thresholds))

        # Recreate model from architecture config if present
        encoder_name = self.cfg.get("model", {}).get("encoder", "resnet34")
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=len(SEG_CHANNELS),
        )
        model.load_state_dict(checkpoint["model_state"])
        return model

    def predict(self, frame_bgr: np.ndarray, return_prob: bool = False) -> np.ndarray:
        """Run segmentation inference on a single BGR uint8 video frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Input frame in OpenCV BGR format (H, W, 3) with uint8 dtype.
        return_prob : bool
            If True, returns float32 probability maps in [0, 1].
            If False, returns binary uint8 mask {0, 1} after applying thresholds.

        Returns
        -------
        np.ndarray
            Output array of shape (3, H_orig, W_orig) matching frame dimensions.
        """
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError(f"Expected frame_bgr as np.ndarray, got {type(frame_bgr).__name__}")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                f"Expected BGR frame with shape (H, W, 3), got shape {frame_bgr.shape}"
            )
        if frame_bgr.dtype != np.uint8:
            raise ValueError(f"Expected frame dtype uint8, got {frame_bgr.dtype}")

        orig_h, orig_w = frame_bgr.shape[:2]

        # CLAUDE.md rule #20: Preprocess inference must share same function as validation
        input_tensor = preprocess_for_segmentation(
            frame_bgr=frame_bgr,
            size_hw=self.size_hw,
            add_batch_dim=True,
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # (3, H_model, W_model)

        # Resize each channel back to original frame size (W_orig, H_orig)
        out_masks = np.zeros((len(SEG_CHANNELS), orig_h, orig_w), dtype=np.float32)
        for c in range(len(SEG_CHANNELS)):
            out_masks[c] = cv2.resize(
                probs[c],
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )

        if return_prob:
            return out_masks

        # Apply per-channel thresholds to obtain binary uint8 mask
        binary_masks = np.zeros((len(SEG_CHANNELS), orig_h, orig_w), dtype=np.uint8)
        for c in range(len(SEG_CHANNELS)):
            binary_masks[c] = (out_masks[c] >= self.thresholds[c]).astype(np.uint8)

        return binary_masks
