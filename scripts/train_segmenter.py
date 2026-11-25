"""Train DeepLabV3+ semantic segmentation model on BDD100K with AMP and ComboLoss."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

# Allow running directly as `python scripts/train_segmenter.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data.bdd_dataset import BDDSegmentationDataset  # noqa: E402
from data.transforms import get_seg_train_transforms, get_seg_val_transforms  # noqa: E402
from models.segmenter import (  # noqa: E402
    ComboLoss,
    compute_channel_ious,
    save_segmenter_checkpoint,
)
from utils.config import load_project_configs  # noqa: E402
from utils.constants import SEG_CHANNELS  # noqa: E402

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for segmentation training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Train or evaluate DeepLabV3+ on BDD100K road & lane segmentation."
    )
    parser.add_argument("--bdd-root", type=Path, default=None, help="Override cfg.paths.bdd_root.")
    parser.add_argument("--epochs", type=int, default=None, help="Epoch count (overrides config).")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Batch size (overrides config)."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config).")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu').")
    parser.add_argument(
        "--output-weights",
        type=Path,
        default=None,
        help="Path to save best checkpoint (default from configs/paths.yaml).",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Run evaluation on a checkpoint without training.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Checkpoint path to evaluate when --evaluate-only is set.",
    )
    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: ComboLoss,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> float:
    """Train model for one epoch and return average loss."""
    model.train()
    running_loss = 0.0
    count = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        batch_size = images.size(0)
        running_loss += float(loss.item()) * batch_size
        count += batch_size

    return running_loss / max(1, count)


def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    criterion: ComboLoss,
    device: torch.device,
    thresholds: tuple[float, float, float] | list[float],
) -> tuple[float, dict[str, float]]:
    """Evaluate model on validation loader and compute loss and channel IoUs."""
    model.eval()
    running_loss = 0.0
    count = 0
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, masks)

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            count += batch_size

            probs = torch.sigmoid(logits).cpu().numpy()
            targets_np = masks.cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets_np)

    val_loss = running_loss / max(1, count)
    if not all_probs:
        return val_loss, {"mean_iou": 0.0}

    cat_probs = np.concatenate(all_probs, axis=0)
    cat_targets = np.concatenate(all_targets, axis=0)
    metrics = compute_channel_ious(cat_probs, cat_targets, thresholds=thresholds)
    return val_loss, metrics


def run_training() -> None:
    """Main training workflow coordinating data loaders, optimizer, and checkpoints."""
    args = parse_args()
    cfg = load_project_configs()

    device_str = args.device or cfg.project.device
    device = torch.device(
        device_str if (device_str != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    logger.info("Using device: %s", device)

    seg_cfg = cfg.segmenter
    size_hw = tuple(seg_cfg.model.input_size_hw)
    epochs = args.epochs if args.epochs is not None else int(seg_cfg.train.epochs)
    batch_size = args.batch_size if args.batch_size is not None else int(seg_cfg.train.batch_size)
    lr = args.lr if args.lr is not None else float(seg_cfg.train.lr)
    output_weights = args.output_weights or Path(cfg.paths.segmenter_weights)

    bdd_root = args.bdd_root if args.bdd_root is not None else Path(cfg.paths.bdd_root)
    images_train = bdd_root / "images" / "100k" / "train"
    images_val = bdd_root / "images" / "100k" / "val"
    drv_train = bdd_root / "drivable_maps" / "labels" / "train"
    drv_val = bdd_root / "drivable_maps" / "labels" / "val"
    lane_train = bdd_root / "lane" / "labels" / "train"
    lane_val = bdd_root / "lane" / "labels" / "val"

    if not images_train.exists():
        logger.error(
            "BDD images training directory not found at: %s. "
            "Please check bdd_root in configs/paths.yaml or download data first.",
            images_train,
        )
        return

    train_ds = BDDSegmentationDataset(
        images_dir=images_train,
        drivable_dir=drv_train if drv_train.exists() else None,
        lane_dir=lane_train if lane_train.exists() else None,
        size_hw=size_hw,
        transform=get_seg_train_transforms(size_hw),
    )
    val_ds = BDDSegmentationDataset(
        images_dir=images_val,
        drivable_dir=drv_val if drv_val.exists() else None,
        lane_dir=lane_val if lane_val.exists() else None,
        size_hw=size_hw,
        transform=get_seg_val_transforms(size_hw),
    )

    num_workers = int(seg_cfg.train.get("num_workers", 2))
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    encoder_name = seg_cfg.model.encoder
    logger.info("Initializing DeepLabV3Plus (encoder=%s, classes=3)...", encoder_name)
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=len(SEG_CHANNELS),
    ).to(device)

    channel_weights = seg_cfg.train.get("channel_weights", [1.0, 1.0, 1.5])
    criterion = ComboLoss(
        bce_weight=float(seg_cfg.train.bce_weight),
        dice_weight=float(seg_cfg.train.dice_weight),
        channel_weights=channel_weights,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=float(seg_cfg.train.weight_decay)
    )
    scaler = torch.amp.GradScaler("cuda") if (seg_cfg.train.amp and device.type == "cuda") else None
    thresholds = list(seg_cfg.model.thresholds)

    best_mean_iou = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
        )
        val_loss, metrics = validate_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            thresholds=thresholds,
        )

        mean_iou = metrics["mean_iou"]
        logger.info(
            "Epoch [%d/%d] | Train Loss: %.4f | Val Loss: %.4f | Mean IoU: %.4f "
            "(Direct: %.4f, Alt: %.4f, Lane: %.4f)",
            epoch,
            epochs,
            train_loss,
            val_loss,
            mean_iou,
            metrics.get("iou_drivable_direct", 0.0),
            metrics.get("iou_drivable_alternative", 0.0),
            metrics.get("iou_lane_marking", 0.0),
        )

        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            save_segmenter_checkpoint(
                checkpoint_path=output_weights,
                model=model,
                cfg=seg_cfg.to_dict(),
                epoch=epoch,
                metrics=metrics,
                thresholds=thresholds,
                size_hw=size_hw,
            )
            logger.info("New best checkpoint saved with Mean IoU: %.4f", best_mean_iou)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    run_training()
