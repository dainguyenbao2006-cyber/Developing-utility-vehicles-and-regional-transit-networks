"""Train DeepLabV3+ semantic segmentation model on BDD100K with AMP and ComboLoss."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader

# Allow running directly as `python scripts/train_segmenter.py`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data.bdd_dataset import BDDSegmentationDataset, resolve_bdd_seg_dirs  # noqa: E402
from data.transforms import get_seg_train_transforms, get_seg_val_transforms  # noqa: E402
from evaluation import evaluate_segmenter_bdd_loader  # noqa: E402
from models.segmenter import ComboLoss, save_segmenter_checkpoint  # noqa: E402
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
        "--direct-id", type=int, default=0, help="Pixel value for drivable_direct (default: 0)."
    )
    parser.add_argument(
        "--alt-id", type=int, default=1, help="Pixel value for drivable_alternative (default: 1)."
    )
    parser.add_argument(
        "--lane-bg-id",
        type=int,
        default=255,
        help="Background pixel value in lane mask (default: 255).",
    )
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
    """Streaming validation computing average loss and per-channel IoU in O(1) memory."""
    model.eval()
    running_loss = 0.0
    count = 0

    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, masks)
            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            count += batch_size

    val_loss = running_loss / max(1, count)
    metrics = evaluate_segmenter_bdd_loader(
        model=model,
        loader=loader,
        device=device,
        thresholds=thresholds,
    )
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
    splits_dir = Path(cfg.paths.splits_dir)

    bdd_root = args.bdd_root if args.bdd_root is not None else Path(cfg.paths.bdd_root)
    images_train, drv_train, lane_train = resolve_bdd_seg_dirs(bdd_root, "train")
    images_val, drv_val, lane_val = resolve_bdd_seg_dirs(bdd_root, "val")

    if not images_train.exists():
        logger.error(
            "BDD images training directory not found at: %s. "
            "Please check bdd_root in configs/paths.yaml or download data first.",
            images_train,
        )
        return

    train_split_file = splits_dir / "train.txt"
    val_split_file = splits_dir / "val.txt"

    train_ds = BDDSegmentationDataset(
        images_dir=images_train,
        drivable_dir=drv_train,
        lane_dir=lane_train,
        size_hw=size_hw,
        transform=get_seg_train_transforms(size_hw),
        split_file=train_split_file if train_split_file.exists() else None,
        direct_value=args.direct_id,
        alternative_value=args.alt_id,
        lane_bg_value=args.lane_bg_id,
    )
    val_ds = BDDSegmentationDataset(
        images_dir=images_val,
        drivable_dir=drv_val,
        lane_dir=lane_val,
        size_hw=size_hw,
        transform=get_seg_val_transforms(size_hw),
        split_file=val_split_file if val_split_file.exists() else None,
        direct_value=args.direct_id,
        alternative_value=args.alt_id,
        lane_bg_value=args.lane_bg_id,
    )

    if args.evaluate_only:
        ckpt_to_eval = args.weights or output_weights
        from models.segmenter import DeepLabV3PlusSegmenter

        seg_wrapper = DeepLabV3PlusSegmenter(weights_path=ckpt_to_eval, device=str(device))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        eval_metrics = evaluate_segmenter_bdd_loader(
            model=seg_wrapper.model,
            loader=val_loader,
            device=device,
            thresholds=seg_wrapper.thresholds,
        )
        logger.info("Validation metrics: %s", eval_metrics)
        return

    num_workers = int(seg_cfg.train.get("num_workers", 2))
    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
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
