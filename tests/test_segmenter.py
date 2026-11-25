"""Unit tests for Milestone 4: DeepLabV3+ segmenter, ComboLoss, and checkpoint contracts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

from models.segmenter import (
    ComboLoss,
    DeepLabV3PlusSegmenter,
    compute_channel_ious,
    save_segmenter_checkpoint,
)
from utils.constants import SEG_CHANNELS


def test_combo_loss_computation() -> None:
    loss_fn = ComboLoss(bce_weight=0.5, dice_weight=0.5, channel_weights=[1.0, 1.0, 1.5])
    logits = torch.randn(2, 3, 32, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 3, 32, 64)).float()

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert float(loss.item()) > 0.0

    loss.backward()
    assert logits.grad is not None


def test_combo_loss_shape_mismatch_raises() -> None:
    loss_fn = ComboLoss()
    logits = torch.randn(2, 3, 32, 64)
    targets = torch.randn(2, 2, 32, 64)
    with pytest.raises(ValueError, match="Shape mismatch"):
        loss_fn(logits, targets)


def test_compute_channel_ious() -> None:
    # Perfect match test
    probs = np.zeros((1, 3, 10, 10), dtype=np.float32)
    probs[0, 0, :5, :] = 0.9  # channel 0
    probs[0, 1, 5:, :] = 0.8  # channel 1
    probs[0, 2, 2:4, :] = 0.7  # channel 2

    targets = np.zeros((1, 3, 10, 10), dtype=np.float32)
    targets[0, 0, :5, :] = 1.0
    targets[0, 1, 5:, :] = 1.0
    targets[0, 2, 2:4, :] = 1.0

    metrics = compute_channel_ious(probs, targets, thresholds=(0.5, 0.5, 0.4))
    for name in SEG_CHANNELS:
        assert metrics[f"iou_{name}"] == pytest.approx(1.0, abs=1e-3)
    assert metrics["mean_iou"] == pytest.approx(1.0, abs=1e-3)

    # Complete mismatch test
    mismatch_probs = np.zeros((1, 3, 10, 10), dtype=np.float32)
    mismatch_metrics = compute_channel_ious(mismatch_probs, targets)
    for name in SEG_CHANNELS:
        assert mismatch_metrics[f"iou_{name}"] == pytest.approx(0.0, abs=1e-3)


def test_save_and_load_segmenter_checkpoint(tmp_path: Path) -> None:
    dummy_model = nn.Sequential(nn.Conv2d(3, 3, kernel_size=1))
    ckpt_path = tmp_path / "segmenter_best.pt"

    cfg = {"model": {"encoder": "resnet34"}, "train": {"lr": 0.001}}
    metrics = {"mean_iou": 0.75, "iou_drivable_direct": 0.80}

    save_segmenter_checkpoint(
        checkpoint_path=ckpt_path,
        model=dummy_model,
        cfg=cfg,
        epoch=5,
        metrics=metrics,
        thresholds=[0.5, 0.5, 0.4],
        size_hw=(384, 640),
    )

    assert ckpt_path.exists()
    payload = torch.load(ckpt_path, map_location="cpu")
    assert payload["epoch"] == 5
    assert payload["channels"] == SEG_CHANNELS
    assert payload["size_hw"] == (384, 640)
    assert payload["thresholds"] == [0.5, 0.5, 0.4]


def test_segmenter_checkpoint_missing_keys_raises(tmp_path: Path) -> None:
    bad_ckpt = tmp_path / "bad.pt"
    torch.save({"model_state": {}}, bad_ckpt)

    with pytest.raises(ValueError, match="missing required keys"):
        DeepLabV3PlusSegmenter(weights_path=bad_ckpt)


def test_segmenter_checkpoint_channel_mismatch_raises(tmp_path: Path) -> None:
    bad_ckpt = tmp_path / "bad_channels.pt"
    torch.save(
        {
            "model_state": {},
            "cfg": {},
            "epoch": 1,
            "metrics": {},
            "thresholds": [0.5, 0.5, 0.4],
            "size_hw": (384, 640),
            "channels": ("lane_marking", "drivable_direct", "drivable_alternative"),
        },
        bad_ckpt,
    )
    with pytest.raises(ValueError, match="Channel order mismatch"):
        DeepLabV3PlusSegmenter(weights_path=bad_ckpt)


def test_segmenter_predict_pipeline() -> None:
    # Create mock model that outputs fixed logits
    mock_model = MagicMock()
    # Output logits of shape (1, 3, 384, 640)
    mock_logits = torch.full((1, 3, 384, 640), 2.0)  # sigmoid(2.0) ~ 0.88
    mock_model.return_value = mock_logits

    segmenter = DeepLabV3PlusSegmenter(
        model_instance=mock_model,
        size_hw=(384, 640),
        thresholds=(0.5, 0.5, 0.4),
        device="cpu",
    )

    orig_h, orig_w = 120, 240
    frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    # 1. Binary mask output
    binary_mask = segmenter.predict(frame, return_prob=False)
    assert binary_mask.shape == (3, orig_h, orig_w)
    assert binary_mask.dtype == np.uint8
    assert np.all(binary_mask == 1)

    # 2. Probability mask output
    prob_mask = segmenter.predict(frame, return_prob=True)
    assert prob_mask.shape == (3, orig_h, orig_w)
    assert prob_mask.dtype == np.float32
    assert np.all(prob_mask > 0.8)


def test_segmenter_input_validation() -> None:
    mock_model = MagicMock()
    segmenter = DeepLabV3PlusSegmenter(model_instance=mock_model, device="cpu")

    with pytest.raises(TypeError, match="Expected frame_bgr as np.ndarray"):
        segmenter.predict([1, 2, 3])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Expected BGR frame with shape"):
        segmenter.predict(np.zeros((100, 100), dtype=np.uint8))

    with pytest.raises(ValueError, match="Expected frame dtype uint8"):
        segmenter.predict(np.zeros((100, 100, 3), dtype=np.float32))
