"""Unit tests for Milestone 7: quantitative evaluation, Cityscapes union, and latency benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from adas.alert_manager import ADASAnalyzer
from evaluation import (
    benchmark_end_to_end_latency,
    compute_cityscapes_road_union_metrics,
    compute_segmentation_metrics,
    evaluate_segmenter_bdd_loader,
    evaluate_segmenter_cityscapes_loader,
    render_markdown_report,
    save_evaluation_outputs,
    sweep_optimal_thresholds,
)
from utils.config import load_project_configs
from visualization.hud import HUDVisualizer


class _FixedLogitNet(nn.Module):
    """Tiny module outputting fixed logits for deterministic loader tests."""

    def __init__(self, out_logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("out_logits", out_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_logits.expand(x.shape[0], -1, -1, -1)


def test_compute_segmentation_metrics_iou_and_dice() -> None:
    probs = np.zeros((1, 3, 4, 4), dtype=np.float32)
    targets = np.zeros((1, 3, 4, 4), dtype=np.float32)

    # Channel 0: pred covers 2 pixels, target covers 2 pixels, overlap = 1 -> IoU=1/3, Dice=2/4=0.5
    probs[0, 0, 0, 0:2] = 0.9
    targets[0, 0, 0, 1:3] = 1.0

    # Channel 1 & 2: exact match on 4 pixels -> IoU=1.0, Dice=1.0
    probs[0, 1, 1, :] = 0.8
    targets[0, 1, 1, :] = 1.0
    probs[0, 2, 2, :] = 0.7
    targets[0, 2, 2, :] = 1.0

    res = compute_segmentation_metrics(probs, targets, thresholds=(0.5, 0.5, 0.4))
    assert res["iou_drivable_direct"] == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert res["dice_drivable_direct"] == pytest.approx(0.5, abs=1e-4)
    assert res["iou_drivable_alternative"] == pytest.approx(1.0, abs=1e-4)
    assert res["dice_lane_marking"] == pytest.approx(1.0, abs=1e-4)


def test_cityscapes_road_union_combines_direct_and_alternative() -> None:
    probs = np.zeros((1, 3, 8, 8), dtype=np.float32)
    # Direct predicts left half of lower road; alternative predicts right half
    probs[0, 0, 4:, :4] = 0.85
    probs[0, 1, 4:, 4:] = 0.85

    # Cityscapes ground-truth road covers entire lower half
    road_gt = np.zeros((1, 8, 8), dtype=np.float32)
    road_gt[0, 4:, :] = 1.0

    res = compute_cityscapes_road_union_metrics(probs, road_gt, thresholds=(0.5, 0.5, 0.4))
    assert res["cityscapes_road_iou"] == pytest.approx(1.0, abs=1e-4)
    assert res["cityscapes_road_dice"] == pytest.approx(1.0, abs=1e-4)


def test_sweep_optimal_thresholds_picks_low_threshold_for_faint_lane() -> None:
    probs = np.zeros((2, 3, 8, 8), dtype=np.float32)
    targets = np.zeros((2, 3, 8, 8), dtype=np.float32)

    probs[:, 0, :4, :] = 0.8
    targets[:, 0, :4, :] = 1.0

    probs[:, 1, 4:, :] = 0.75
    targets[:, 1, 4:, :] = 1.0

    # Faint lane marking probabilities around 0.32 (would be missed at 0.50)
    probs[:, 2, 3, :] = 0.32
    targets[:, 2, 3, :] = 1.0

    best = sweep_optimal_thresholds(probs, targets)
    assert len(best) == 3
    assert best[2] <= 0.32


def test_streaming_bdd_and_cityscapes_loader_evaluators() -> None:
    logits = torch.full((1, 3, 8, 8), -5.0)
    logits[0, 0, 4:, :4] = 5.0   # direct on bottom-left
    logits[0, 1, 4:, 4:] = 5.0   # alternative on bottom-right
    logits[0, 2, 4:, 3:5] = 5.0  # lane marking in center
    model = _FixedLogitNet(logits)

    bdd_targets = (torch.sigmoid(logits) > 0.5).float().repeat(2, 1, 1, 1)
    dummy_imgs = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
    bdd_loader = DataLoader(TensorDataset(dummy_imgs, bdd_targets), batch_size=1)

    bdd_res = evaluate_segmenter_bdd_loader(
        model=model,
        loader=bdd_loader,
        device=torch.device("cpu"),
        thresholds=(0.5, 0.5, 0.4),
    )
    assert bdd_res["samples"] == 2.0
    assert bdd_res["mean_iou"] == pytest.approx(1.0, abs=1e-4)

    cs_targets = torch.zeros((2, 8, 8), dtype=torch.float32)
    cs_targets[:, 4:, :] = 1.0
    cs_loader = DataLoader(TensorDataset(dummy_imgs, cs_targets), batch_size=2)

    cs_res = evaluate_segmenter_cityscapes_loader(
        model=model,
        loader=cs_loader,
        device=torch.device("cpu"),
        thresholds=(0.5, 0.5, 0.4),
    )
    assert cs_res["cityscapes_road_iou"] == pytest.approx(1.0, abs=1e-4)


def test_benchmark_end_to_end_latency_and_report_saving(tmp_path: Path) -> None:
    cfg = load_project_configs()
    adas_dict = cfg.adas.to_dict()
    analyzer = ADASAnalyzer.from_config(adas_dict)
    visualizer = HUDVisualizer.from_config(adas_dict)

    lat = benchmark_end_to_end_latency(
        detector=None,
        segmenter=None,
        analyzer=analyzer,
        visualizer=visualizer,
        frame_hw=(180, 320),
        warmup_runs=1,
        benchmark_runs=3,
    )
    assert lat["runs"] == 3.0
    assert lat["end_to_end_fps"] > 0.0
    assert lat["total_ms"] > 0.0

    report = {"device": "cpu", "latency": lat}
    md = render_markdown_report(report)
    # Detection was not evaluated -> must remain '-' (CLAUDE.md rule #24)
    assert "| mAP@0.5 (9 classes) | - |" in md
    assert "| **End-to-End FPS** | - |" not in md

    json_out = tmp_path / "eval.json"
    md_out = tmp_path / "eval.md"
    save_evaluation_outputs(report, json_path=json_out, markdown_path=md_out)

    assert json_out.exists()
    assert md_out.exists()
    loaded = json.loads(json_out.read_text(encoding="utf-8"))
    assert loaded["latency"]["runs"] == 3.0
