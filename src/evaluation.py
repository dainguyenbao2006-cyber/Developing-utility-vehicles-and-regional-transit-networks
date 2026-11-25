"""Quantitative evaluation, Cityscapes cross-dataset road union, and latency benchmarking."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from adas.alert_manager import ADASAnalyzer
from pipeline import DetectorProtocol, SegmenterProtocol
from utils.constants import SEG_CHANNELS
from visualization.hud import HUDVisualizer

logger = logging.getLogger(__name__)


def compute_segmentation_metrics(
    probs: torch.Tensor | np.ndarray,
    targets: torch.Tensor | np.ndarray,
    thresholds: tuple[float, float, float] | list[float] = (0.5, 0.5, 0.4),
    eps: float = 1e-6,
) -> dict[str, float]:
    """Compute per-channel IoU and Dice score across SEG_CHANNELS."""
    p_arr = probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else np.asarray(probs)
    t_arr = (
        targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    )

    if p_arr.shape != t_arr.shape:
        raise ValueError(f"Shape mismatch: probs {p_arr.shape} != targets {t_arr.shape}")
    if p_arr.ndim != 4 or p_arr.shape[1] != len(SEG_CHANNELS):
        raise ValueError(
            f"Expected (N, {len(SEG_CHANNELS)}, H, W) arrays, got shape {p_arr.shape}"
        )

    metrics: dict[str, float] = {}
    iou_sum = 0.0
    dice_sum = 0.0

    for idx, ch_name in enumerate(SEG_CHANNELS):
        pred_bin = p_arr[:, idx] >= float(thresholds[idx])
        targ_bin = t_arr[:, idx] >= 0.5

        inter = float(np.logical_and(pred_bin, targ_bin).sum())
        union = float(np.logical_or(pred_bin, targ_bin).sum())
        pred_sum = float(pred_bin.sum())
        targ_sum = float(targ_bin.sum())

        iou = (inter + eps) / (union + eps)
        dice = (2.0 * inter + eps) / (pred_sum + targ_sum + eps)

        metrics[f"iou_{ch_name}"] = float(iou)
        metrics[f"dice_{ch_name}"] = float(dice)
        iou_sum += iou
        dice_sum += dice

    n_ch = float(len(SEG_CHANNELS))
    metrics["mean_iou"] = float(iou_sum / n_ch)
    metrics["mean_dice"] = float(dice_sum / n_ch)
    return metrics


def compute_cityscapes_road_union_metrics(
    probs: torch.Tensor | np.ndarray,
    road_targets: torch.Tensor | np.ndarray,
    thresholds: tuple[float, float, float] | list[float] = (0.5, 0.5, 0.4),
    eps: float = 1e-6,
) -> dict[str, float]:
    """Compute Cityscapes cross-dataset road IoU/Dice via union (direct | alternative)."""
    p_arr = probs.detach().cpu().numpy() if isinstance(probs, torch.Tensor) else np.asarray(probs)
    t_arr = (
        road_targets.detach().cpu().numpy()
        if isinstance(road_targets, torch.Tensor)
        else np.asarray(road_targets)
    )

    if p_arr.ndim != 4 or p_arr.shape[1] < 2:
        raise ValueError(f"Expected probs shape (N, >=2, H, W), got {p_arr.shape}")
    if t_arr.ndim == 4 and t_arr.shape[1] == 1:
        t_arr = t_arr[:, 0]
    if t_arr.shape != (p_arr.shape[0], p_arr.shape[2], p_arr.shape[3]):
        raise ValueError(
            f"Road target shape {t_arr.shape} incompatible with probs shape {p_arr.shape}"
        )

    pred_direct = p_arr[:, 0] >= float(thresholds[0])
    pred_alt = p_arr[:, 1] >= float(thresholds[1])
    pred_road = np.logical_or(pred_direct, pred_alt)
    targ_road = t_arr >= 0.5

    inter = float(np.logical_and(pred_road, targ_road).sum())
    union = float(np.logical_or(pred_road, targ_road).sum())
    pred_sum = float(pred_road.sum())
    targ_sum = float(targ_road.sum())

    return {
        "cityscapes_road_iou": float((inter + eps) / (union + eps)),
        "cityscapes_road_dice": float((2.0 * inter + eps) / (pred_sum + targ_sum + eps)),
    }


def sweep_optimal_thresholds(
    probs: np.ndarray,
    targets: np.ndarray,
    candidates: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65),
    eps: float = 1e-6,
) -> list[float]:
    """Find per-channel probability thresholds that maximize validation IoU."""
    if probs.shape != targets.shape or probs.ndim != 4:
        raise ValueError(f"Expected matching 4D arrays, got {probs.shape} and {targets.shape}")

    best_thresholds: list[float] = []
    for c in range(probs.shape[1]):
        targ_bin = targets[:, c] >= 0.5
        best_t = 0.5
        best_iou = -1.0
        for t in candidates:
            pred_bin = probs[:, c] >= t
            inter = float(np.logical_and(pred_bin, targ_bin).sum())
            union = float(np.logical_or(pred_bin, targ_bin).sum())
            iou = (inter + eps) / (union + eps)
            if iou > best_iou:
                best_iou = iou
                best_t = float(t)
        best_thresholds.append(best_t)

    return best_thresholds


def evaluate_segmenter_bdd_loader(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    thresholds: tuple[float, float, float] | list[float],
    eps: float = 1e-6,
) -> dict[str, float]:
    """Streaming batch evaluation of DeepLabV3+ on BDD100K validation DataLoader."""
    model.eval()
    n_ch = len(SEG_CHANNELS)
    inter_totals = np.zeros(n_ch, dtype=np.float64)
    union_totals = np.zeros(n_ch, dtype=np.float64)
    pred_totals = np.zeros(n_ch, dtype=np.float64)
    targ_totals = np.zeros(n_ch, dtype=np.float64)
    samples_seen = 0

    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            probs = torch.sigmoid(model(images)).cpu().numpy()
            targs = masks.numpy()
            samples_seen += int(images.shape[0])

            for c in range(n_ch):
                pred_bin = probs[:, c] >= float(thresholds[c])
                targ_bin = targs[:, c] >= 0.5
                inter_totals[c] += float(np.logical_and(pred_bin, targ_bin).sum())
                union_totals[c] += float(np.logical_or(pred_bin, targ_bin).sum())
                pred_totals[c] += float(pred_bin.sum())
                targ_totals[c] += float(targ_bin.sum())

    if samples_seen == 0:
        return {"samples": 0.0, "mean_iou": 0.0, "mean_dice": 0.0}

    results: dict[str, float] = {"samples": float(samples_seen)}
    ious: list[float] = []
    dices: list[float] = []
    for c, name in enumerate(SEG_CHANNELS):
        iou = float((inter_totals[c] + eps) / (union_totals[c] + eps))
        dice = float((2.0 * inter_totals[c] + eps) / (pred_totals[c] + targ_totals[c] + eps))
        results[f"iou_{name}"] = iou
        results[f"dice_{name}"] = dice
        ious.append(iou)
        dices.append(dice)

    results["mean_iou"] = float(np.mean(ious))
    results["mean_dice"] = float(np.mean(dices))
    return results


def evaluate_segmenter_cityscapes_loader(
    model: torch.nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    thresholds: tuple[float, float, float] | list[float],
    eps: float = 1e-6,
) -> dict[str, float]:
    """Streaming cross-dataset evaluation on Cityscapes val using (direct | alternative)."""
    model.eval()
    inter_total = 0.0
    union_total = 0.0
    pred_total = 0.0
    targ_total = 0.0
    samples_seen = 0

    with torch.inference_mode():
        for images, road_masks in loader:
            images = images.to(device, non_blocking=True)
            probs = torch.sigmoid(model(images)).cpu().numpy()
            targs = road_masks.numpy() >= 0.5
            samples_seen += int(images.shape[0])

            pred_road = np.logical_or(
                probs[:, 0] >= float(thresholds[0]),
                probs[:, 1] >= float(thresholds[1]),
            )
            inter_total += float(np.logical_and(pred_road, targs).sum())
            union_total += float(np.logical_or(pred_road, targs).sum())
            pred_total += float(pred_road.sum())
            targ_total += float(targs.sum())

    if samples_seen == 0:
        return {"samples": 0.0, "cityscapes_road_iou": 0.0, "cityscapes_road_dice": 0.0}

    return {
        "samples": float(samples_seen),
        "cityscapes_road_iou": float((inter_total + eps) / (union_total + eps)),
        "cityscapes_road_dice": float((2.0 * inter_total + eps) / (pred_total + targ_total + eps)),
    }


def benchmark_end_to_end_latency(
    detector: DetectorProtocol | None,
    segmenter: SegmenterProtocol | None,
    analyzer: ADASAnalyzer,
    visualizer: HUDVisualizer,
    frame_hw: tuple[int, int] = (720, 1280),
    warmup_runs: int = 2,
    benchmark_runs: int = 10,
) -> dict[str, float]:
    """Benchmark stage-by-stage latency (ms) and total end-to-end FPS on 720p frames."""
    if benchmark_runs <= 0:
        raise ValueError(f"benchmark_runs must be positive, got {benchmark_runs}")

    height, width = frame_hw
    dummy_frame = np.full((height, width, 3), 114, dtype=np.uint8)
    empty_seg = np.zeros((len(SEG_CHANNELS), height, width), dtype=np.uint8)

    # Warmup passes
    analyzer.reset()
    for i in range(max(0, warmup_runs)):
        dets = detector.predict(dummy_frame) if detector is not None else []
        seg = segmenter.predict(dummy_frame) if segmenter is not None else empty_seg
        res = analyzer.analyze_frame(i, i * 0.04, dets, seg, (height, width))
        visualizer.draw_overlay(dummy_frame, dets, seg, analysis=res, fps=25.0)

    det_times: list[float] = []
    seg_times: list[float] = []
    adas_times: list[float] = []
    hud_times: list[float] = []
    total_times: list[float] = []

    analyzer.reset()
    for idx in range(benchmark_runs):
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        dets = detector.predict(dummy_frame) if detector is not None else []
        det_times.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        seg = segmenter.predict(dummy_frame) if segmenter is not None else empty_seg
        seg_times.append((time.perf_counter() - t1) * 1000.0)

        t2 = time.perf_counter()
        analysis = analyzer.analyze_frame(idx, idx * 0.04, dets, seg, (height, width))
        adas_times.append((time.perf_counter() - t2) * 1000.0)

        t3 = time.perf_counter()
        visualizer.draw_overlay(dummy_frame, dets, seg, analysis=analysis, fps=25.0)
        hud_times.append((time.perf_counter() - t3) * 1000.0)

        total_times.append((time.perf_counter() - t_start) * 1000.0)

    mean_total_ms = float(np.mean(total_times))
    end_to_end_fps = 1000.0 / max(1e-6, mean_total_ms)

    return {
        "frame_height": float(height),
        "frame_width": float(width),
        "runs": float(benchmark_runs),
        "detection_ms": float(np.mean(det_times)),
        "segmentation_ms": float(np.mean(seg_times)),
        "adas_logic_ms": float(np.mean(adas_times)),
        "visualization_ms": float(np.mean(hud_times)),
        "total_ms": mean_total_ms,
        "end_to_end_fps": float(end_to_end_fps),
    }


def _fmt_metric(mapping: dict[str, Any] | None, key: str) -> str:
    """Format metric value to 4 decimals or '-' if not yet evaluated (CLAUDE.md rule #24)."""
    if not mapping or key not in mapping or mapping[key] is None:
        return "-"
    return f"{float(mapping[key]):.4f}"


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render truthful Markdown evaluation summary leaving unmeasured fields as '-'."""
    det = report.get("detection")
    seg = report.get("segmentation_bdd")
    cs = report.get("segmentation_cityscapes")
    lat = report.get("latency")

    lines = [
        "# ADAS-Vision — Báo cáo Đánh giá Định lượng",
        "",
        "> Các ô có ký hiệu `-` biểu thị hạng mục chưa được chạy đánh giá trong lần thực thi này.",
        "",
        "## 1. Object Detection (BDD100K Val)",
        "",
        "| Metric | Kết quả |",
        "| :--- | :---: |",
        f"| mAP@0.5 (9 classes) | {_fmt_metric(det, 'map50')} |",
        f"| mAP@0.5:0.95 | {_fmt_metric(det, 'map50_95')} |",
        f"| Precision | {_fmt_metric(det, 'mp')} |",
        f"| Recall | {_fmt_metric(det, 'mr')} |",
        f"| AP@0.5 (`car`) | {_fmt_metric(det, 'ap50_car')} |",
        f"| AP@0.5 (`traffic_sign`) | {_fmt_metric(det, 'ap50_traffic_sign')} |",
        "",
        "## 2. Semantic Segmentation (BDD100K Val & Cityscapes Cross-Dataset)",
        "",
        "| Hạng mục | IoU | Dice |",
        "| :--- | :---: | :---: |",
        (
            f"| BDD100K `drivable_direct` | {_fmt_metric(seg, 'iou_drivable_direct')} "
            f"| {_fmt_metric(seg, 'dice_drivable_direct')} |"
        ),
        (
            f"| BDD100K `drivable_alternative` | {_fmt_metric(seg, 'iou_drivable_alternative')} "
            f"| {_fmt_metric(seg, 'dice_drivable_alternative')} |"
        ),
        (
            f"| BDD100K `lane_marking` | {_fmt_metric(seg, 'iou_lane_marking')} "
            f"| {_fmt_metric(seg, 'dice_lane_marking')} |"
        ),
        f"| BDD100K **Mean** | {_fmt_metric(seg, 'mean_iou')} | {_fmt_metric(seg, 'mean_dice')} |",
        (
            f"| Cityscapes `road` (direct \\| alt) | {_fmt_metric(cs, 'cityscapes_road_iou')} "
            f"| {_fmt_metric(cs, 'cityscapes_road_dice')} |"
        ),
        "",
        "## 3. Benchmark Độ trễ & FPS End-to-End (720p)",
        "",
        "| Giai đoạn | Thời gian trung bình (ms) |",
        "| :--- | :---: |",
        f"| Detection | {_fmt_metric(lat, 'detection_ms')} |",
        f"| Segmentation | {_fmt_metric(lat, 'segmentation_ms')} |",
        f"| ADAS Logic (NumPy) | {_fmt_metric(lat, 'adas_logic_ms')} |",
        f"| HUD Visualization | {_fmt_metric(lat, 'visualization_ms')} |",
        f"| **Tổng thời gian / frame** | {_fmt_metric(lat, 'total_ms')} |",
        f"| **End-to-End FPS** | {_fmt_metric(lat, 'end_to_end_fps')} |",
        "",
    ]
    return "\n".join(lines)


def save_evaluation_outputs(
    report: dict[str, Any],
    json_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> None:
    """Persist structured JSON and human-readable Markdown evaluation reports."""
    if json_path is not None:
        j_path = Path(json_path)
        j_path.parent.mkdir(parents=True, exist_ok=True)
        j_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved JSON evaluation report to %s", j_path)

    if markdown_path is not None:
        m_path = Path(markdown_path)
        m_path.parent.mkdir(parents=True, exist_ok=True)
        m_path.write_text(render_markdown_report(report), encoding="utf-8")
        logger.info("Saved Markdown evaluation report to %s", m_path)
