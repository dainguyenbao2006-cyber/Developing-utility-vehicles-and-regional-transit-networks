"""HUD overlay visualizer for segmentation masks, detection boxes, corridor, and ADAS alerts."""

from __future__ import annotations

import csv
import logging
import unicodedata
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.types import ADASAlert, Detection, FrameAnalysis

logger = logging.getLogger(__name__)

# BGR color palette (OpenCV convention)
COLOR_DIRECT_BGR = (0, 200, 0)         # Green for drivable_direct
COLOR_ALT_BGR = (200, 120, 0)          # Blue-ish for drivable_alternative
COLOR_LANE_BGR = (0, 220, 255)         # Yellow-orange for lane_marking

COLOR_VEHICLE_BGR = (255, 160, 40)     # Blue-cyan for vehicles
COLOR_VRU_BGR = (0, 140, 255)          # Orange for pedestrians/riders/bicycles
COLOR_SIGNAL_BGR = (255, 255, 0)       # Cyan for traffic lights/signs

COLOR_SAFE_BGR = (60, 200, 60)
COLOR_WARN_BGR = (0, 200, 255)
COLOR_CRIT_BGR = (0, 0, 240)
COLOR_PANEL_BG = (25, 25, 25)
COLOR_TEXT_WHITE = (245, 245, 245)

VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle"})
VRU_CLASSES = frozenset({"pedestrian", "rider", "bicycle"})
SIGNAL_CLASSES = frozenset({"traffic_light", "traffic_sign"})


def strip_accents_for_cv2(text: str) -> str:
    """Convert Vietnamese accented string to ASCII so cv2.putText renders cleanly."""
    replaced = text.replace("Đ", "D").replace("đ", "d")
    normalized = unicodedata.normalize("NFKD", replaced)
    ascii_chars = [ch for ch in normalized if not unicodedata.combining(ch)]
    # Keep printable ASCII characters
    return "".join(ch if 32 <= ord(ch) < 127 else " " for ch in ascii_chars).strip()


def _status_color(status: str) -> tuple[int, int, int]:
    upper = status.upper()
    if upper in {"CRITICAL", "DEPARTED"}:
        return COLOR_CRIT_BGR
    if upper in {"WARNING", "CAUTION"}:
        return COLOR_WARN_BGR
    return COLOR_SAFE_BGR


def _bboxes_close(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
    tol: float = 1.5,
) -> bool:
    return all(abs(a - b) <= tol for a, b in zip(box_a, box_b, strict=False))


class HUDVisualizer:
    """Renders non-destructive ADAS HUD overlays onto copies of BGR uint8 frames."""

    def __init__(
        self,
        mask_alpha: float = 0.35,
        corridor_top_y_ratio: float = 0.45,
        corridor_top_width_ratio: float = 0.18,
        corridor_bottom_width_ratio: float = 0.45,
        draw_masks: bool = True,
        draw_boxes: bool = True,
        draw_corridor: bool = True,
        draw_hud: bool = True,
    ) -> None:
        self.mask_alpha = float(np.clip(mask_alpha, 0.0, 1.0))
        self.corridor_top_y_ratio = float(corridor_top_y_ratio)
        self.corridor_top_width_ratio = float(corridor_top_width_ratio)
        self.corridor_bottom_width_ratio = float(corridor_bottom_width_ratio)
        self.draw_masks = bool(draw_masks)
        self.draw_boxes = bool(draw_boxes)
        self.draw_corridor = bool(draw_corridor)
        self.draw_hud = bool(draw_hud)

    @classmethod
    def from_config(cls, adas_cfg: dict[str, Any], mask_alpha: float = 0.35) -> HUDVisualizer:
        """Create HUDVisualizer aligned with FCW corridor geometry from adas config."""
        rules = adas_cfg.get("adas", adas_cfg)
        fcw_cfg = rules.get("fcw", {})
        return cls(
            mask_alpha=mask_alpha,
            corridor_top_y_ratio=float(fcw_cfg.get("corridor_top_y_ratio", 0.45)),
            corridor_top_width_ratio=float(fcw_cfg.get("corridor_top_width_ratio", 0.18)),
            corridor_bottom_width_ratio=float(fcw_cfg.get("corridor_bottom_width_ratio", 0.45)),
        )

    def draw_overlay(
        self,
        frame_bgr: np.ndarray,
        detections: list[Detection],
        seg_mask: np.ndarray | None,
        analysis: FrameAnalysis | None = None,
        fps: float | None = None,
    ) -> np.ndarray:
        """Return a new BGR uint8 frame with segmentation, boxes, corridor, and HUD drawn.

        Does NOT mutate `frame_bgr` or `seg_mask` in-place (CLAUDE.md rule #16).
        """
        if not isinstance(frame_bgr, np.ndarray):
            raise TypeError(f"Expected frame_bgr as np.ndarray, got {type(frame_bgr).__name__}")
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(
                f"Expected BGR frame with shape (H, W, 3), got shape {frame_bgr.shape}"
            )
        if frame_bgr.dtype != np.uint8:
            raise ValueError(f"Expected frame dtype uint8, got {frame_bgr.dtype}")

        canvas = frame_bgr.copy()

        if self.draw_masks and seg_mask is not None:
            self._render_segmentation_mask(canvas, seg_mask)

        if self.draw_corridor:
            self._render_ego_corridor(canvas)

        if self.draw_boxes and detections:
            self._render_detections(
                canvas=canvas,
                detections=detections,
                alerts=analysis.alerts if analysis is not None else [],
                frame_idx=analysis.frame_idx if analysis is not None else 0,
            )

        if self.draw_hud:
            self._render_status_panel(canvas, analysis=analysis, fps=fps)
            if analysis is not None:
                self._render_critical_banner(canvas, analysis.alerts)

        return canvas

    def _render_segmentation_mask(self, canvas: np.ndarray, seg_mask: np.ndarray) -> None:
        """Blend 3-channel segmentation mask onto canvas in-place (canvas is already a copy)."""
        if seg_mask.ndim != 3 or seg_mask.shape[0] != 3:
            raise ValueError(f"Expected seg_mask shape (3, H, W), got {seg_mask.shape}")

        height, width = canvas.shape[:2]
        mask_h, mask_w = seg_mask.shape[1], seg_mask.shape[2]

        if (mask_h, mask_w) != (height, width):
            resized = np.zeros((3, height, width), dtype=np.float32)
            for c in range(3):
                resized[c] = cv2.resize(
                    seg_mask[c].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
            active = resized > 0.5
        else:
            active = seg_mask > 0.5

        color_layer = np.zeros_like(canvas)
        any_mask = np.zeros((height, width), dtype=bool)

        # Paint in priority order: alternative -> direct -> lane_marking on top
        for ch_idx, color in (
            (1, COLOR_ALT_BGR),
            (0, COLOR_DIRECT_BGR),
            (2, COLOR_LANE_BGR),
        ):
            ch_bool = active[ch_idx]
            if np.any(ch_bool):
                color_layer[ch_bool] = color
                any_mask |= ch_bool

        if np.any(any_mask):
            blended = cv2.addWeighted(
                canvas, 1.0 - self.mask_alpha, color_layer, self.mask_alpha, 0.0
            )
            canvas[any_mask] = blended[any_mask]

    def _render_ego_corridor(self, canvas: np.ndarray) -> None:
        """Draw subtle trapezoid guideline representing the ego collision corridor."""
        height, width = canvas.shape[:2]
        top_y = int(round(height * self.corridor_top_y_ratio))
        bottom_y = height - 1
        center_x = 0.5 * width

        top_half_w = 0.5 * width * self.corridor_top_width_ratio
        bot_half_w = 0.5 * width * self.corridor_bottom_width_ratio

        pts = np.array(
            [
                [int(round(center_x - top_half_w)), top_y],
                [int(round(center_x + top_half_w)), top_y],
                [int(round(center_x + bot_half_w)), bottom_y],
                [int(round(center_x - bot_half_w)), bottom_y],
            ],
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        overlay = canvas.copy()
        cv2.polylines(overlay, [pts], isClosed=True, color=(220, 220, 220), thickness=1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0.0, dst=canvas)

    def _render_detections(
        self,
        canvas: np.ndarray,
        detections: list[Detection],
        alerts: list[ADASAlert],
        frame_idx: int,
    ) -> None:
        """Draw bounding boxes color-coded by category and active alert state."""
        height, width = canvas.shape[:2]
        fcw_boxes = [
            a.target_bbox for a in alerts if a.alert_type == "FCW" and a.target_bbox is not None
        ]
        pcw_boxes = [
            a.target_bbox for a in alerts if a.alert_type == "PCW" and a.target_bbox is not None
        ]

        for det in detections:
            x1, y1, x2, y2 = det.bbox_xyxy
            ix1 = max(0, min(width - 1, int(round(x1))))
            iy1 = max(0, min(height - 1, int(round(y1))))
            ix2 = max(0, min(width - 1, int(round(x2))))
            iy2 = max(0, min(height - 1, int(round(y2))))

            is_fcw_target = any(_bboxes_close(det.bbox_xyxy, b) for b in fcw_boxes)
            is_pcw_target = any(_bboxes_close(det.bbox_xyxy, b) for b in pcw_boxes)

            thickness = 2
            if is_pcw_target:
                # Blinking red/yellow border on PCW critical violation
                color = COLOR_CRIT_BGR if (frame_idx % 2 == 0) else COLOR_WARN_BGR
                thickness = 3
            elif is_fcw_target:
                color = COLOR_CRIT_BGR
                thickness = 3
            elif det.class_name in VEHICLE_CLASSES:
                color = COLOR_VEHICLE_BGR
            elif det.class_name in VRU_CLASSES:
                color = COLOR_VRU_BGR
            elif det.class_name in SIGNAL_CLASSES:
                color = COLOR_SIGNAL_BGR
            else:
                color = COLOR_TEXT_WHITE

            cv2.rectangle(canvas, (ix1, iy1), (ix2, iy2), color, thickness)

            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            bg_y1 = max(0, iy1 - th - 6)
            bg_y2 = max(th + 6, iy1)
            cv2.rectangle(canvas, (ix1, bg_y1), (min(width - 1, ix1 + tw + 6), bg_y2), color, -1)
            cv2.putText(
                canvas,
                label,
                (ix1 + 3, bg_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (15, 15, 15),
                1,
                cv2.LINE_AA,
            )

    def _render_status_panel(
        self,
        canvas: np.ndarray,
        analysis: FrameAnalysis | None,
        fps: float | None,
    ) -> None:
        """Draw compact top-left HUD status box showing FPS, LDW, and FCW states."""
        ldw_status = analysis.ldw_status if analysis is not None else "N/A"
        fcw_status = analysis.fcw_status if analysis is not None else "N/A"
        fps_text = f"FPS: {fps:.1f}" if fps is not None else "FPS: --"

        panel_w, panel_h = 225, 92
        h, w = canvas.shape[:2]
        if w < panel_w + 16 or h < panel_h + 16:
            return

        x0, y0 = 10, 10
        sub = canvas[y0 : y0 + panel_h, x0 : x0 + panel_w]
        bg = np.full_like(sub, COLOR_PANEL_BG)
        canvas[y0 : y0 + panel_h, x0 : x0 + panel_w] = cv2.addWeighted(sub, 0.35, bg, 0.65, 0.0)
        cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h), (90, 90, 90), 1)

        cv2.putText(
            canvas,
            fps_text,
            (x0 + 10, y0 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_TEXT_WHITE,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"LDW: {ldw_status}",
            (x0 + 10, y0 + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _status_color(ldw_status),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"FCW: {fcw_status}",
            (x0 + 10, y0 + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _status_color(fcw_status),
            2,
            cv2.LINE_AA,
        )

    def _render_critical_banner(self, canvas: np.ndarray, alerts: list[ADASAlert]) -> None:
        """Draw prominent top-center banner when a CRITICAL alert is active."""
        crit_alerts = [a for a in alerts if a.severity.upper() == "CRITICAL"]
        if not crit_alerts:
            return

        height, width = canvas.shape[:2]
        if width < 260 or height < 80:
            return

        raw_msg = crit_alerts[0].message
        banner_text = f"! {strip_accents_for_cv2(raw_msg)}"

        (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        pad_x, pad_y = 18, 10
        box_w = min(width - 20, tw + 2 * pad_x)
        box_h = th + 2 * pad_y

        bx1 = max(10, (width - box_w) // 2)
        by1 = 12
        bx2 = min(width - 10, bx1 + box_w)
        by2 = min(height - 10, by1 + box_h)

        roi = canvas[by1:by2, bx1:bx2]
        red_fill = np.full_like(roi, COLOR_CRIT_BGR)
        canvas[by1:by2, bx1:bx2] = cv2.addWeighted(roi, 0.25, red_fill, 0.75, 0.0)
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), COLOR_TEXT_WHITE, 2)

        text_x = max(bx1 + 6, bx1 + (box_w - tw) // 2)
        text_y = by1 + pad_y + th
        cv2.putText(
            canvas,
            banner_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            COLOR_TEXT_WHITE,
            2,
            cv2.LINE_AA,
        )


class EventLogger:
    """CSV event logger recording ADAS alerts according to schema in docs/06."""

    CSV_COLUMNS = (
        "timestamp_sec",
        "frame_idx",
        "alert_type",
        "severity",
        "message",
        "extra_info",
    )

    def __init__(self, csv_path: str | Path | None = None) -> None:
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self.records: list[dict[str, Any]] = []
        self._header_written = False

        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()
            self._header_written = True

    def log(self, analysis: FrameAnalysis) -> None:
        """Record any alerts emitted in FrameAnalysis into memory and optional CSV file."""
        if not analysis.alerts:
            return

        new_rows: list[dict[str, Any]] = []
        for alert in analysis.alerts:
            extra = (
                f"bbox={tuple(round(v, 1) for v in alert.target_bbox)}"
                if alert.target_bbox is not None
                else f"ldw={analysis.ldw_status};fcw={analysis.fcw_status}"
            )
            row = {
                "timestamp_sec": round(float(analysis.timestamp_sec), 3),
                "frame_idx": int(analysis.frame_idx),
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "extra_info": extra,
            }
            self.records.append(row)
            new_rows.append(row)

        if self.csv_path is not None and new_rows:
            mode = "a" if self._header_written else "w"
            with self.csv_path.open(mode, encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.CSV_COLUMNS)
                if not self._header_written:
                    writer.writeheader()
                    self._header_written = True
                writer.writerows(new_rows)
