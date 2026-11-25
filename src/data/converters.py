"""Bộ chuyển đổi nhãn BDD100K (docs/03 §2–§4).

1. Detection JSON (Scalabel) -> YOLO ``.txt`` 9 lớp theo ``DET_CLASSES``::

       <class_id> <x_center> <y_center> <width> <height>     # chuẩn hoá [0, 1]

2. Mask BDD100K -> mask 3 kênh nhị phân theo ``SEG_CHANNELS``
   ``(drivable_direct, drivable_alternative, lane_marking)``:

   * Drivable: mask id PNG (``labels/drivable/masks/<split>/*.png``), giá trị pixel
     direct / alternative lấy từ ``DrivableIds`` — MẶC ĐỊNH 0 / 1 / nền 2, phải
     xác nhận bằng ``scripts/inspect_bdd.py`` trước khi chạy thật. Không có mask
     thì vẽ từ polygon ``area/drivable`` / ``area/alternative`` trong JSON.
   * Lane: mask PNG (``labels/lane/masks``) -> pixel ≠ nền (nền 255 hoặc 0, tự nhận
     diện hoặc truyền vào). Không có mask thì vẽ polyline ``poly2d`` của JSON lane.

Mask lưu ra đĩa: PNG 3 kênh uint8 {0, 255}; ``load_mask`` trả {0, 1}.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from src.utils.constants import DET_CLASS_TO_ID, NUM_SEG_CHANNELS, SEG_CHANNEL_TO_ID

# ---------------------------------------------------------------------------
# Ánh xạ category BDD100K -> DET_CLASSES (docs/03 §2)
# ---------------------------------------------------------------------------
BDD_CATEGORY_ALIASES: dict[str, str] = {
    "motor": "motorcycle",
    "bike": "bicycle",
    "person": "pedestrian",
    "traffic light": "traffic_light",
    "traffic sign": "traffic_sign",
}
# Category bị bỏ có chủ đích (docs/03 §2) — không coi là "lạ" khi inspect
BDD_IGNORED_CATEGORIES: frozenset[str] = frozenset({"train", "other vehicle", "trailer", "other person"})

BDD_IMAGE_SIZE: tuple[int, int] = (1280, 720)  # (W, H)


@dataclass(frozen=True)
class DrivableIds:
    """Giá trị pixel trong mask drivable id. Mặc định theo bản BDD chính thức — XÁC NHẬN bằng inspect."""

    direct: int = 0
    alternative: int = 1
    background: int = 2


DEFAULT_DRIVABLE_IDS = DrivableIds()


def normalize_category(name: str) -> str:
    """Tên category BDD -> tên lớp dự án (lower-case + alias, vd "traffic light" -> "traffic_light")."""
    name = name.strip().lower()
    return BDD_CATEGORY_ALIASES.get(name, name)


# ===========================================================================
# Detection -> YOLO
# ===========================================================================
@dataclass(frozen=True)
class YoloBox:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float

    def to_line(self) -> str:
        return f"{self.class_id} {self.xc:.6f} {self.yc:.6f} {self.w:.6f} {self.h:.6f}"


def box2d_to_yolo(
    box2d: dict[str, float],
    class_id: int,
    img_w: int = BDD_IMAGE_SIZE[0],
    img_h: int = BDD_IMAGE_SIZE[1],
    min_size: float = 0.0,
) -> YoloBox | None:
    """Đổi box2d (x1,y1,x2,y2 pixel) -> YoloBox. Trả None nếu box suy biến.

    Theo spec: clip vào trong ảnh (toạ độ chuẩn hoá ∈ [0, 1]) rồi loại box có
    width ≤ 0 hoặc height ≤ 0. ``min_size`` (pixel) cho phép lọc chặt hơn nếu cần.
    """
    x1 = max(0.0, min(float(box2d["x1"]), img_w))
    y1 = max(0.0, min(float(box2d["y1"]), img_h))
    x2 = max(0.0, min(float(box2d["x2"]), img_w))
    y2 = max(0.0, min(float(box2d["y2"]), img_h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0 or bw < min_size or bh < min_size:
        return None
    return YoloBox(
        class_id=class_id,
        xc=(x1 + bw / 2) / img_w,
        yc=(y1 + bh / 2) / img_h,
        w=bw / img_w,
        h=bh / img_h,
    )


def frame_to_yolo(
    frame: dict[str, Any],
    img_w: int = BDD_IMAGE_SIZE[0],
    img_h: int = BDD_IMAGE_SIZE[1],
    class_to_id: dict[str, int] | None = None,
    min_size: float = 0.0,
) -> list[YoloBox]:
    """Chuyển 1 frame BDD -> danh sách YoloBox (bỏ qua lớp ngoài danh sách)."""
    class_to_id = class_to_id or DET_CLASS_TO_ID
    out: list[YoloBox] = []
    for lab in frame.get("labels") or []:
        box2d = lab.get("box2d")
        if not box2d:
            continue  # nhãn lane/drivable dạng poly2d nằm chung file cũ
        cat = normalize_category(lab.get("category", ""))
        cid = class_to_id.get(cat)
        if cid is None:
            continue
        yb = box2d_to_yolo(box2d, cid, img_w, img_h, min_size)
        if yb is not None:
            out.append(yb)
    return out


def yolo_lines(boxes: Iterable[YoloBox]) -> str:
    """Danh sách YoloBox -> nội dung file .txt."""
    lines = [b.to_line() for b in boxes]
    return "\n".join(lines) + ("\n" if lines else "")


def parse_yolo_line(line: str) -> YoloBox:
    """1 dòng YOLO -> YoloBox."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"Dòng YOLO không hợp lệ: {line!r}")
    return YoloBox(int(parts[0]), *(float(p) for p in parts[1:]))


def yolo_to_xyxy(box: YoloBox, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Ngược lại: YoloBox -> (x1, y1, x2, y2) pixel. Dùng cho augmentation & test."""
    x1 = (box.xc - box.w / 2) * img_w
    y1 = (box.yc - box.h / 2) * img_h
    x2 = (box.xc + box.w / 2) * img_w
    y2 = (box.yc + box.h / 2) * img_h
    return x1, y1, x2, y2

# ===========================================================================
# Segmentation -> mask 3 kênh
# ===========================================================================
CH_DRIVABLE = SEG_CHANNEL_TO_ID["drivable_direct"]
CH_ALTERNATIVE = SEG_CHANNEL_TO_ID["drivable_alternative"]
CH_LANE = SEG_CHANNEL_TO_ID["lane_marking"]


def empty_mask(img_w: int = BDD_IMAGE_SIZE[0], img_h: int = BDD_IMAGE_SIZE[1]) -> np.ndarray:
    """Mask rỗng uint8 (H, W, NUM_SEG_CHANNELS)."""
    return np.zeros((img_h, img_w, NUM_SEG_CHANNELS), dtype=np.uint8)


def drivable_png_to_channels(
    drivable: np.ndarray, mask: np.ndarray, ids: DrivableIds = DEFAULT_DRIVABLE_IDS
) -> np.ndarray:
    """Điền kênh direct/alternative từ mask drivable id (ảnh 1 kênh). ``mask`` được ghi trực tiếp."""
    if drivable.ndim == 3:
        drivable = drivable[..., 0]
    if drivable.shape[:2] != mask.shape[:2]:
        drivable = cv2.resize(drivable, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask[..., CH_DRIVABLE] = np.where(drivable == ids.direct, 255, 0).astype(np.uint8)
    mask[..., CH_ALTERNATIVE] = np.where(drivable == ids.alternative, 255, 0).astype(np.uint8)
    return mask


def detect_background(m: np.ndarray) -> int:
    """Giá trị nền của mask lane = giá trị xuất hiện nhiều nhất (thường 255 hoặc 0)."""
    return int(np.bincount(m.ravel(), minlength=256).argmax())


def lane_png_to_channel(lane: np.ndarray, mask: np.ndarray, background: int | None = None) -> np.ndarray:
    """Điền kênh lane từ mask lane dạng ảnh: foreground = pixel ≠ nền."""
    if lane.ndim == 3:
        lane = lane[..., 0]
    if lane.shape[:2] != mask.shape[:2]:
        lane = cv2.resize(lane, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    bg = detect_background(lane) if background is None else background
    mask[..., CH_LANE] = np.where(lane != bg, 255, 0).astype(np.uint8)
    return mask


def _poly_vertices(poly: dict[str, Any]) -> np.ndarray:
    pts = np.asarray(poly.get("vertices", []), dtype=np.float32)
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def draw_drivable_polygons(frame: dict[str, Any], mask: np.ndarray) -> np.ndarray:
    """Vẽ drivable từ polygon trong JSON (format cũ ``area/drivable``)."""
    for lab in frame.get("labels") or []:
        cat = str(lab.get("category", "")).lower()
        area_type = str((lab.get("attributes") or {}).get("areaType", "")).lower()
        # format mới: "area/drivable" | "area/alternative"
        # format cũ : "drivable area" + attributes.areaType = "direct" | "alternative"
        if cat == "area/drivable" or (cat == "drivable area" and area_type != "alternative"):
            ch = CH_DRIVABLE
        elif cat == "area/alternative" or (cat == "drivable area" and area_type == "alternative"):
            ch = CH_ALTERNATIVE
        else:
            continue
        layer = np.ascontiguousarray(mask[..., ch])
        for poly in lab.get("poly2d") or []:
            pts = _poly_vertices(poly)
            if len(pts) >= 3:
                cv2.fillPoly(layer, [pts], 255)
        mask[..., ch] = layer
    return mask


def draw_lanes(frame: dict[str, Any], mask: np.ndarray, lane_thickness: int = 8) -> np.ndarray:
    """Vẽ polyline làn đường vào kênh lane.

    BDD dùng ``types`` cho từng đỉnh: ``L`` = điểm thẳng, ``C`` = điểm điều khiển
    Bezier. Ở đây nối thẳng các đỉnh (đủ chính xác với nét dày 8px @1280x720).
    """
    layer = np.ascontiguousarray(mask[..., CH_LANE])
    for lab in frame.get("labels") or []:
        cat = str(lab.get("category", "")).lower()
        if not cat.startswith("lane"):
            continue
        for poly in lab.get("poly2d") or []:
            pts = _poly_vertices(poly)
            if len(pts) < 2:
                continue
            closed = bool(poly.get("closed", False))
            cv2.polylines(layer, [pts], closed, 255, thickness=max(1, int(lane_thickness)), lineType=cv2.LINE_8)
    mask[..., CH_LANE] = layer
    return mask


def build_mask(
    img_w: int = BDD_IMAGE_SIZE[0],
    img_h: int = BDD_IMAGE_SIZE[1],
    drivable_png: np.ndarray | None = None,
    drivable_ids: DrivableIds = DEFAULT_DRIVABLE_IDS,
    drivable_frame: dict[str, Any] | None = None,
    lane_frame: dict[str, Any] | None = None,
    lane_thickness: int = 8,
    lane_png: np.ndarray | None = None,
    lane_background: int | None = None,
) -> np.ndarray:
    """Ghép đủ 3 kênh cho 1 ảnh. Trả về uint8 (H, W, 3) giá trị {0, 255}."""
    mask = empty_mask(img_w, img_h)
    if drivable_png is not None:
        drivable_png_to_channels(drivable_png, mask, drivable_ids)
    elif drivable_frame is not None:
        draw_drivable_polygons(drivable_frame, mask)
    if lane_png is not None:
        lane_png_to_channel(lane_png, mask, lane_background)
    elif lane_frame is not None:
        draw_lanes(lane_frame, mask, lane_thickness)
    return mask


def save_mask(path: str, mask: np.ndarray) -> None:
    """Lưu mask (H, W, 3) {0,255} ra PNG; kênh 0 nằm ở R khi mở bằng trình xem ảnh."""
    # cv2 lưu theo BGR -> đảo để kênh 0 nằm ở R khi mở bằng trình xem ảnh
    ok = cv2.imwrite(str(path), mask[..., ::-1])
    if not ok:
        raise IOError(f"Không ghi được mask: {path}")


def load_mask(path: str, binary: bool = True) -> np.ndarray:
    """Đọc mask đã lưu -> (H, W, 3). ``binary=True`` trả về {0, 1}."""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 2:
        m = np.repeat(m[..., None], NUM_SEG_CHANNELS, axis=2)
    m = m[..., ::-1].copy()
    return (m > 127).astype(np.uint8) if binary else m
