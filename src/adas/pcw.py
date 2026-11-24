from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PCWConfig:
    target_classes: tuple[str, ...] = ("pedestrian", "rider", "bicycle")
    min_confidence: float = 0.35
    camera_offset_ratio: float = 0.0
    roi_top_y_ratio: float = 0.40                 # Mép trên vùng xét PCW
    roi_bot_y_ratio: float = 0.96                 # Mép dưới vùng xét PCW
    corridor_top_half_w_ratio: float = 0.14       # Nửa rộng hành lang mặc định ở trên
    corridor_bot_half_w_ratio: float = 0.38       # Nửa rộng hành lang mặc định ở dưới
    use_drivable_mask: bool = True                # Dùng drivable_direct để tạo vùng nguy hiểm nếu có
    drivable_expand_px: int = 24                  # Số pixel mở rộng ngang quanh drivable_direct
    overlap_thresh: float = 0.20                  # Tỷ lệ bbox giao với vùng nguy hiểm tối thiểu
    warning_height_ratio: float = 0.12            # Chiều cao bbox / H ngưỡng WARNING
    critical_height_ratio: float = 0.25           # Chiều cao bbox / H ngưỡng CRITICAL
    critical_area_ratio: float = 0.045            # Diện tích chuẩn hóa ngưỡng CRITICAL


class PedestrianCollisionWarning:
    """Bộ cảnh báo người đi bộ & xe thô sơ (PCW) thuần NumPy."""

    def __init__(self, config: PCWConfig | None = None) -> None:
        self.cfg = config or PCWConfig()

    def _build_fallback_corridor(self, img_h: int, img_w: int) -> np.ndarray:
        """Tạo mask hành lang hình thang dự phòng khi không dùng drivable_direct."""
        corridor = np.zeros((img_h, img_w), dtype=bool)
        y_top = int(np.clip(img_h * self.cfg.roi_top_y_ratio, 0, img_h))
        y_bot = int(np.clip(img_h * self.cfg.roi_bot_y_ratio, 0, img_h))
        if y_bot <= y_top:
            return corridor

        x_ego = (0.5 * img_w) + self.cfg.camera_offset_ratio * (0.5 * img_w)
                                                  # Tâm X của xe (có bù lệch camera)
        w_top = img_w * self.cfg.corridor_top_half_w_ratio
        w_bot = img_w * self.cfg.corridor_bot_half_w_ratio

        ys = np.arange(y_top, y_bot, dtype=np.float64)
        denom = float(max(1, y_bot - y_top - 1))
        ratios = (ys - y_top) / denom             # Hệ số nội suy dọc [0, 1]
        half_widths = w_top + ratios * (w_bot - w_top)
                                                  # Nửa bề rộng hành lang theo từng hàng Y

        xs = np.arange(img_w, dtype=np.float64)[None, :]
        in_x = np.abs(xs - x_ego) <= half_widths[:, None]
                                                  # Broadcasting tạo vùng hình thang 2D (y_bot - y_top, img_w)
        corridor[y_top:y_bot, :] = in_x
        return corridor

    def _expand_horizontal_mask(self, base_mask: np.ndarray, pad_x: int) -> np.ndarray:
        """Mở rộng vùng mask theo chiều ngang bằng thuần NumPy (không cần SciPy/OpenCV)."""
        if pad_x <= 0 or not np.any(base_mask):
            return base_mask

        h, w = base_mask.shape
        p = min(int(pad_x), w)
        padded = np.pad(base_mask, ((0, 0), (p, p)), mode="constant", constant_values=False)
                                                  # Đệm False ở 2 biên trái/phải để trượt cửa sổ ngang
        expanded = np.zeros_like(base_mask, dtype=bool)

        # Dùng cumulative sum theo trục X để tính nhanh vùng phủ trong [-p, +p] trong O(1) mỗi pixel
        csum = np.cumsum(padded.astype(np.int32), axis=1)
        # Với mỗi cột j in [0, w-1], vùng tương ứng trên padded là [j, j + 2*p]
        right_idx = np.arange(2 * p, w + 2 * p)
        left_idx = np.arange(0, w)
        left_sum = np.where(left_idx > 0, csum[:, np.maximum(0, left_idx - 1)], 0)
        window_counts = csum[:, right_idx] - left_sum
        expanded[:, :] = window_counts > 0
        return expanded

    def _build_danger_zone(
        self,
        img_h: int,
        img_w: int,
        drivable_direct: np.ndarray | None = None,
    ) -> np.ndarray:
        """Tạo mặt nạ vùng nguy hiểm phía trước cho người đi bộ & xe thô sơ."""
        fallback = self._build_fallback_corridor(img_h, img_w)

        if not self.cfg.use_drivable_mask or drivable_direct is None:
            return fallback

        mask = np.asarray(drivable_direct)
        if mask.ndim != 2 or mask.shape != (img_h, img_w):
            return fallback

        active = mask > 0
        if not np.any(active):                     # Nếu mask rỗng thì dùng hành lang dự phòng
            return fallback

        y_top = int(np.clip(img_h * self.cfg.roi_top_y_ratio, 0, img_h))
        y_bot = int(np.clip(img_h * self.cfg.roi_bot_y_ratio, 0, img_h))
        vertical_band = np.zeros((img_h, img_w), dtype=bool)
        if y_bot > y_top:
            vertical_band[y_top:y_bot, :] = True

        expanded_direct = self._expand_horizontal_mask(active, self.cfg.drivable_expand_px)
                                                  # Mở rộng biên ngang quanh làn đường đi được
        danger = (expanded_direct & vertical_band) | (fallback & vertical_band)
                                                  # Kết hợp làn mở rộng và hành lang dự phòng trong vùng ROI
        return danger

    def update(
        self,
        detections: list[dict],
        img_shape: tuple[int, int],
        drivable_direct: np.ndarray | None = None,
    ) -> dict:
        img_h, img_w = int(img_shape[0]), int(img_shape[1])
        img_area = float(max(1, img_h * img_w))
        danger_zone = self._build_danger_zone(img_h, img_w, drivable_direct=drivable_direct)

        hazards: list[dict] = []                  # Danh sách đối tượng VRU nằm trong vùng nguy hiểm

        for det in detections:
            label = str(det.get("label", ""))
            conf = float(det.get("confidence", 0.0))
            if label not in self.cfg.target_classes or conf < self.cfg.min_confidence:
                continue                          # Chỉ xét pedestrian, rider, bicycle đủ ngưỡng confidence

            bbox = np.asarray(det["bbox"], dtype=np.float64)
            x1 = int(np.clip(np.floor(bbox[0]), 0, img_w))
            y1 = int(np.clip(np.floor(bbox[1]), 0, img_h))
            x2 = int(np.clip(np.ceil(bbox[2]), 0, img_w))
            y2 = int(np.clip(np.ceil(bbox[3]), 0, img_h))

            box_w = max(0, x2 - x1)
            box_h = max(0, y2 - y1)
            if box_w == 0 or box_h == 0:
                continue

            box_area = float(box_w * box_h)
            height_ratio = box_h / float(max(1, img_h))
                                                  # Chiều cao bbox chuẩn hóa theo H (ước lượng độ gần)
            area_ratio = box_area / img_area      # Diện tích bbox chuẩn hóa theo W * H

            box_danger = danger_zone[y1:y2, x1:x2]
                                                  # Cắt vùng nguy hiểm tương ứng với bounding box
            overlap_px = int(np.sum(box_danger))
            overlap_ratio = overlap_px / box_area
                                                  # Tỷ lệ phần bbox xâm nhập vùng nguy hiểm

            # Kiểm tra thêm phần nửa dưới bbox (gần chân / bánh xe đạp) có chạm vùng nguy hiểm không
            half_y = y1 + (box_h // 2)
            lower_danger = danger_zone[half_y:y2, x1:x2]
            lower_hit = bool(np.any(lower_danger))

            is_in_danger = (overlap_ratio >= self.cfg.overlap_thresh) or (
                lower_hit and height_ratio >= self.cfg.warning_height_ratio
            )
            if not is_in_danger:
                continue

            severity = "WARNING"
            if (
                height_ratio >= self.cfg.critical_height_ratio
                or area_ratio >= self.cfg.critical_area_ratio
            ):
                severity = "CRITICAL"

            hazards.append(
                {
                    "label": label,
                    "confidence": conf,
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "overlap_ratio": float(overlap_ratio),
                    "height_ratio": float(height_ratio),
                    "area_ratio": float(area_ratio),
                    "severity": severity,
                }
            )

        highest_severity = "NONE"
        primary_hazard = None
        rank_map = {"NONE": 0, "WARNING": 1, "CRITICAL": 2}

        for item in hazards:
            if rank_map[item["severity"]] > rank_map[highest_severity]:
                highest_severity = item["severity"]
                primary_hazard = item
            elif item["severity"] != "NONE" and primary_hazard is not None:
                if (
                    item["severity"] == highest_severity
                    and item["height_ratio"] > primary_hazard["height_ratio"]
                ):
                    primary_hazard = item         # Cùng mức độ thì ưu tiên đối tượng lớn hơn (gần xe hơn)

        return {
            "alert": highest_severity != "NONE",
            "severity": highest_severity,
            "target": primary_hazard,
            "hazards": hazards,
        }