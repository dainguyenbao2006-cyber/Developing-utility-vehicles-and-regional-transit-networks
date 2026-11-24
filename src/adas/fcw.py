from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class FCWConfig:
    target_classes: tuple[str, ...] = ("car", "truck", "bus", "motorcycle")
    min_confidence: float = 0.35
    camera_offset_ratio: float = 0.0
    corridor_top_y_ratio: float = 0.45            # Mép trên hành lang (gần đường chân trời)
    corridor_bot_y_ratio: float = 0.95            # Mép dưới hành lang (gần mũi xe)
    corridor_top_half_w_ratio: float = 0.08       # Nửa bề rộng hành lang ở mép trên (% chiều rộng ảnh)
    corridor_bot_half_w_ratio: float = 0.32       # Nửa bề rộng hành lang ở mép dưới (% chiều rộng ảnh)
    use_drivable_mask: bool = False               # Có kiểm tra thêm giao với drivable_direct hay không
    drivable_min_ratio: float = 0.20              # Tỷ lệ pixel drivable_direct tối thiểu quanh tâm đáy bbox
    history_len: int = 8                          # Số mốc lịch sử tối đa lưu cho mỗi track xe
    lookback_steps: int = 3                       # Số bước lùi để tính delta_area / delta_t
    min_dt_sec: float = 1e-3                      # Ngưỡng epsilon tránh chia cho 0 khi timestamp sát nhau
    growth_thresh: float = 0.08                   # Ngưỡng tốc độ tăng diện tích/giây để bật WARNING
    critical_growth_thresh: float = 0.20          # Ngưỡng tốc độ tăng diện tích/giây để bật CRITICAL
    warning_area_thresh: float = 0.06             # Diện tích chuẩn hóa cảnh báo WARNING
    critical_area_thresh: float = 0.14            # Diện tích chuẩn hóa cảnh báo CRITICAL
    match_max_dist_ratio: float = 0.12            # Bán kính tối đa (theo đường chéo ảnh) để ghép track


class ForwardCollisionWarning:
    """Bộ cảnh báo va chạm phía trước (FCW Proxy) thuần NumPy."""

    def __init__(self, config: FCWConfig | None = None) -> None:
        self.cfg = config or FCWConfig()
        # Mỗi phần tử track: {"label": str, "history": deque[(t, area, cx, cy)]}
        self._tracks: list[dict] = []
        self._next_track_id: int = 1

    def reset(self) -> None:
        """Xóa lịch sử track khi chuyển video mới hoặc chạy test độc lập."""
        self._tracks.clear()
        self._next_track_id = 1

    def _corridor_half_width_at_y(self, y: float, img_h: int, img_w: int) -> float:
        y_top = img_h * self.cfg.corridor_top_y_ratio              # Tọa độ Y mép trên hành lang
        y_bot = img_h * self.cfg.corridor_bot_y_ratio              # Tọa độ Y mép dưới hành lang
        w_top = img_w * self.cfg.corridor_top_half_w_ratio         # Nửa rộng hành lang tại y_top
        w_bot = img_w * self.cfg.corridor_bot_half_w_ratio         # Nửa rộng hành lang tại y_bot

        if y_bot <= y_top:                                           # Tránh chia cho 0 nếu cấu hình suy biến
            return w_bot

        ratio = float(np.clip((y - y_top) / (y_bot - y_top), 0.0, 1.0))
                                                                     # Tỷ lệ nội suy dọc theo chiều cao hành lang [0, 1]
        return float(w_top + ratio * (w_bot - w_top))                # Nội suy tuyến tính nửa bề rộng hành lang tại y

    def _is_in_corridor(
        self,
        bbox: np.ndarray,
        img_h: int,
        img_w: int,
        drivable_direct: np.ndarray | None = None,
    ) -> tuple[bool, float, float]:
        x1, y1, x2, y2 = [float(v) for v in bbox]                    # Giải nén tọa độ bounding box
        if x2 <= x1 or y2 <= y1:                                     # Bỏ qua box không hợp lệ
            return False, 0.0, 0.0

        cx = 0.5 * (x1 + x2)                                         # Tâm X của bbox
        cy = y2                                                      # Điểm đáy bbox (tiếp xúc mặt đường)

        y_top = img_h * self.cfg.corridor_top_y_ratio
        y_bot = img_h * self.cfg.corridor_bot_y_ratio
        if not (y_top <= cy <= y_bot):                               # Kiểm tra điểm đáy có nằm trong dải Y của hành lang
            return False, cx, cy

        x_ego = (0.5 * img_w) + self.cfg.camera_offset_ratio * (0.5 * img_w)
                                                                     # Tâm X của xe (có bù lệch camera)
        half_w = self._corridor_half_width_at_y(cy, img_h, img_w)   # Nửa bề rộng hành lang tại độ cao cy
        if abs(cx - x_ego) > half_w:                                # Ngoài biên trái/phải của hình thang hành lang
            return False, cx, cy

        if self.cfg.use_drivable_mask and drivable_direct is not None:
            mask = np.asarray(drivable_direct)
            if mask.ndim == 2 and mask.shape == (img_h, img_w):
                r = 8                                                # Vùng lân cận nhỏ quanh điểm tiếp xúc (cx, cy)
                xi, yi = int(np.clip(cx, 0, img_w - 1)), int(np.clip(cy, 0, img_h - 1))
                x0, x1_m = max(0, xi - r), min(img_w, xi + r + 1)
                y0, y1_m = max(0, yi - r), min(img_h, yi + r + 1)
                if x1_m > x0 and y1_m > y0:
                    direct_ratio = float(np.mean(mask[y0:y1_m, x0:x1_m] > 0))
                    if direct_ratio < self.cfg.drivable_min_ratio:   # Điểm tiếp xúc không nằm trên làn đường đi được
                        return False, cx, cy

        return True, cx, cy                                          # Xe nằm trong Ego Corridor

    def _match_or_create_track(
        self,
        label: str,
        timestamp: float,
        area_ratio: float,
        cx: float,
        cy: float,
        img_diag: float,
    ) -> dict:
        max_dist = self.cfg.match_max_dist_ratio * img_diag         # Ngưỡng khoảng cách tối đa để coi là cùng 1 xe
        best_track = None
        best_dist = float("inf")

        for tr in self._tracks:                                       # Duyệt các track xe đang theo dõi
            if tr["label"] != label or len(tr["history"]) == 0:
                continue
            last_t, _, last_cx, last_cy = tr["history"][-1]
            if timestamp < last_t:                                   # Bỏ qua nếu timestamp lùi bất thường
                continue
            dist = float(np.hypot(cx - last_cx, cy - last_cy))       # Khoảng cách Euclidean giữa 2 tâm liên tiếp
            if dist <= max_dist and dist < best_dist:
                best_dist = dist
                best_track = tr

        if best_track is None:                                       # Chưa khớp xe nào -> tạo track mới
            best_track = {
                "id": self._next_track_id,
                "label": label,
                "history": deque(maxlen=self.cfg.history_len),
            }
            self._next_track_id += 1
            self._tracks.append(best_track)

        best_track["history"].append((float(timestamp), float(area_ratio), float(cx), float(cy)))
                                                                     # Ghi nhận mẫu mới vào cuối buffer lịch sử
        return best_track

    def _compute_growth_rate(self, history: deque) -> float:
        if len(history) < 2:                                         # Cần ít nhất 2 điểm thời gian để tính tốc độ biến thiên
            return 0.0

        t_now, a_now, _, _ = history[-1]                             # Mẫu hiện tại
        idx_prev = max(0, len(history) - 1 - self.cfg.lookback_steps)
                                                                     # Chỉ số mẫu quá khứ cách lookback_steps bước
        if idx_prev == len(history) - 1:
            idx_prev = 0

        t_prev, a_prev, _, _ = history[idx_prev]                     # Mẫu quá khứ
        dt = max(float(t_now - t_prev), self.cfg.min_dt_sec)         # Khoảng thời gian delta_t (có chặn min_dt_sec)
        return float((a_now - a_prev) / dt)                          # Tốc độ phình diện tích chuẩn hóa trên mỗi giây

    def update(
        self,
        detections: list[dict],
        img_shape: tuple[int, int],
        timestamp: float,
        drivable_direct: np.ndarray | None = None,
    ) -> dict:
        img_h, img_w = int(img_shape[0]), int(img_shape[1])          # Chiều cao và chiều rộng frame gốc
        img_area = float(max(1, img_h * img_w))                      # Tổng diện tích khung hình
        img_diag = float(np.hypot(img_h, img_w))                     # Đường chéo khung hình để chuẩn hóa khoảng cách

        candidates: list[dict] = []                                  # Danh sách phương tiện trong hành lang được đánh giá

        for det in detections:                                       # Duyệt từng detection từ YOLO11
            label = str(det.get("label", ""))
            conf = float(det.get("confidence", 0.0))
            if label not in self.cfg.target_classes or conf < self.cfg.min_confidence:
                continue                                             # Lọc đúng class phương tiện & đủ confidence

            bbox = np.asarray(det["bbox"], dtype=np.float64)
            in_corridor, cx, cy = self._is_in_corridor(
                bbox=bbox,
                img_h=img_h,
                img_w=img_w,
                drivable_direct=drivable_direct,
            )
            if not in_corridor:                                      # Bỏ qua xe nằm ngoài Ego Corridor
                continue

            box_w = max(0.0, float(bbox[2] - bbox[0]))
            box_h = max(0.0, float(bbox[3] - bbox[1]))
            area_ratio = (box_w * box_h) / img_area                  # Diện tích chuẩn hóa a_t trong [0, 1]

            track = self._match_or_create_track(
                label=label,
                timestamp=float(timestamp),
                area_ratio=area_ratio,
                cx=cx,
                cy=cy,
                img_diag=img_diag,
            )
            growth_rate = self._compute_growth_rate(track["history"])
                                                                     # Tính delta_Area / delta_t

            severity = "NONE"                                        # Đánh giá mức độ nguy hiểm cho từng xe
            if area_ratio >= self.cfg.critical_area_thresh or growth_rate >= self.cfg.critical_growth_thresh:
                severity = "CRITICAL"
            elif growth_rate >= self.cfg.growth_thresh or area_ratio >= self.cfg.warning_area_thresh:
                severity = "WARNING"

            candidates.append(
                {
                    "track_id": track["id"],
                    "label": label,
                    "confidence": conf,
                    "bbox": bbox.tolist(),
                    "area_ratio": area_ratio,
                    "growth_rate": growth_rate,
                    "severity": severity,
                }
            )

        highest_severity = "NONE"                                    # Chọn mức cảnh báo cao nhất trong frame
        primary_target = None
        rank_map = {"NONE": 0, "WARNING": 1, "CRITICAL": 2}

        for item in candidates:
            if rank_map[item["severity"]] > rank_map[highest_severity]:
                highest_severity = item["severity"]
                primary_target = item
            elif item["severity"] != "NONE" and primary_target is not None:
                if item["severity"] == highest_severity and item["area_ratio"] > primary_target["area_ratio"]:
                    primary_target = item                            # Cùng mức độ thì ưu tiên xe có diện tích lớn hơn (gần hơn)

        return {
            "alert": highest_severity != "NONE",
            "severity": highest_severity,
            "target": primary_target,
            "corridor_vehicles": candidates,
        }