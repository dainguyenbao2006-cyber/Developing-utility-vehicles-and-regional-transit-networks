# 09 — ADAS logic

Tất cả module trong `src/adas/adas/` **chỉ dùng NumPy/stdlib**, nhận dataclass từ `types.py`, trả `Alert` hoặc cấu trúc trung gian. Mọi ngưỡng đọc từ `cfg.adas.*` (xem `17`).

> ⚠️ Đây là **demo học thuật**. Không có calibration camera, không có tốc độ xe → các "cảnh báo" mang tính tham khảo. Xem `16`.

## 1. Khái niệm chung

- Hệ toạ độ ảnh: gốc trên-trái, x sang phải, y xuống dưới.
- "Điểm chân" (ground contact point) của đối tượng = `bottom_center` của bbox.
- **Ego region** = `seg.masks[0]` (drivable_direct), có thể dilate nhẹ (`adas.ego_dilate_px`).
- `vehicle_x` (tâm xe trên ảnh) = `W/2 + camera_offset_ratio * W`.

## 2. Hình học làn (`lane_geometry.py`)

```python
def estimate_lane_geometry(direct_mask: np.ndarray, cfg) -> LaneGeometry:
    """direct_mask: bool (H, W)."""
    H, W = direct_mask.shape
    y0, y1 = int(H * cfg.band[0]), int(H * cfg.band[1])       # mặc định (0.70, 0.90)
    veh_x = W / 2 + cfg.camera_offset_ratio * W
    lefts, rights = [], []
    for y in range(y0, y1, cfg.row_step):                       # row_step = 2
        xs = np.flatnonzero(direct_mask[y])
        if xs.size < cfg.min_row_pixels_ratio * W:              # 0.05
            continue
        run = _best_run(xs, veh_x)
        lefts.append(run[0]); rights.append(run[-1])
    if len(lefts) < cfg.min_rows:                               # 5
        return LaneGeometry(valid=False, vehicle_x=veh_x)
    l, r = float(np.median(lefts)), float(np.median(rights))
    width = r - l
    if width < cfg.min_width_ratio * W:                         # 0.15
        return LaneGeometry(valid=False, vehicle_x=veh_x)
    center = (l + r) / 2
    return LaneGeometry(True, l, r, width, center, veh_x, (veh_x - center) / width)

def _best_run(xs: np.ndarray, center: float) -> np.ndarray:
    """Run liên tiếp (không đứt) chứa/gần center nhất."""
    runs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
    def dist(r): return 0.0 if r[0] <= center <= r[-1] else min(abs(r[0] - center), abs(r[-1] - center))
    return min(runs, key=lambda r: (dist(r), -len(r)))
```

Ý nghĩa `offset_ratio`: ≈ 0 khi xe ở giữa làn; +0.5 khi tâm xe chạm biên phải; −0.5 khi chạm biên trái. (>0.5 hoặc <−0.5: đã ra khỏi làn.)

Lưu ý: nhãn `direct` của BDD đôi khi bao cả làn lân cận (đường không vạch) → `lane_width` lớn bất thường; `max_width_ratio` (0.9) để loại.

## 3. Lane Departure Warning (`ldw.py`)

Trạng thái: `NORMAL → WARNING → CRITICAL` với **hysteresis + debounce**.

Tham số (`adas.ldw`):

| Key | Mặc định | Ý nghĩa |
|---|---|---|
| `warn_offset` | 0.25 | |offset| ≥ → WARNING |
| `critical_offset` | 0.38 | |offset| ≥ → CRITICAL |
| `exit_factor` | 0.8 | Thoát trạng thái khi |offset| < ngưỡng × 0.8 |
| `min_frames` | 5 | Phải vượt ngưỡng liên tiếp N frame mới kích hoạt |
| `ema_alpha` | 0.4 | Làm mượt offset |
| `invalid_grace_frames` | 10 | Mất làn ≤ N frame: giữ trạng thái; lâu hơn → reset |
| `suppress_if_turn_signal` | false | (không có tín hiệu xi-nhan → bỏ qua) |

```python
class LaneDepartureWarning:
    def __init__(self, cfg): self.cfg = cfg; self.reset()
    def reset(self): self.ema = None; self.state = "NORMAL"; self.counter = 0; self.invalid = 0

    def update(self, lane: LaneGeometry, frame_idx: int, ts: float) -> Alert | None:
        if not lane.valid:
            self.invalid += 1
            if self.invalid > self.cfg.invalid_grace_frames: self.reset()
            return None
        self.invalid = 0
        o = lane.offset_ratio
        self.ema = o if self.ema is None else self.cfg.ema_alpha * o + (1 - self.cfg.ema_alpha) * self.ema
        a = abs(self.ema)
        target = ("CRITICAL" if a >= self.cfg.critical_offset else
                  "WARNING" if a >= self.cfg.warn_offset else "NORMAL")
        # hysteresis khi giảm trạng thái
        if self.state == "CRITICAL" and a >= self.cfg.critical_offset * self.cfg.exit_factor: target = "CRITICAL"
        elif self.state in ("WARNING", "CRITICAL") and a >= self.cfg.warn_offset * self.cfg.exit_factor:
            target = max(target, "WARNING", key=("NORMAL", "WARNING", "CRITICAL").index)
        # debounce khi tăng trạng thái
        order = ("NORMAL", "WARNING", "CRITICAL")
        if order.index(target) > order.index(self.state):
            self.counter += 1
            if self.counter >= self.cfg.min_frames: self.state = target; self.counter = 0
            else: return None
        else:
            self.counter = 0; self.state = target
        if self.state == "NORMAL": return None
        direction = "RIGHT" if self.ema > 0 else "LEFT"
        return Alert(AlertType[f"LANE_DEPARTURE_{direction}"],
                     Severity.CRITICAL if self.state == "CRITICAL" else Severity.WARNING,
                     f"Lệch làn sang {'phải' if direction == 'RIGHT' else 'trái'}",
                     frame_idx, ts, {"offset": round(self.ema, 3)})
```
(Code trên là tham chiếu logic; AI được phép refactor cho gọn miễn test `test_ldw.py` pass.)

Mở rộng: kiểm tra đè vạch liền/đứt khi mở rộng kênh lane (solid/dashed).

## 4. Forward Collision Warning proxy (`fcw.py`)

Không có khoảng cách thật ⇒ dùng 3 tín hiệu:

1. **Có nằm trong làn ego không**: điểm chân `(x, y_bottom)` thuộc `ego_region` (sau dilate) hoặc nằm trong hành lang hình thang quanh tâm ảnh khi mask mất (`fallback_corridor`).
2. **Độ gần** `proximity = y_bottom / H` (vật càng gần, chân càng thấp trong ảnh) và `rel_height = h / H`.
3. **TTC proxy** từ tốc độ phình to bbox (theo `track_id`):

   `TTC ≈ h / (dh/dt)` (giây), với `dh/dt` hồi quy tuyến tính trên `ttc_window` frame gần nhất (`t` tính bằng giây theo `timestamp_s`). Nếu `dh/dt ≤ eps` → `inf`.

Phân mức:

| Điều kiện (đối tượng ∈ VEHICLES, trong làn ego) | Mức |
|---|---|
| `proximity ≥ crit_proximity` (0.88) **hoặc** `TTC ≤ crit_ttc` (1.5 s) | CRITICAL |
| `proximity ≥ warn_proximity` (0.78) **hoặc** `TTC ≤ warn_ttc` (3.0 s) | WARNING |
| Ngược lại | không cảnh báo |

Chi tiết:
- Chỉ xét đối tượng có `conf ≥ adas.fcw.min_conf` (0.4) và `track_id` tồn tại ≥ `min_track_age` (3 frame) để tránh nhiễu.
- Chọn **đối tượng gần nhất** trong làn (y_bottom lớn nhất) làm "lead vehicle"; chỉ phát 1 alert/frame.
- Lưu `data={"track_id":..., "cls":..., "proximity":..., "ttc_s":...}`.
- Giữ lịch sử `dict[track_id] -> deque[(t, h)]`, xoá track không xuất hiện sau `track_ttl` frame.
- Với xe máy/xe đạp: dùng chung quy tắc, nhưng cho phép `lane_margin_px` rộng hơn vì chúng chạy lệch.

## 5. Pedestrian in path (`pedestrian.py`)

- Đối tượng ∈ VRU, `conf ≥ 0.35`.
- Điều kiện: điểm chân ∈ `ego_region` (dilate `ped_dilate_px`) **hoặc** trong hành lang fallback.
- Mức: `proximity ≥ 0.70` → WARNING, `≥ 0.82` → CRITICAL.
- Debounce `min_frames = 3`.

## 6. Cảnh báo giao thông (`traffic.py`)

### 6.1 Đèn đỏ
- Chọn `traffic_light` có `attrs["light_color"] == "red"`, `conf ≥ 0.4`.
- Chỉ xét đèn "liên quan": tâm x trong `[W*(0.5-rel_x), W*(0.5+rel_x)]` (mặc định `rel_x=0.3`), tâm y < `0.6*H`, chiều cao bbox ≥ `min_h_px` (12).
- Làm mượt màu theo `track_id` (đa số trong 5 frame) trước khi cảnh báo.
- Alert `TRAFFIC_LIGHT_RED`, WARNING.

### 6.2 Biển báo phía trước
- `traffic_sign`, `conf ≥ 0.5`, vùng trung tâm-phải mở rộng (xe chạy bên phải: `x_center ∈ [0.35W, 1.0W]`, tham số `sign_region_x`), chiều cao ≥ `min_h_px` (16).
- Alert INFO `TRAFFIC_SIGN_AHEAD`, cooldown dài (`cooldown_s=5`) để không spam.
- Nếu bật `sign_classifier` → message chứa tên biển.
- Cờ `adas.traffic.drive_on_right: true/false` điều khiển vùng quan tâm (VN lái bên phải).

## 7. Alert Engine (`alerts.py`)

```python
class AlertManager:
    def __init__(self, cfg, logger): ...
    def filter(self, alerts: list[Alert]) -> list[Alert]:
        """Áp cooldown theo (type, severity-level): cùng type không phát lại trong cooldown_s;
        severity tăng (WARNING→CRITICAL) được phát ngay, bỏ qua cooldown."""
```
- Cooldown mặc định: LDW 2 s, FCW 1 s, PEDESTRIAN 1.5 s, TRAFFIC_LIGHT 3 s, SIGN 5 s (cấu hình).
- Overlay hiển thị alert **còn hiệu lực** trong `display_s` (vd 1.5 s) kể cả khi bị cooldown chặn việc log.
- `EventLogger` ghi `events.jsonl`: 1 dòng JSON mỗi alert phát mới:
```json
{"frame": 1234, "t": 41.13, "type": "FORWARD_COLLISION", "severity": "CRITICAL",
 "message": "Nguy cơ va chạm phía trước", "data": {"track_id": 17, "cls": "car", "proximity": 0.91, "ttc_s": 1.2}}
```

## 8. `AdasEngine` (`engine.py`)

```python
class AdasEngine:
    def __init__(self, cfg): self.ldw = LaneDepartureWarning(cfg.ldw); self.fcw = ForwardCollision(cfg.fcw); ...
    def reset(self): ...
    def update(self, frame_idx, ts, shape_hw, detections, seg, lane) -> list[Alert]:
        ego = seg.masks[0] if seg is not None else None
        alerts = []
        if lane is not None: alerts += filter(None, [self.ldw.update(lane, frame_idx, ts)])
        alerts += self.fcw.update(...); alerts += self.ped.update(...); alerts += self.traffic.update(...)
        return self.manager.filter(alerts)
```
Thứ tự ưu tiên hiển thị: CRITICAL > WARNING > INFO; cùng mức: FCW > PEDESTRIAN > LDW > TRAFFIC_LIGHT > SIGN.

## 9. Test bằng dữ liệu giả lập
Xem `15_TESTING.md`: tạo mask làn hình thang tổng hợp (`make_lane_mask(offset)`), dịch ngang để mô phỏng lệch; chuỗi bbox với `h(t)` tăng để mô phỏng áp sát.
