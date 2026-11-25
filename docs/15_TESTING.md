# 15 — Chiến lược kiểm thử

Công cụ: `pytest`. Chạy: `pytest -q`. Test không cần GPU, không cần dataset thật (dùng dữ liệu tổng hợp nhỏ).

## 1. Nguyên tắc
- Test **ADAS logic** bằng input giả lập (mask hình thang, chuỗi bbox).
- Test **data conversion** bằng frame JSON giả.
- Test **model wrapper** bằng mock (không tải weights).
- Mỗi bug sửa xong → thêm test tái hiện.

## 2. Fixtures dùng chung (`tests/conftest.py`)

```python
import numpy as np, pytest

def make_lane_mask(H=720, W=1280, center_x=640, top_w=200, bottom_w=700, y_top=0.55):
    """Hình thang mô phỏng làn ego: hẹp ở xa, rộng ở gần. Trả bool (H, W)."""
    m = np.zeros((H, W), bool)
    y0 = int(H * y_top)
    for y in range(y0, H):
        t = (y - y0) / (H - 1 - y0)
        half = (top_w + t * (bottom_w - top_w)) / 2
        l, r = int(max(0, center_x - half)), int(min(W, center_x + half))
        m[y, l:r] = True
    return m

@pytest.fixture
def lane_center(): return make_lane_mask()
@pytest.fixture
def lane_shifted_right(): return make_lane_mask(center_x=640 - 160)   # làn dịch trái → xe lệch phải
```

Helper `make_det(cls_name, xyxy, conf=0.9, track_id=1)` tạo `Detection`.

## 3. Danh sách test

### Constants / config
| Test | Kiểm tra |
|---|---|
| `test_constants` | 9 class đúng thứ tự; 3 kênh seg đúng thứ tự; `BDD_DET_MAP` không có `train` |
| `test_config_missing_key` | Thiếu key → ValueError rõ ràng |

### Data
| Test | Kiểm tra |
|---|---|
| `test_bdd_to_yolo_basic` | Box hợp lệ → dòng YOLO đúng (cx, cy, w, h chuẩn hoá) |
| `test_bdd_clip_box` | Box vượt biên bị clip, giá trị ∈ [0,1] |
| `test_bdd_skip_small` | Box < `min_box_px` bị bỏ |
| `test_bdd_skip_train_class` | Category `train` bị bỏ |
| `test_bdd_no_box2d` | Label không có `box2d` (vd lane) bị bỏ |
| `test_bdd_empty_frame` | Frame không có label → danh sách rỗng |
| `test_merge_rider` | Cờ merge → rider thành pedestrian |
| `test_mask_roundtrip` | `unpack(pack(a,b,c)) == (a,b,c)` |
| `test_resize_keeps_thin_lane` | Vạch dày 2 px (1280×720) vẫn còn pixel sau resize (640×384) |
| `test_cityscapes_polygon_bbox` | Polygon → bbox + map class; `cargroup` bị bỏ |

### Traffic light colour
| Test | Kiểm tra |
|---|---|
| `test_light_red/green/yellow` | Crop tổng hợp màu bão hoà → đúng nhãn |
| `test_light_dark_unknown` | Crop tối/xám → `unknown` |
| `test_light_tiny_crop` | Crop quá nhỏ → `unknown` (không crash) |

### Lane geometry
| Test | Kiểm tra |
|---|---|
| `test_lane_center_offset_zero` | Làn giữa ảnh → `|offset| < 0.03` |
| `test_lane_offset_sign` | Làn dịch trái → `offset > 0` (xe lệch phải) và ngược lại |
| `test_lane_invalid_empty_mask` | Mask rỗng → `valid=False` |
| `test_lane_too_narrow` | Làn quá hẹp → `valid=False` |
| `test_best_run_picks_center` | Hai run: chọn run chứa tâm xe |

### LDW
| Test | Kiểm tra |
|---|---|
| `test_ldw_no_alert_center` | 30 frame đi giữa → không alert |
| `test_ldw_alert_right` | Offset +0.3 liên tục ≥ `min_frames` → WARNING `LANE_DEPARTURE_RIGHT` |
| `test_ldw_debounce` | Chỉ 2 frame vượt ngưỡng → không alert |
| `test_ldw_hysteresis` | Offset dao động quanh ngưỡng không gây nhấp nháy |
| `test_ldw_critical` | Offset ≥ `critical_offset` → CRITICAL |
| `test_ldw_invalid_grace` | Mất làn ≤ grace → giữ trạng thái; > grace → reset |

### FCW
| Test | Kiểm tra |
|---|---|
| `test_fcw_outside_lane` | Xe ngoài `ego_region` → không alert |
| `test_fcw_proximity` | Chân bbox y/H ≥ crit → CRITICAL |
| `test_fcw_ttc_growing_bbox` | `h(t)` tăng nhanh → TTC nhỏ → CRITICAL; ghi `ttc_s` đúng xấp xỉ giải tích |
| `test_fcw_static_bbox_inf_ttc` | bbox không đổi → TTC = inf, không alert (nếu xa) |
| `test_fcw_picks_nearest` | Nhiều xe: chỉ 1 alert cho xe gần nhất |
| `test_fcw_min_track_age` | Track mới < `min_track_age` → bỏ qua |
| `test_fcw_track_ttl_cleanup` | Track biến mất quá `ttl` bị xoá lịch sử |

### Pedestrian / traffic
| Test | Kiểm tra |
|---|---|
| `test_ped_in_path` | Chân người trong `ego_region`, `proximity ≥ ngưỡng` → alert |
| `test_ped_on_sidewalk` | Ngoài vùng → không alert |
| `test_red_light_center` | Đèn đỏ ở giữa, đủ lớn → `TRAFFIC_LIGHT_RED` |
| `test_red_light_side_ignored` | Đèn đỏ rìa ảnh → bỏ qua |
| `test_sign_ahead_cooldown` | Biển liên tiếp → chỉ 1 INFO trong cooldown |

### Alert manager
| Test | Kiểm tra |
|---|---|
| `test_cooldown` | Cùng type trong cooldown bị lọc |
| `test_severity_escalation_bypasses_cooldown` | WARNING → CRITICAL phát ngay |
| `test_event_logger_jsonl` | Ghi đúng định dạng JSONL, parse lại được |

### Pipeline (smoke, mock)
| Test | Kiểm tra |
|---|---|
| `test_pipeline_smoke_image` | Mock Detector/Segmenter, ảnh 720p → `FrameResult` đủ trường |
| `test_pipeline_video_tmp` | Tạo video ngẫu nhiên 20 frame bằng cv2 → chạy → có `events.jsonl`, `summary.json` |
| `test_renderer_no_mutate` | `draw()` không đổi frame đầu vào |

## 4. Test thủ công (checklist trước khi demo)
- [ ] 3 clip ngày/đêm/mưa chạy không lỗi.
- [ ] Mask làn bám đường, không nhấp nháy mạnh.
- [ ] Đi thẳng: LDW không báo giả quá 1 lần/phút.
- [ ] Xe phía trước phanh/áp sát: FCW kích hoạt hợp lý.
- [ ] Người băng đường trước xe: cảnh báo.
- [ ] FPS hiển thị khớp `summary.json`.

## 5. CI (tuỳ chọn)
GitHub Actions: Python 3.11, `pip install -r requirements.txt` (CPU torch), `ruff check`, `pytest -q`. Bỏ qua test cần GPU bằng marker `@pytest.mark.gpu`.
