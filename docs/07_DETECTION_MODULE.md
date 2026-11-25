# 07 — Detection module

## 1. Lựa chọn model

| Hạng mục | Mặc định | Ghi chú |
|---|---|---|
| Framework | Ultralytics (`pip install ultralytics`) | AGPL-3.0 |
| Model | `yolo11s.pt` (pretrained COCO) | Nhẹ → `yolo11n.pt`; chính xác hơn → `yolo11m.pt`. Nếu phiên bản ultralytics cũ: `yolov8s.pt` |
| imgsz | **960** | Ảnh 1280×720, biển báo/đèn nhỏ → 960 giữ chi tiết. 640 nhanh nhưng giảm AP vật nhỏ |
| Tracker | `bytetrack.yaml` | `persist=True` |

## 2. Train

```bash
yolo detect train data=data/processed/detection/data.yaml model=yolo11s.pt \
     imgsz=960 epochs=50 batch=16 device=0 workers=8 seed=42 \
     patience=15 cos_lr=True close_mosaic=10 amp=True \
     project=runs/detect name=yolo11s_bdd_960 exist_ok=True
```

Hoặc `python scripts/train_detector.py --config configs/default.yaml` (wrapper: đọc `detection.train.*` rồi gọi `YOLO(...).train(**kwargs)`).

Gợi ý hyperparameter:

| Tham số | Giá trị | Lý do |
|---|---|---|
| `epochs` | 50 (subset 10K) / 30 (full 70K) | Full dataset hội tụ nhanh hơn trên mỗi epoch |
| `batch` | 16 (T4, imgsz 960, FP16) | Hết VRAM → 8 hoặc `batch=-1` (auto) |
| `fliplr` | 0.5 | Giữ mặc định |
| `flipud` | 0.0 | Không lật dọc |
| `mosaic` | 1.0, `close_mosaic=10` | Tắt mosaic 10 epoch cuối |
| `hsv_h/s/v` | mặc định | Đủ đa dạng màu |
| `scale` | 0.5 | Mặc định |
| `workers` | Windows: 2–4; Colab: 2 | Windows hay lỗi/chậm với nhiều workers |
| `cache` | `ram` nếu đủ RAM, ngược lại `False` | Colab: copy dataset vào `/content` (SSD) trước |

Fine-tune hai giai đoạn (tuỳ chọn): epoch 1–5 `freeze=10` (đóng băng backbone), sau đó mở hết với lr thấp.

## 3. Mất cân bằng class & vật nhỏ
- BDD: `car` chiếm đa số; `rider`, `motorcycle`, `bicycle`, `bus` hiếm.
- Chiến lược:
  1. Báo cáo AP **từng class**, không chỉ mAP trung bình.
  2. Tăng `cls` loss weight nhẹ (`cls=0.7`) nếu AP class hiếm thấp.
  3. Oversample ảnh chứa class hiếm khi tạo subset (`make_subset.py --boost motorcycle,rider,bus`).
  4. Tăng imgsz lên 1280 (rect) nếu cần AP traffic_sign/traffic_light cao hơn.
- `traffic_light`, `traffic_sign` thường nhỏ (< 32 px): theo dõi `AP_small`.

## 4. `Detector` wrapper (`detection/detector.py`)

```python
class Detector:
    def __init__(self, weights: str, imgsz: int = 960, conf: float = 0.25,
                 iou: float = 0.6, device: str | int = 0, half: bool = True,
                 class_names: list[str] = DET_CLASSES, tracker: str = "bytetrack.yaml"):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        ...

    def predict(self, frame_bgr: np.ndarray) -> list[Detection]: ...
    def track(self, frame_bgr: np.ndarray) -> list[Detection]:
        r = self.model.track(frame_bgr, persist=True, tracker=self.tracker, imgsz=self.imgsz,
                             conf=self.conf, iou=self.iou, half=self.half, verbose=False)[0]
        ...
    def reset_tracker(self) -> None: ...   # gọi khi bắt đầu video mới
```

Chi tiết bắt buộc:
- Kết quả: `r.boxes.xyxy`, `r.boxes.cls`, `r.boxes.conf`, `r.boxes.id` (có thể `None` khi chưa có track → `track_id=None`).
- Map `cls_id` → `cls_name` bằng `class_names` của **config**, kiểm tra khớp `model.names` khi load (raise nếu khác để tránh lệch thứ tự class).
- Ngưỡng conf riêng theo class (tuỳ chọn): `detection.conf_per_class: {traffic_sign: 0.35, pedestrian: 0.2}` — lọc sau khi predict (đặt `conf` thấp nhất cho model).
- `reset_tracker()`: tạo lại predictor tracker khi đổi video (với Ultralytics: `self.model.predictor.trackers[0].reset()` nếu tồn tại; hoặc khởi tạo lại `YOLO(weights)`).

## 5. Màu đèn giao thông (`detection/traffic_light.py`)

Cách mặc định: heuristic HSV trên crop (nhẹ, không cần train thêm).

```python
def classify_light_color(crop_bgr: np.ndarray, min_ratio: float = 0.02) -> str:
    """Trả về 'red' | 'yellow' | 'green' | 'unknown'."""
    if crop_bgr.size == 0 or crop_bgr.shape[0] < 6 or crop_bgr.shape[1] < 3:
        return "unknown"
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bright = (s > 90) & (v > 150)
    red    = bright & ((h <= 10) | (h >= 165))
    yellow = bright & (h >= 15) & (h <= 35)
    green  = bright & (h >= 45) & (h <= 95)
    total = crop_bgr.shape[0] * crop_bgr.shape[1]
    ratios = {"red": red.sum() / total, "yellow": yellow.sum() / total, "green": green.sum() / total}
    best = max(ratios, key=ratios.get)
    return best if ratios[best] >= min_ratio else "unknown"
```

Cải tiến (làm sau): thu nhỏ vùng crop vào phần sáng nhất; chia đôi theo chiều dọc/ngang (đèn dọc: đỏ trên, xanh dưới) để tránh nhầm; làm mượt màu theo `track_id` (voting 5 frame gần nhất).

Phương án thay thế: tách 3 class `traffic_light_red/green/yellow` lúc convert bằng thuộc tính `trafficLightColor` của BDD (`detection.split_traffic_light_color: true`). Khó hơn vì phải phân biệt màu ở vật rất nhỏ; so sánh với heuristic bằng thí nghiệm.

## 6. Biển báo (traffic sign)
- BDD100K chỉ có **1 class** `traffic sign`. Detector cho biết "có biển báo + vị trí + kích thước", đủ cho cảnh báo `TRAFFIC_SIGN_AHEAD`.
- **Mở rộng** (phân loại loại biển: tốc độ, dừng, cấm...): pipeline 2 giai đoạn — crop box `traffic_sign` → classifier nhỏ (MobileNetV3 / ResNet18) train trên GTSRB hoặc dataset biển báo Việt Nam. Đặt ở `src/adas/detection/sign_classifier.py`, tuỳ chọn bật bằng `detection.sign_classifier.enabled`. **Không bắt buộc cho phiên bản 1.**

## 7. Đánh giá
```bash
yolo detect val model=runs/detect/yolo11s_bdd_960/weights/best.pt \
    data=data/processed/detection/data.yaml imgsz=960 split=val
```
- Báo cáo: mAP50, mAP50-95, P, R **từng class**, confusion matrix, đường PR.
- Cross-dataset: `data=data/processed/cityscapes_det/data.yaml`.
- Phân tích theo điều kiện: tách val theo `timeofday`/`weather` bằng file list → val riêng (`scripts/eval_detector.py --by timeofday`).
- Lỗi thường gặp cần nhìn: FP biển quảng cáo/đèn xe, FN xe bị che, xe máy bị nhận là bicycle/pedestrian.

## 8. Export (tuỳ chọn)
```bash
yolo export model=weights/detector_best.pt format=onnx imgsz=960 half=False
```
(`scripts/export_onnx.py` bọc lệnh trên; ghi chú rằng tracker vẫn chạy ở Python.)
