# 04 — Kiến trúc hệ thống

## 1. Tổng quan luồng xử lý

```mermaid
flowchart LR
  A[Video / Image / Webcam] --> B[FrameSource]
  B --> C1[Detector YOLO + ByteTrack]
  B --> C2[Segmenter DeepLabV3+]
  C1 --> D[Detections + track_id]
  C2 --> E[SegResult masks 3 kênh]
  D --> F[ADAS Engine]
  E --> F
  F --> G[Alerts]
  D --> H[Renderer overlay]
  E --> H
  G --> H
  G --> I[EventLogger JSONL]
  H --> J[VideoWriter / Streamlit]
```

Nguyên tắc: **mỗi khối có input/output là dataclass thuần**, không phụ thuộc chéo. ADAS Engine **không import torch/ultralytics**.

## 2. Data contracts (đặt trong `src/adas/types.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class AlertType(str, Enum):
    LANE_DEPARTURE_LEFT = "LANE_DEPARTURE_LEFT"
    LANE_DEPARTURE_RIGHT = "LANE_DEPARTURE_RIGHT"
    FORWARD_COLLISION = "FORWARD_COLLISION"
    PEDESTRIAN_IN_PATH = "PEDESTRIAN_IN_PATH"
    TRAFFIC_LIGHT_RED = "TRAFFIC_LIGHT_RED"
    TRAFFIC_SIGN_AHEAD = "TRAFFIC_SIGN_AHEAD"

@dataclass(frozen=True)
class Detection:
    xyxy: tuple[float, float, float, float]   # pixel, frame gốc
    cls_id: int
    cls_name: str
    conf: float
    track_id: int | None = None
    attrs: dict = field(default_factory=dict)  # vd {"light_color": "red"}

    @property
    def w(self) -> float: return self.xyxy[2] - self.xyxy[0]
    @property
    def h(self) -> float: return self.xyxy[3] - self.xyxy[1]
    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.xyxy[0] + self.xyxy[2]) / 2.0, self.xyxy[3])

@dataclass
class SegResult:
    # bool mask, shape (3, H, W) ở kích thước frame gốc
    # thứ tự kênh: 0 drivable_direct, 1 drivable_alternative, 2 lane_marking
    masks: np.ndarray
    probs: np.ndarray | None = None  # (3, H, W) float16, tuỳ chọn

@dataclass
class LaneGeometry:
    valid: bool
    left_x: float | None = None        # x biên trái làn ego tại dải ROI (pixel)
    right_x: float | None = None
    lane_width_px: float | None = None
    center_x: float | None = None
    vehicle_x: float | None = None
    offset_ratio: float | None = None  # (vehicle_x - center_x) / lane_width; >0: xe lệch phải

@dataclass
class Alert:
    type: AlertType
    severity: Severity
    message: str
    frame_idx: int
    timestamp_s: float
    data: dict = field(default_factory=dict)

@dataclass
class FrameResult:
    frame_idx: int
    timestamp_s: float
    detections: list[Detection]
    seg: SegResult | None
    lane: LaneGeometry | None
    alerts: list[Alert]
    timings_ms: dict[str, float] = field(default_factory=dict)
```

## 3. Module & trách nhiệm

| Module | File | Trách nhiệm | Phụ thuộc |
|---|---|---|---|
| Config | `config.py` | Load YAML → dataclass/dict có validate, merge override CLI | pyyaml |
| Types | `types.py` | Dataclass ở trên | numpy |
| Detector | `detection/detector.py` | Bọc Ultralytics: `predict`, `track`; chuẩn hoá ra `Detection` | ultralytics |
| Traffic light colour | `detection/traffic_light.py` | HSV → red/yellow/green/unknown | opencv |
| Segmenter | `segmentation/segmenter.py` | Load ckpt, preprocess, forward, upsample → `SegResult` | torch, smp |
| Dataset | `segmentation/dataset.py` | `BDDSegDataset` + augment | torch, albumentations |
| Model factory | `segmentation/model.py` | `build_model(cfg)` | smp |
| Loss/metrics | `segmentation/losses.py`, `metrics.py` | BCE+Dice, IoU tích luỹ | torch |
| Lane geometry | `adas/lane_geometry.py` | mask → `LaneGeometry` | numpy |
| LDW | `adas/ldw.py` | state machine lệch làn | numpy |
| FCW | `adas/fcw.py` | ego-lane check + TTC proxy | numpy |
| Pedestrian | `adas/pedestrian.py` | VRU trong vùng drivable | numpy |
| Traffic warnings | `adas/traffic.py` | đèn đỏ, biển báo | numpy |
| Alert engine | `adas/alerts.py` | debounce, cooldown, log | stdlib |
| Engine | `adas/engine.py` | Gom tất cả, `update(frame_result_inputs) -> alerts` | — |
| Pipeline | `pipeline.py` | Nối Source → Detector/Segmenter → Engine → Renderer | all |
| Renderer | `render.py` | Vẽ overlay | opencv |
| IO | `io.py` | FrameSource, VideoWriter, EventLogger | opencv |
| CLI | `cli.py` | `run`, `eval`, `convert` | argparse/typer |

## 4. Luồng một frame

1. `FrameSource` đọc frame BGR `(H, W, 3)` + `timestamp_s = frame_idx / fps`.
2. `Detector.track(frame)` → `list[Detection]` (có `track_id`).
3. `traffic_light.classify_colors(frame, detections)` gán `attrs["light_color"]`.
4. `Segmenter.predict(frame)` → `SegResult` (bool masks kích thước frame).
5. `lane_geometry.estimate(seg, cfg)` → `LaneGeometry`.
6. `AdasEngine.update(frame_idx, ts, detections, seg, lane)` → `list[Alert]`.
7. `Renderer.draw(...)` → frame overlay; `VideoWriter.write`.
8. `EventLogger.log(alerts)`.
9. Ghi thời gian từng bước vào `timings_ms`.

## 5. Quyết định thiết kế & lý do

| Quyết định | Lý do | Phương án thay thế |
|---|---|---|
| 2 model rời | Dễ train/debug; thay YOLO/seg độc lập; GPU T4 đủ | YOLOP / HybridNets (multi-task, khó tuỳ biến) |
| Seg multi-label sigmoid | Vạch kẻ nằm trên vùng drivable → pixel thuộc nhiều lớp | Softmax 4 lớp (xung đột nhãn) |
| ADAS thuần NumPy | Test nhanh bằng input giả lập | — |
| ByteTrack qua Ultralytics | Có sẵn, tích hợp, đủ tốt cho demo | DeepSORT, BoT-SORT |
| Config YAML tập trung | Chỉnh ngưỡng không sửa code | Hydra (thừa cho scale này) |

## 6. Hiệu năng
- Detector & Segmenter chạy tuần tự trên cùng GPU, FP16 (`half: true`).
- Segmenter input `384×640` (mặc định); upsample logits bằng `torch.nn.functional.interpolate` (bilinear) về `(H, W)` trên GPU rồi `> thr` → chuyển CPU 1 lần.
- Tuỳ chọn `pipeline.seg_every_n_frames: 1` (chạy seg mỗi N frame, dùng lại mask frame trước) để tăng FPS.
- Đo thời gian từng khối bằng `time.perf_counter()` (sau `torch.cuda.synchronize()` khi đo GPU).
