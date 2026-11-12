# 02 — Kiến trúc Chi tiết, Data Contracts & Cấu hình Hệ thống

## 1. Nguyên tắc phân tách lớp (Separation of Concerns)

Hệ thống được chia làm 4 tầng độc lập để dễ phát triển song song và kiểm thử đơn vị:

| Tầng (Layer) | Thư mục | Thư viện cho phép | Trách nhiệm |
| :--- | :--- | :--- | :--- |
| **Data Layer** | `src/data/` | `numpy`, `cv2`, `albumentations`, `torch` | Đọc ảnh/nhãn BDD100K & Cityscapes, tạo Dataset/DataLoader, chạy augmentation & preprocess chuẩn hóa |
| **Model Layer** | `src/models/` | `torch`, `ultralytics`, `segmentation-models-pytorch` | Nạp weights, kiểm tra metadata, chạy forward inference và trả về cấu trúc chuẩn |
| **ADAS Logic Layer** | `src/adas/` | **Chỉ `numpy` + Python stdlib** | Tính toán hình học LDW, FCW, PCW, quản lý cảnh báo. **Cấm import `torch`, `ultralytics`, `cv2`** |
| **Presentation & App** | `src/visualization/`, `app/` | `cv2`, `streamlit`, `pandas` | Vẽ HUD overlay trên bản sao frame, xuất video, ghi log CSV, giao diện web |

---

## 2. Quy chuẩn Hằng số & Data Contracts (`src/utils/constants.py`)

Mọi module phải import hằng số từ `src/utils/constants.py`, tuyệt đối không hard-code chuỗi tên lớp hay chỉ số kênh rời rạc.

### 2.1. Thứ tự lớp Object Detection (`DET_CLASSES`)
```python
DET_CLASSES: tuple[str, ...] = (
    "car",            # 0
    "truck",          # 1
    "bus",            # 2
    "motorcycle",     # 3
    "bicycle",        # 4
    "pedestrian",     # 5
    "rider",          # 6
    "traffic_light",  # 7
    "traffic_sign",   # 8
)
```
* Lưu ý: Khi nạp model YOLO, bắt buộc kiểm tra `model.names` khớp chính xác với `DET_CLASSES`; nếu sai thứ tự hoặc thiếu lớp phải `raise ValueError`.

### 2.2. Thứ tự kênh Segmentation (`SEG_CHANNELS`)
```python
SEG_CHANNELS: tuple[str, ...] = (
    "drivable_direct",       # Kênh 0: Làn xe đang đi (ego lane)
    "drivable_alternative",  # Kênh 1: Làn xe lân cận
    "lane_marking",          # Kênh 2: Vạch kẻ đường
)
```

### 2.3. Quy chuẩn Dữ liệu Trao đổi Giữa các Module
* **Ảnh nội bộ (Frame):** `np.ndarray` kiểu `uint8`, hệ màu **BGR**, shape `(H, W, 3)`. Việc chuyển sang RGB và normalize `(mean, std)` chỉ thực hiện bên trong hàm preprocess của model.
* **Bounding Box:** Tọa độ pixel tuyệt đối theo ảnh gốc `[x1, y1, x2, y2]` (`xyxy`), kiểu `float32`.
* **Segmentation Mask (Binary/Prob):** `np.ndarray` shape `(3, H, W)` theo kích thước ảnh gốc (hoặc kích thước chuẩn quy định trong config) với xác suất `float32` $\in [0, 1]$ hoặc mask nhị phân `uint8` $\{0, 1\}$.
* **Bất biến đầu vào (Immutability):** Các hàm trong `src/adas/` và `src/visualization/` không được sửa trực tiếp (in-place) mảng `frame` hay `mask` đầu vào; phải `.copy()` trước khi vẽ overlay.
* **Kích thước OpenCV vs NumPy:** Luôn nhớ `cv2.resize(img, (W, H))` nhận `(Width, Height)` trong khi shape NumPy là `(Height, Width)`.

---

## 3. Cấu trúc Dữ liệu Nội bộ (Dataclasses gợi ý)

Đặt tại `src/utils/types.py` (thuần Python/NumPy):

```python
@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]  # (x1, y1, x2, y2) theo pixel frame gốc
    confidence: float
    class_id: int
    class_name: str

@dataclass(frozen=True)
class ADASAlert:
    alert_type: str        # "LDW" | "FCW" | "PCW" | "TSR"
    severity: str          # "SAFE" | "WARNING" | "CRITICAL"
    message: str
    target_bbox: tuple[float, float, float, float] | None = None

@dataclass
class FrameAnalysis:
    frame_idx: int
    timestamp_sec: float
    detections: list[Detection]
    ldw_status: str        # "NORMAL" | "WARNING" | "DEPARTED"
    fcw_status: str        # "SAFE" | "CAUTION" | "CRITICAL"
    alerts: list[ADASAlert]
```

---

## 4. Thiết kế Cấu hình YAML (`configs/`)

Mọi ngưỡng thuật toán, siêu tham số và đường dẫn phải khai báo trong YAML. **Cấm magic number trong code logic.**

### 4.1. `configs/paths.yaml` (Đường dẫn dữ liệu & trọng số)
```yaml
project:
  seed: 42
  device: "cuda"

paths:
  bdd_root: "D:/datasets/bdd100k"
  cityscapes_root: "D:/datasets/cityscapes"
  detector_weights: "weights/detector_best.pt"
  segmenter_weights: "weights/segmenter_best.pt"
  output_dir: "outputs"
```

### 4.2. `configs/detector_yolo11.yaml`
```yaml
model:
  architecture: "yolo11s.pt"
  img_size: 640
  conf_threshold: 0.35
  iou_threshold: 0.45

train:
  epochs: 50
  batch_size: 16
  optimizer: "AdamW"
  lr0: 0.001
  workers: 4
```

### 4.3. `configs/segmenter_deeplab.yaml`
```yaml
model:
  architecture: "DeepLabV3Plus"
  encoder: "resnet34"
  encoder_weights: "imagenet"
  input_size_hw: [384, 640]   # (Height, Width)
  num_classes: 3
  thresholds: [0.5, 0.5, 0.4] # Ngưỡng riêng cho từng kênh [direct, alt, lane]

train:
  epochs: 30
  batch_size: 8
  lr: 0.0003
  loss: "bce_dice"
```

### 4.4. `configs/adas_rules.yaml`
```yaml
adas:
  ldw:
    camera_offset_ratio: 0.0      # Bù lệch tâm camera so với tâm xe (-0.1 đến 0.1)
    ego_y_ratio: 0.88             # Vị trí trục ngang kiểm tra bánh xe (tính từ trên xuống)
    wheel_half_width_ratio: 0.12  # Nửa bề rộng vệt bánh xe ước lượng theo bề rộng ảnh
    lane_overlap_threshold: 0.15  # Tỷ lệ điểm ảnh vạch kẻ đè vào vùng bánh xe
    hysteresis_frames: 3          # Số frame liên tiếp để xác nhận chuyển trạng thái

  fcw:
    corridor_top_width_ratio: 0.18
    corridor_bottom_width_ratio: 0.45
    corridor_top_y_ratio: 0.45
    caution_area_ratio: 0.04      # Diện tích bbox / diện tích frame
    critical_area_ratio: 0.10
    growth_rate_critical: 0.25    # Tốc độ tăng diện tích bbox giữa các frame
    history_window: 5

  pcw:
    target_classes: ["pedestrian", "rider", "bicycle"]
    danger_y_min_ratio: 0.50      # Chỉ cảnh báo khẩn khi đối tượng ở nửa dưới khung hình
    in_lane_iou_threshold: 0.10
```
