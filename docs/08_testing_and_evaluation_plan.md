# 08 — Kế hoạch Kiểm thử (Testing) & Đánh giá (Evaluation)

## 1. Chiến lược Kiểm thử Phần mềm (`tests/` với `pytest`)

> [!IMPORTANT]
> Unit tests không được phụ thuộc vào dataset BDD100K thật hay file trọng số `.pt` nặng. Toàn bộ test phải chạy nhanh (< 10 giây) bằng dữ liệu tổng hợp (synthetic arrays/images) tạo trong bộ nhớ.

### Danh sách các nhóm Test bắt buộc:
1. **`tests/test_constants_and_configs.py`:**
   * Kiểm tra file YAML trong `configs/` load hợp lệ, đầy đủ khóa bắt buộc.
   * Kiểm tra `DET_CLASSES` (9 lớp) và `SEG_CHANNELS` (3 kênh) khớp cấu hình.
2. **`tests/test_data_conversion.py`:**
   * Test hàm chuyển đổi bounding box BDD100K (`xyxy` pixel) sang YOLO (`cx, cy, w, h` normalized) và ngược lại; xử lý trường hợp biên ngoài khung ảnh.
   * Test hàm chuyển mask thô thành 3 kênh nhị phân `(3, H, W)` với mảng synthetic nhỏ (ví dụ `64x64`).
3. **`tests/test_adas_ldw.py`:**
   * Mask `drivable_direct` phủ giữa ảnh, không có vạch đè $\rightarrow$ trả về `NORMAL`.
   * Vạch `lane_marking` cắt vào vệt bánh xe trái/phải $\rightarrow$ trả về `WARNING`.
   * Tâm xe nằm hẳn ngoài `drivable_direct` và sang `drivable_alternative` $\rightarrow$ trả về `DEPARTED`.
   * Test cơ chế `hysteresis_frames` (1 frame nhiễu đơn lẻ không làm nhảy cảnh báo).
4. **`tests/test_adas_fcw_pcw.py`:**
   * Xe `car` ở ngoài Ego Corridor (làn bên cạnh) $\rightarrow$ không kích hoạt `FCW`.
   * Xe `car` trong Ego Corridor có diện tích bbox tăng nhanh qua 3 frame liên tiếp $\rightarrow$ kích hoạt `CRITICAL`.
   * Đối tượng `pedestrian` xuất hiện trong `drivable_direct` ở nửa dưới khung hình $\rightarrow$ kích hoạt `PCW`.
5. **`tests/test_visualizer.py`:**
   * Đảm bảo `draw_overlay()` trả về ảnh cùng shape, `uint8`, và **không sửa đổi (mutate) mảng `frame` đầu vào**.

### Lệnh kiểm tra chất lượng code trước khi merge:
```powershell
ruff check .
pytest -v
```

---

## 2. Đánh giá Định lượng Mô hình (`scripts/evaluate.py`)

> [!NOTE]
> Chỉ điền số liệu thật vào bảng kết quả sau khi chạy đánh giá trên checkpoint đã huấn luyện; không điền số ước lượng hay giả định.

### 2.1. Đánh giá Object Detection (BDD100K Val)
* Metrics thu thập: `mAP@0.5`, `mAP@0.5:0.95`, `Precision`, `Recall` tổng thể và chi tiết từng lớp trong `DET_CLASSES`.
* Bảng so sánh Benchmark (YOLO11n vs YOLO11s vs Baseline YOLO khác):

| Mô hình | Img Size | Params (M) | mAP@0.5 (All) | mAP@0.5:0.95 | AP (`car`) | AP (`traffic_sign`) | Latency (ms/frame - T4) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline (YOLOv8s / khác) | 640 | - | - | - | - | - | - |
| YOLO11n | 640 | - | - | - | - | - | - |
| YOLO11s | 640 | - | - | - | - | - | - |

### 2.2. Đánh giá Semantic Segmentation (BDD100K Val & Cityscapes Val)
* Metrics thu thập: IoU và Dice Score theo từng kênh độc lập:

| Mô hình / Backbone | Tập dữ liệu | IoU `drivable_direct` | IoU `drivable_alternative` | IoU `lane_marking` | Mean IoU | Latency (ms - T4) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| DeepLabV3+ (`resnet34`) | BDD100K Val | - | - | - | - | - |
| DeepLabV3+ (`mobilenet_v2`) | BDD100K Val | - | - | - | - | - |
| DeepLabV3+ (`resnet34`) | Cityscapes Val (Road union) | *(N/A)* | *(N/A)* | *(N/A)* | - | - |

### 2.3. Benchmark Hệ thống End-to-End
Đo trên GPU T4 với video chuẩn 720p (`1280x720`):
* Thời gian Preprocess trung bình (ms)
* Thời gian Detection (ms)
* Thời gian Segmentation (ms)
* Thời gian ADAS Logic + Visualization (ms)
* **Tổng FPS End-to-End** (Mục tiêu $\ge 15\text{ FPS}$).

---

## 3. Đánh giá Định tính & Phân tích Lỗi (Qualitative Error Analysis)

Chạy pipeline trên tối thiểu **3–5 video dashcam thực tế** đại diện cho các kịch bản:
1. **Ban ngày, đường quang đãng (Baseline):** Kiểm tra độ ổn định của LDW khi chạy thẳng (đếm số lần báo giả / phút).
2. **Ban đêm / Ánh sáng yếu / Đèn pha ngược chiều:** Quan sát suy giảm khả năng bắt vạch kẻ đường và xe máy.
3. **Trời mưa / Mặt đường phản chiếu ánh sáng:** Kiểm tra nhiễu false positive trên `drivable_direct` và `lane_marking`.
4. **Đường Việt Nam (Domain Gap Check):** Đánh giá khả năng nhận diện mật độ xe máy dày đặc so với đường Mỹ trong BDD100K.

Với mỗi trường hợp lỗi nghiêm trọng, lưu ảnh snapshot kèm phân tích nguyên nhân gốc (dữ liệu, ngưỡng threshold, hay giới hạn góc camera).
