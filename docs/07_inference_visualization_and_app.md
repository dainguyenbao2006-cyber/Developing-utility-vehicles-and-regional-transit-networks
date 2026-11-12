# 07 — Inference Pipeline, Visualization HUD & Ứng dụng Demo

## 1. Luồng Xử lý End-to-End Pipeline (`src/pipeline.py`)

Pipeline kết nối tuần tự các module trên mỗi khung hình video hoặc webcam:

```
Frame (BGR uint8, HxWx3)
  ├──► Detector.predict(frame)       ──► list[Detection] (xyxy gốc)
  ├──► Segmenter.predict(frame)      ──► seg_mask (3, H, W)
  ▼
ADASAnalyzer.update(detections, seg_mask, frame_shape)
  │
  ▼  FrameAnalysis (ldw_status, fcw_status, alerts)
Visualizer.draw_overlay(frame.copy(), detections, seg_mask, analysis)
  │
  ▼
Output Frame (VideoWriter / Streamlit Canvas) + EventLogger.log(analysis)
```

* **Nguyên tắc bộ nhớ:**
  * Đọc video theo từng frame bằng `cv2.VideoCapture` (không load toàn bộ video vào RAM).
  * Không sửa đổi `frame` đầu vào tại chỗ (`frame = frame.copy()` trước khi vẽ).

---

## 2. Thiết kế Visualization HUD (`src/visualization/hud.py`)

Giao diện hiển thị trên video (Head-Up Display) cần trực quan, dễ quan sát khi lái xe nhưng không che khuất tầm nhìn:

1. **Lớp phủ Segmentation bán trong suốt (`alpha = 0.35`):**
   * `drivable_direct`: Tô màu **Xanh lá** nhạt (`(0, 200, 0)` BGR).
   * `drivable_alternative`: Tô màu **Xanh dương** nhạt (`(200, 120, 0)` BGR).
   * `lane_marking`: Tô màu **Vàng/Đỏ cam** nổi bật (`(0, 220, 255)` BGR) để dễ kiểm tra độ khớp vạch kẻ.
2. **Bounding Boxes & Labels:**
   * Vẽ box theo màu phân nhóm:
     * Xe cơ giới (`car`, `truck`, `bus`, `motorcycle`): Xanh dương / Trắng (chuyển sang **Đỏ** nếu là đối tượng gây cảnh báo FCW).
     * Đối tượng yếu thế (`pedestrian`, `rider`, `bicycle`): Cam / Tím (chuyển **Đỏ nhấp nháy** nếu vi phạm PCW).
     * Tín hiệu (`traffic_light`, `traffic_sign`): Xanh ngọc (`Cyan`).
3. **Bảng trạng thái ADAS (Góc trên trái/phải):**
   * Hiển thị FPS thực tế (`FPS: 18.4`).
   * Đèn trạng thái **LDW** (`NORMAL` / `WARNING` / `DEPARTED`).
   * Đèn trạng thái **FCW** (`SAFE` / `CAUTION` / `CRITICAL`).
   * Banner cảnh báo lớn ở mép trên khi có trạng thái `CRITICAL` (ví dụ: `"⚠ CANH BAO VA CHAM PHIA TRUOC!"`, `"⚠ LECH LAN DUONG!"`).
   * Vẽ vạch Ego Corridor mờ để người xem hiểu vùng ADAS đang quét.

---

## 3. Thiết kế Giao diện Dòng lệnh (CLI)

Điểm chạy chuẩn qua module: `python -m adas.cli run ...` (hoặc `scripts/run_inference.py`).

### Các tham số CLI hỗ trợ:
* `--source`: Đường dẫn file video `.mp4`/`.avi`, thư mục ảnh, hoặc `0` cho webcam.
* `--output`: Đường dẫn lưu video kết quả (mặc định `outputs/demo_out.mp4`).
* `--log-csv`: Đường dẫn lưu log sự kiện cảnh báo.
* `--config-paths`: File `configs/paths.yaml`.
* `--config-adas`: File `configs/adas_rules.yaml`.
* `--no-display`: Chạy headless không mở cửa sổ `cv2.imshow` (hữu ích trên Colab).
* `--max-frames`: Giới hạn số frame để test nhanh.

---

## 4. Thiết kế Web Demo Streamlit (`app/streamlit_app.py`)

Giao diện web phục vụ trình diễn và tinh chỉnh tham số trực tiếp:

### Bố cục giao diện:
* **Thanh bên (Sidebar — Cấu hình trực tiếp):**
  * Chọn nguồn đầu vào: Upload video dashcam hoặc chọn video mẫu có sẵn.
  * Tùy chọn bật/tắt từng lớp hiển thị: Detection Boxes, Segmentation Mask, Ego Corridor, HUD Banner.
  * Thanh trượt (Sliders) tinh chỉnh trực tiếp các ngưỡng:
    * Detector Confidence Threshold (`0.1` – `0.9`).
    * LDW `camera_offset_ratio` (`-0.15` – `0.15`).
    * FCW `caution_area_ratio` & `critical_area_ratio`.
* **Khu vực chính (Main Area):**
  * **Tab 1 — Video Inference:** Trình phát video trực tiếp (hoặc xử lý xuất video H.264/MP4 tương thích trình duyệt) kèm biểu đồ FPS theo thời gian thực.
  * **Tab 2 — Thống kê & Log Sự kiện:**
    * Bảng đếm số lượng đối tượng phát hiện được.
    * Biểu đồ timeline các sự kiện `LDW`, `FCW`, `PCW`.
    * Bảng chi tiết `Event Log` và nút **Download CSV Log** / **Download Output Video**.
