# 04 — Thiết kế & Huấn luyện Object Detection (YOLO11)

## 1. Mục tiêu Module
Xây dựng bộ phát hiện đối tượng giao thông theo thời gian thực (`src/models/detector.py`) đạt độ chính xác cao cho phương tiện lớn (`car`, `truck`, `bus`), đồng thời cải thiện khả năng phát hiện đối tượng nhỏ (`traffic_sign`, `traffic_light`, `pedestrian`, `motorcycle`).

---

## 2. Kiến trúc & Lựa chọn Mô hình

### 2.1. Mô hình chính: YOLO11
* **Lý do chọn:** YOLO11 (Ultralytics) cải tiến cấu trúc với block **C3k2** (giảm chi phí tính toán nhưng giữ trường tiếp nhận hiệu quả) và **C2PSA** (tăng cường cơ chế attention không gian cho các vật thể nhỏ ở xa như biển báo giao thông và người đi bộ).
* **Biến thể sử dụng:**
  * `yolo11n.pt` (Nano): Dành cho kịch bản ưu tiên FPS cao nhất hoặc thử nghiệm trên CPU/GPU yếu.
  * `yolo11s.pt` (Small — **Khuyến nghị chính**): Cân bằng tốt nhất giữa mAP và tốc độ ($\ge 30+$ FPS riêng phần detection trên T4).
  * `yolo11m.pt` (Medium): Dùng làm đối chứng độ chính xác nếu còn dư ngân sách độ trễ.

### 2.2. Kịch bản So sánh Mô hình (Benchmark Suite)
Theo định hướng nghiên cứu dự án (so sánh YOLO11 với các baseline / phiên bản YOLO khác):
* Xây dựng script chuẩn hóa đầu vào/đầu ra cùng trên tập **BDD100K val**.
* Các chỉ số ghi nhận trong bảng đối sánh:
  * **Độ chính xác:** `mAP@0.5`, `mAP@0.5:0.95`, `AP_car`, `AP_motorcycle`, `AP_pedestrian`, `AP_traffic_sign`.
  * **Hiệu năng phần cứng:** Số lượng tham số (`Params M`), `GFLOPs`, Latency trung bình (`ms/frame`) trên T4 với `imgsz=640`, `batch=1`.

---

## 3. Thiết kế Wrapper (`src/models/detector.py`)

### Yêu cầu kỹ thuật:
1. **Khởi tạo & Xác thực lớp:**
   * Load file trọng số `.pt` thông qua `ultralytics.YOLO`.
   * Kiểm tra bắt buộc:
     ```python
     # Kiểm tra tên class trong checkpoint khớp hoàn toàn DET_CLASSES
     loaded_names = tuple(self.model.names[i] for i in range(len(DET_CLASSES)))
     if loaded_names != DET_CLASSES:
         raise ValueError(f"Mismatch classes: {loaded_names} != {DET_CLASSES}")
     ```
2. **Hàm `predict(frame_bgr: np.ndarray) -> list[Detection]`:**
   * Đầu vào: `frame_bgr` (`uint8`, shape `(H, W, 3)`).
   * Suy luận với `conf=cfg.conf_threshold`, `iou=cfg.iou_threshold`, `imgsz=cfg.img_size`, `verbose=False`.
   * Trích xuất tensor `.boxes.xyxy`, `.boxes.conf`, `.boxes.cls` đưa về CPU NumPy.
   * Chuyển đổi thành danh sách đối tượng `Detection` có tọa độ pixel nguyên bản trên frame gốc.

---

## 4. Chiến lược Huấn luyện trên Colab T4 (`scripts/train_detector.py`)

### 4.1. Cấu hình Dataset YAML cho Ultralytics
Tạo file động hoặc cấu hình `configs/bdd_yolo.yaml`:
```yaml
path: /path/to/bdd100k_yolo
train: images/train
val: images/val
names:
  0: car
  1: truck
  2: bus
  3: motorcycle
  4: bicycle
  5: pedestrian
  6: rider
  7: traffic_light
  8: traffic_sign
```

### 4.2. Siêu tham số huấn luyện khuyến nghị
* **Pretrained:** Luôn khởi tạo từ weights COCO (`yolo11s.pt`).
* **Input resolution:** `imgsz=640` (có thể thử nghiệm `imgsz=960` nếu biển báo quá nhỏ, lưu ý giảm `batch_size` tương ứng).
* **Batch size:** `16` (hoặc `32` với `amp=True` FP16 trên T4 16GB).
* **Optimizer & LR:** `AdamW` hoặc `SGD`, `cos_lr=True`, `warmup_epochs=3`.
* **Augmentation:** Giữ `mosaic=1.0` (tắt ở 10 epoch cuối `close_mosaic=10` để bounding box hội tụ sát biên thật), `fliplr=0.5`, tắt `flipud=0.0` (xe không bao giờ lộn ngược trên đường).

### 4.3. Lưu trữ & Checkpoint
* Lưu best weights (`best.pt`) về `weights/detector_best.pt`.
* Lưu log TensorBoard / CSV kết quả từng epoch (`results.csv`) để đưa vào báo cáo.

---

## 5. Các vấn đề thường gặp & Cách xử lý

| Vấn đề | Nguyên nhân | Giải pháp |
| :--- | :--- | :--- |
| AP lớp `motorcycle`, `bicycle` thấp | Mất cân bằng dữ liệu (BDD100K ở Mỹ chủ yếu là `car`) | Tăng cường trọng số mẫu (oversampling các ảnh có chứa xe máy/xe đạp) hoặc dùng `copy_paste` augmentation nhẹ |
| AP lớp `traffic_sign` thấp | Biển báo kích thước nhỏ (chỉ vài chục pixel) | Thử nghiệm tăng `imgsz` lên 800/960 hoặc dùng biến thể `yolo11s/m`; giảm `conf_threshold` riêng cho hiển thị cảnh báo |
| Chập chờn box giữa các frame video | Nhiễu dự đoán frame-by-frame | Có thể bật ByteTrack/BoT-SORT nhẹ tích hợp sẵn trong Ultralytics ở giai đoạn ADAS tracking (M6) |
