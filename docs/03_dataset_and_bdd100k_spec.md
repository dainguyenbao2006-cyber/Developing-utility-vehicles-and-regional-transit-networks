# 03 — Đặc tả Dữ liệu, Kiểm tra Nhãn & Pipeline Tiền xử lý

> [!CAUTION]
> **Quy tắc bất biến về dữ liệu:** Không được tự suy đoán cấu trúc thư mục, định dạng JSON hay giá trị pixel trong mask của bản BDD100K tải về. Bắt buộc viết và chạy `scripts/inspect_bdd.py` để xác minh thực tế trước khi viết bộ chuyển đổi nhãn.

---

## 1. Bộ dữ liệu chính: BDD100K

### 1.1. Nhãn Object Detection
* **Định dạng gốc BDD100K:** Tệp JSON chứa danh sách frame, mỗi frame có trường `name`, `attributes` (thời tiết, thời gian trong ngày, khung cảnh), và `labels` (chứa `category`, `box2d: {x1, y1, x2, y2}`).
* **Ánh xạ lớp sang chuẩn dự án (`DET_CLASSES`):**
  * Trong BDD100K gốc có 10 lớp phổ biến (`car`, `truck`, `bus`, `motor`, `bike`, `person`, `rider`, `traffic light`, `traffic sign`, `train`).
  * Quy tắc chuẩn hóa ánh xạ:
    | Nhãn gốc BDD100K | Chuẩn hóa `DET_CLASSES` | ID | Ghi chú |
    | :--- | :--- | :---: | :--- |
    | `car` | `car` | 0 | |
    | `truck` | `truck` | 1 | |
    | `bus` | `bus` | 2 | |
    | `motor` / `motorcycle` | `motorcycle` | 3 | |
    | `bike` / `bicycle` | `bicycle` | 4 | |
    | `person` / `pedestrian` | `pedestrian` | 5 | |
    | `rider` | `rider` | 6 | Người đi xe máy/xe đạp |
    | `traffic light` | `traffic_light` | 7 | |
    | `traffic sign` | `traffic_sign` | 8 | |
    | `train` / `other vehicle` / `trailer` | *(Bỏ qua)* | - | Loại trừ khỏi ground truth |
* **Chuyển đổi sang định dạng Ultralytics YOLO:**
  * Chuyển `[x1, y1, x2, y2]` sang chuẩn hóa $[0, 1]$: `[class_id, x_center, y_center, width, height]`.
  * Kiểm tra biên (`clip(0, 1)`) và lọc các box lỗi (`width <= 0` hoặc `height <= 0`).

### 1.2. Nhãn Semantic Segmentation (3 kênh độc lập)
Khác với bài toán phân loại đơn nhãn (Softmax), bài toán phân vùng đường và vạch kẻ đường dùng **3 kênh nhị phân độc lập** (vì vạch kẻ đường có thể nằm đè lên rìa vùng drivable):

| Kênh | Tên hằng số | Ý nghĩa | Nguồn gốc từ BDD100K |
| :---: | :--- | :--- | :--- |
| **0** | `drivable_direct` | Làn xe hiện tại đang chạy | Thư mục `drivable_maps` (kiểm tra pixel ID thực tế qua `inspect_bdd.py`, thường là ID 0 hoặc màu đỏ tùy bản PNG color/label) |
| **1** | `drivable_alternative` | Làn đường bên cạnh có thể đi | Thư mục `drivable_maps` (ID 1 hoặc màu xanh dương) |
| **2** | `lane_marking` | Vạch kẻ sơn đường | Thư mục `lane` / `lane_maps` (toàn bộ pixel foreground $\neq$ background; cần verify giá trị background là `255` hay `0`) |

---

## 2. Script Kiểm tra Dữ liệu (`scripts/inspect_bdd.py`)

Trước khi chạy chuyển đổi hay huấn luyện, chạy script kiểm tra để in ra:
1. Số lượng ảnh trong `train`, `val`, `test` và tỷ lệ khớp file giữa ảnh và JSON/mask.
2. Toàn bộ các giá trị `category` duy nhất xuất hiện trong JSON detection.
3. Shape, dtype, giá trị `np.unique()` của ảnh mask trong `drivable` và `lane` (xác nhận đây là mask grayscale ID hay mask màu RGB).
4. Thống kê phân bố số lượng bbox theo từng lớp và tỷ lệ pixel foreground trên background của từng kênh segmentation.

*Nếu kết quả thực tế khác với tài liệu này, dừng lại và cập nhật bảng ánh xạ ở Mục 1 trước khi code Converter.*

---

## 3. Bộ dữ liệu Kiểm thử Chéo: Cityscapes

* **Mục đích:** Đánh giá độ tổng quát hóa (Cross-Dataset Evaluation) trên tập `cityscapes/val` (500 ảnh độ phân giải $2048 \times 1024$).
* **Mapping nhãn Segmentation Cityscapes:**
  * Lớp `road` (trainId = 0 / labelId = 7) trong Cityscapes không tách biệt giữa `direct` và `alternative` như BDD100K.
  * Do đó, khi đánh giá chéo mô hình Segmenter trên Cityscapes, sử dụng phép hợp:
    $$\text{Pred\_Road} = (\text{Pred\_Direct} \lor \text{Pred\_Alternative})$$
    để tính IoU đối chiếu với ground truth `road` của Cityscapes.
* **Mapping nhãn Detection Cityscapes (nếu xuất bbox từ `gtFine` polygons):**
  * `car`, `truck`, `bus`, `motorcycle`, `bicycle`, `person` $\rightarrow$ `pedestrian`, `rider`.

---

## 4. Chiến lược Chia Tập & Subset (GPU Colab T4)

Do BDD100K đầy đủ có 70,000 ảnh train và 10,000 ảnh val:
1. **Subset phát triển nhanh (Dev/Debug Subset):**
   * Lấy mẫu phân tầng (stratified theo thời tiết/ngày-đêm nếu có) ví dụ 5,000 ảnh train / 1,000 ảnh val để kiểm chứng pipeline và debug nhanh.
2. **Tập huấn luyện chính (Full / Large Subset):**
   * Chạy trên Colab T4 theo khả năng lưu trữ và thời gian phiên (ví dụ 20,000–70,000 ảnh cho Detection; 10,000–20,000 ảnh cho Segmentation).
3. **Tính tái lập (Reproducibility):**
   * Mọi thao tác chọn subset bắt buộc cố định bằng `project.seed` trong YAML (`seed = 42`) và lưu danh sách tên file vào `data/splits/*.txt` để hai model dùng chung cùng tập val.

---

## 5. Pipeline Tiền xử lý & Augmentation (`src/data/transforms.py`)

> [!IMPORTANT]
> Hàm tiền xử lý (Resize, RGB conversion, Normalize) cho Inference **bắt buộc dùng chung** với hàm validation trong quá trình Train để tránh sai lệch phân phối dữ liệu (train-inference skew).

### 5.1. Cho Segmentation (DeepLabV3+)
Sử dụng `albumentations`:
* **Train Augmentations:**
  * `Resize(height=H, width=W)` (mặc định $384 \times 640$ giữ gần đúng tỷ lệ 16:9 của dashcam).
  * `HorizontalFlip(p=0.5)` *(Lưu ý: Khi lật ngang ảnh, `drivable_direct` vẫn là `drivable_direct`, chỉ đảm bảo đồng bộ mask).*
  * `RandomBrightnessContrast(p=0.4)`, `HueSaturationValue(p=0.3)`, `RandomShadow(p=0.2)` / `MotionBlur(p=0.2)` để tăng khả năng chống chịu điều kiện đêm và mưa.
  * `Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))` (ImageNet stats).
  * `ToTensorV2()`.
* **Val / Inference Preprocess:**
  * Chỉ gồm `Resize(H, W)` $\rightarrow$ `BGR2RGB` $\rightarrow$ `Normalize` $\rightarrow$ `ToTensorV2()`.

### 5.2. Cho Object Detection (YOLO11)
* Tận dụng pipeline augmentation tích hợp của Ultralytics (Mosaic, HSV, Scale, Flip) đã được kiểm soát qua file hyperparameter YAML, và giữ nguyên ảnh BGR/RGB đúng theo API của `ultralytics`.
