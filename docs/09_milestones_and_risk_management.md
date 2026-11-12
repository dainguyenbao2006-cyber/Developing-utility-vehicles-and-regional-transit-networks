# 09 — Lộ trình Triển khai (Milestones) & Quản lý Rủi ro

## 1. Lộ trình Triển khai theo Milestone (M1 – M7)

Nguyên tắc triển khai: **Thực hiện tuần tự từng Milestone, chỉ chuyển Milestone khi Definition of Done (DoD) của bước hiện tại đạt chuẩn.**

| Milestone | Tên giai đoạn | Công việc chính | Tiêu chí hoàn thành (DoD) |
| :---: | :--- | :--- | :--- |
| **M1** | **Core Infra & Configs** | Xây dựng `constants.py`, `types.py`, config loader YAML, chuẩn các file `configs/*.yaml` | Load config sạch, `ruff check` & `test_constants_and_configs.py` pass |
| **M2** | **Data Inspection & Pipeline** | Viết `scripts/inspect_bdd.py`, chạy kiểm tra cấu trúc mask/JSON thật; viết Converter & `src/data/` Dataset + Transforms | Xác minh được giá trị pixel BDD100K; tạo được subset seed=42; test data conversion pass |
| **M3** | **Object Detection (YOLO11)** | Huấn luyện YOLO11 trên Colab T4 (`scripts/train_detector.py`), đánh giá & benchmark đối sánh | Đạt `weights/detector_best.pt`, `mAP@0.5 >= 0.40` (hoặc có báo cáo tuning), `detector.py` load khớp `DET_CLASSES` |
| **M4** | **Semantic Segmentation (DeepLabV3+)** | Huấn luyện DeepLabV3+ (`scripts/train_segmenter.py`), tối ưu Combo Loss & ngưỡng từng kênh | Đạt `weights/segmenter_best.pt` đúng chuẩn keys checkpoint; `drivable_direct IoU >= 0.70`, `lane_marking IoU >= 0.30` |
| **M5** | **ADAS Logic Engine** | Code thuần NumPy cho `ldw.py`, `fcw.py`, `pcw.py`, `alert_manager.py` | Toàn bộ unit test ADAS (`tests/test_adas_*.py`) pass 100%, không import `torch`/`cv2` trong `src/adas/` |
| **M6** | **End-to-End Pipeline, HUD & App** | Ghép `pipeline.py`, vẽ HUD (`visualization/`), CLI và Web App Streamlit | Chạy mượt trọn pipeline trên video mẫu, xuất video overlay + CSV log, `FPS >= 15` trên T4 |
| **M7** | **Cross-Dataset Eval & Báo cáo** | Đánh giá chéo trên Cityscapes val, chạy demo trên ≥ 3 clip (ngày/đêm/mưa), viết báo cáo phân tích lỗi | Đầy đủ bảng metric, 3+ video demo, báo cáo lỗi & hướng phát triển (ONNX/TensorRT ở M8) |

*(Ghi chú: Tối ưu hóa mô hình sang ONNX / TensorRT thuộc pha mở rộng M8 — chỉ thực hiện sau khi hoàn thành toàn bộ M1–M7.)*

---

## 2. Quản lý Rủi ro Kỹ thuật (Risk Register)

| ID | Rủi ro | Mức độ | Nguyên nhân | Giải pháp phòng ngừa & Giảm thiểu |
| :---: | :--- | :---: | :--- | :--- |
| **R1** | **Sai lệch định dạng mask/nhãn BDD100K** | Cao | Các bản phân phối BDD100K (Kaggle vs Official) có thể khác cấu trúc folder hoặc giá trị pixel (RGB vs Index ID) | **Bắt buộc** chạy `scripts/inspect_bdd.py` ở M2 trước khi code Dataset/Converter |
| **R2** | **Vạch kẻ đường quá mỏng bị mất khi resize** | Cao | Resize từ $1280\times720$ xuống $384\times640$ làm vạch mỏng còn 1–2 pixel | Giữ tỷ lệ khung hình 16:9 khi resize; dùng Dice + BCE Loss (hoặc tăng `pos_weight` cho kênh `lane_marking`); quét ngưỡng Sigmoid riêng cho kênh này |
| **R3** | **Domain Gap trên đường Việt Nam** | Trung bình | BDD100K chủ yếu chụp tại Mỹ (ít xe máy, vạch đường khác VN) | Ghi nhận rõ trong giới hạn dự án; tăng cường augmentation ánh sáng/nhòe chuyển động; có thể fine-tune bổ sung dữ liệu VN ở pha mở rộng |
| **R4** | **FCW báo giả khi vào cua hoặc thiếu khoảng cách thật** | Trung bình | Không có CAN-bus (tốc độ xe) và không có chiều sâu (LiDAR) | Lọc xe theo Ego Corridor giao với `drivable_direct` thay vì hình chữ nhật cố định; kết hợp ngưỡng diện tích + tốc độ phình bbox ($\Delta\text{Area}$) và smoothing |
| **R5** | **Giới hạn thời gian & VRAM Google Colab T4** | Trung bình | Phiên Colab miễn phí ngắt sau vài giờ; dataset 100K ảnh rất nặng | Tách nhỏ subset train có seed cố định; lưu checkpoint mỗi epoch lên Google Drive; bật Mixed Precision (`amp=True`) |
| **R6** | **Lỗi lệch tọa độ giữa Detection và Segmentation** | Cao | YOLO11 và DeepLabV3+ dùng kích thước ảnh đầu vào khác nhau (`640x640` vs `384x640`) | Quy chuẩn toàn bộ đầu ra của hai model phải quy đổi ngược về **hệ tọa độ pixel của frame gốc `(H, W)`** trước khi đưa vào ADAS Engine |
