# 10 — Phân chia Nhiệm vụ Nhóm (6 Thành viên)

Tài liệu này phân rã toàn bộ khối lượng công việc của dự án `adas-vision` cho **06 thành viên**, đảm bảo **phát triển song song**, **rõ ràng ranh giới file sở hữu** (giảm thiểu xung đột merge code trên Git) và có **giao ước bàn giao (Handoff Contracts)** giữa các thành viên.

---

## 1. Bảng Phân công Tổng thể (6 Thành viên)

| STT | Vai trò (Role) | Phạm vi phụ trách chính | Thư mục / File sở hữu |
| :---: | :--- | :--- | :--- |
| **TV1** | **Tech Lead & Core Infrastructure** | Thiết lập cấu trúc dự án, Data Contracts, Config Loader, Pipeline ghép nối End-to-End, CLI, Review PR & CI (`ruff`/`pytest`) | `configs/*.yaml`, `src/utils/constants.py`, `src/utils/types.py`, `src/utils/config.py`, `src/pipeline.py`, `src/cli.py` |
| **TV2** | **Data Engineer (BDD100K & Cityscapes)** | Kiểm tra định dạng dữ liệu gốc, chuyển đổi nhãn Detection (YOLO) & Segmentation (3 kênh), xây dựng DataLoader, Augmentation dùng chung, ánh xạ Cross-Dataset Cityscapes | `scripts/inspect_bdd.py`, `scripts/prepare_data.py`, `src/data/*.py`, `tests/test_data_conversion.py` |
| **TV3** | **ML Engineer — Object Detection** | Huấn luyện & tinh chỉnh **YOLO11** trên BDD100K (9 classes), tối ưu siêu tham số trên Colab T4, viết wrapper suy luận và thực hiện **Benchmark so sánh mô hình YOLO** | `src/models/detector.py`, `scripts/train_detector.py`, `notebooks/train_yolo11.ipynb`, `configs/detector_yolo11.yaml` |
| **TV4** | **ML Engineer — Semantic Segmentation** | Xây dựng & huấn luyện **DeepLabV3+** (`smp`, 3 kênh multi-label), thiết kế Combo Loss (BCE + Dice), tối ưu ngưỡng threshold từng kênh, đóng gói checkpoint chuẩn & test chéo Cityscapes | `src/models/segmenter.py`, `scripts/train_segmenter.py`, `notebooks/train_deeplab.ipynb`, `configs/segmenter_deeplab.yaml` |
| **TV5** | **ADAS Algorithm Engineer** | Thiết kế & cài đặt toàn bộ logic cảnh báo thuần NumPy (**LDW**, **FCW Proxy**, **PCW**, **TSR**, **Alert Manager** với hysteresis/cooldown), viết bộ Unit Test ADAS đầy đủ | `src/adas/ldw.py`, `src/adas/fcw.py`, `src/adas/pcw.py`, `src/adas/alert_manager.py`, `tests/test_adas_*.py` |
| **TV6** | **Visualization, Web App & QA/Evaluation** | Xây dựng module vẽ HUD overlay & xuất video, giao diện web **Streamlit**, script đánh giá tổng hợp & đo FPS, chuẩn bị ≥ 3 video demo (ngày/đêm/mưa) và tổng hợp báo cáo kết quả | `src/visualization/*.py`, `app/streamlit_app.py`, `scripts/evaluate.py`, `tests/test_visualizer.py`, tài liệu báo cáo kết quả |

---

## 2. Chi tiết Nhiệm vụ & Sản phẩm Bàn giao (Deliverables) từng Thành viên

### Thành viên 1 — Tech Lead & Core Infrastructure
* **Nhiệm vụ chi tiết:**
  1. Khởi tạo `src/utils/constants.py` (`DET_CLASSES`, `SEG_CHANNELS`), `src/utils/types.py` (`Detection`, `ADASAlert`, `FrameAnalysis`) và `src/utils/config.py` (nạp & validate YAML).
  2. Thiết lập chuẩn lint `ruff` và cấu hình chạy `pytest` tự động cho cả nhóm.
  3. Viết `src/pipeline.py` kết nối `Detector` + `Segmenter` + `ADASAnalyzer` + `Visualizer` + `EventLogger`.
  4. Viết giao diện dòng lệnh `src/cli.py` (`python -m adas.cli run ...`).
  5. Review Pull Request từ 5 thành viên còn lại, kiểm soát không có magic number và không vi phạm quy chuẩn `src/adas/` thuần NumPy.

### Thành viên 2 — Data Engineer (BDD100K & Cityscapes)
* **Nhiệm vụ chi tiết:**
  1. Hoàn thiện script tải dữ liệu (`data/dowload_data_bdd100k.py` hoặc hướng dẫn chuẩn) và viết `scripts/inspect_bdd.py` để xác minh cấu trúc folder, tên class JSON và giá trị pixel trong mask drivable/lane.
  2. Viết bộ chuyển đổi nhãn BDD100K Detection JSON $\rightarrow$ YOLO `.txt` (chuẩn hóa 9 lớp).
  3. Viết bộ chuyển đổi mask BDD100K $\rightarrow$ mask 3 kênh nhị phân `(drivable_direct, drivable_alternative, lane_marking)`.
  4. Xây dựng `src/data/transforms.py` (Albumentations) đảm bảo **hàm preprocess inference dùng chung với validation**.
  5. Tạo script chia subset cố định (`seed=42`) để TV3 và TV4 huấn luyện/đánh giá trên cùng tập val.
  6. Viết dataset loader ánh xạ nhãn `road` của Cityscapes val phục vụ TV4 đánh giá Cross-Dataset.

### Thành viên 3 — ML Engineer (Object Detection — YOLO11)
* **Nhiệm vụ chi tiết:**
  1. Chuẩn bị file cấu hình dataset YOLO (`configs/bdd_yolo.yaml`) khớp 9 lớp `DET_CLASSES`.
  2. Huấn luyện `yolo11n.pt` và `yolo11s.pt` trên Colab T4 (`scripts/train_detector.py`), theo dõi TensorBoard/CSV loss và khắc phục hiện tượng mất cân bằng lớp (`motorcycle`, `bicycle`, `traffic_sign`).
  3. Thực hiện bảng **Benchmark so sánh mô hình** (YOLO11n vs YOLO11s vs mô hình baseline YOLO khác) về mAP@0.5, mAP@0.5:0.95, Params và Latency.
  4. Viết `src/models/detector.py` kèm kiểm tra bắt buộc `model.names == DET_CLASSES`.
  5. Bàn giao file trọng số `weights/detector_best.pt` và bảng metric cho TV1 & TV6.

### Thành viên 4 — ML Engineer (Semantic Segmentation — DeepLabV3+)
* **Nhiệm vụ chi tiết:**
  1. Xây dựng kiến trúc `smp.DeepLabV3Plus` (`resnet34` và thử nghiệm `mobilenet_v2` làm đối chứng) trong `src/models/segmenter.py`.
  2. Cài đặt **Combo Loss** (`BCEWithLogitsLoss + DiceLoss`) trong `scripts/train_segmenter.py` để xử lý vạch kẻ đường mỏng.
  3. Huấn luyện trên Colab T4 với Mixed Precision (AMP), quét tìm ngưỡng `thresholds` tối ưu cho từng kênh trên tập val.
  4. Đóng gói checkpoint `weights/segmenter_best.pt` đúng chuẩn 7 trường (`model_state, cfg, epoch, metrics, thresholds, size_hw, channels`).
  5. Chạy đánh giá Cross-Dataset IoU trên **Cityscapes val** (dùng phép hợp `direct | alternative`).

### Thành viên 5 — ADAS Algorithm Engineer
* **Nhiệm vụ chi tiết:**
  1. Phát triển `src/adas/ldw.py`: Logic tính điểm ego với `camera_offset_ratio`, kiểm tra overlap vệt bánh xe với `lane_marking`, kiểm tra `drivable_direct` vs `drivable_alternative`, cài đặt temporal hysteresis.
  2. Phát triển `src/adas/fcw.py`: Tạo Ego Corridor, lọc phương tiện phía trước, tính tỷ lệ diện tích bbox và tốc độ tăng trưởng ($\Delta\text{Area}/\Delta t$) qua history buffer.
  3. Phát triển `src/adas/pcw.py`: Phát hiện `pedestrian` / `rider` / `bicycle` trong vùng làn nguy hiểm.
  4. Phát triển `src/adas/alert_manager.py`: Xếp hạng ưu tiên cảnh báo, cơ chế cooldown, và xuất cấu trúc Event Log.
  5. Viết trọn bộ Unit Test (`tests/test_adas_ldw.py`, `tests/test_adas_fcw_pcw.py`) bằng mặt nạ và box giả lập (chạy độc lập **không cần chờ TV3/TV4 train xong model**).

### Thành viên 6 — Visualization, Web App & QA/Evaluation
* **Nhiệm vụ chi tiết:**
  1. Viết `src/visualization/hud.py`: Vẽ mask bán trong suốt, vẽ bounding box theo nhóm màu cảnh báo, vẽ Ego Corridor, hiển thị bảng trạng thái LDW/FCW/PCW và FPS (đảm bảo copy frame trước khi vẽ).
  2. Xây dựng Web App tương tác `app/streamlit_app.py`: Upload video, slider chỉnh ngưỡng ADAS trực tiếp, xem video output, thống kê biểu đồ & tải file CSV log.
  3. Viết `scripts/evaluate.py` đo thời gian suy luận từng khâu và tổng **FPS End-to-End** trên Colab T4.
  4. Sưu tầm & chạy test thực tế trên tối thiểu **3–5 video dashcam** (ban ngày, ban đêm, trời mưa, đường đông xe máy).
  5. Tổng hợp kết quả định lượng, chụp ảnh phân tích lỗi (Error Analysis) và hoàn thiện báo cáo cuối kỳ.

---

## 3. Ma trận Trách nhiệm (RACI)

*(**R**: Responsible - Người làm chính | **A**: Accountable - Chịu trách nhiệm duyệt | **C**: Consulted - Hỗ trợ/Góp ý | **I**: Informed - Cập nhật thông tin)*

| Hạng mục / Milestone | TV1 (Lead) | TV2 (Data) | TV3 (Det) | TV4 (Seg) | TV5 (ADAS) | TV6 (UI/QA) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M1:** Core Infra, YAML, Constants | **R / A** | C | C | C | C | I |
| **M2:** Inspect & Convert BDD100K | A | **R** | C | C | I | I |
| **M3:** Train & Benchmark YOLO11 | A | C | **R** | I | I | C |
| **M4:** Train DeepLabV3+ & Cityscapes | A | C | I | **R** | I | C |
| **M5:** ADAS Logic & Unit Tests | A | I | C | C | **R** | C |
| **M6:** Pipeline, HUD & Streamlit App | **R** | I | I | I | C | **R / A** |
| **M7:** End-to-End Eval & Demo Report | A | C | C | C | C | **R** |

---

## 4. Lịch trình Phối hợp Song song (Parallel Execution Plan)

Để nhóm 6 người không bị nghẽn cổ chai chờ đợi nhau:

```
Tuần 1 (M1 + M2):
  ├── TV1: Chốt constants.py, types.py, configs YAML (ngày 1–2) ──► Mở khóa cho TV2,3,4,5,6
  ├── TV2: Chạy inspect_bdd.py, viết Data Converter & Transforms
  ├── TV5: Viết trước ADAS Logic & Unit Test bằng dữ liệu NumPy giả lập
  └── TV6: Dựng khung Visualizer & Streamlit mockup dùng dữ liệu giả lập

Tuần 2 (M3 + M4 + M5):
  ├── TV2: Hoàn thiện tập subset & Cityscapes val loader ──► Bàn giao cho TV3, TV4
  ├── TV3: Train YOLO11 trên Colab T4 + Benchmark các biến thể YOLO
  ├── TV4: Train DeepLabV3+ trên Colab T4 + Tối ưu ngưỡng kênh
  └── TV5 & TV6: Tinh chỉnh ADAS + HUD với mock output, chuẩn bị video test ngày/đêm/mưa

Tuần 3 (M6):
  ├── TV3 & TV4: Bàn giao detector_best.pt & segmenter_best.pt
  ├── TV1: Ghép nối hoàn chỉnh pipeline.py & CLI
  ├── TV5: Hiệu chỉnh tham số adas_rules.yaml trên video thật (giảm báo giả)
  └── TV6: Tích hợp vào Streamlit App & xuất video demo có overlay

Tuần 4 (M7):
  ├── TV1–TV6: Kiểm tra chéo (pytest, ruff check sạch 100%)
  ├── TV4 & TV6: Chạy đánh giá Cityscapes val & đo FPS End-to-End trên T4
  └── Cả nhóm: Phân tích lỗi (Error Analysis), hoàn thiện báo cáo & chuẩn bị thuyết trình
```

---

## 5. Điểm Bàn giao Quan trọng (Handoff Contracts)

1. **TV1 $\rightarrow$ Cả nhóm (Cuối Ngày 2):** Phải đóng băng tên hằng số trong `constants.py` và cấu trúc `Detection` / `FrameAnalysis` trong `types.py` để tất cả các nhánh code khớp kiểu dữ liệu ngay từ đầu.
2. **TV2 $\rightarrow$ TV3 & TV4:** Sau khi chạy `scripts/inspect_bdd.py`, thông báo ngay cấu trúc folder và thống kê mẫu dữ liệu thật, đồng thời chốt file `data/splits/*.txt` để Detection và Segmentation đánh giá trên cùng một tập validation.
3. **TV3 & TV4 $\rightarrow$ TV1, TV5, TV6:** Mô hình bàn giao phải đi kèm file weight `.pt`, log metric thật, và đảm bảo API `predict(frame_bgr)` trả đúng tọa độ pixel frame gốc `(H, W)`.
