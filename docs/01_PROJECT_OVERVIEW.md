# 01 — Tổng quan dự án

## 1. Bối cảnh & mục tiêu

Hệ thống hỗ trợ lái xe (ADAS) cần "nhìn" đường: biết có xe/người ở đâu, xe đang đi trong làn nào, có lệch làn không, có nguy cơ va chạm hay tín hiệu giao thông cần chú ý không.

**Mục tiêu**: xây dựng pipeline thị giác máy tính chạy trên video dashcam, gồm:

1. **Object Detection**: xe hơi, xe tải, xe buýt, xe máy, xe đạp, người đi bộ, người đi xe (rider), đèn giao thông, biển báo.
2. **Semantic Segmentation**: vùng lái xe được (làn ego + làn thay thế) và vạch kẻ đường.
3. **ADAS logic**: cảnh báo lệch làn, cảnh báo va chạm phía trước (proxy), người đi bộ trong làn, đèn đỏ, biển báo phía trước.
4. **Demo**: CLI xử lý video/ảnh/webcam + web demo Streamlit, xuất video có overlay và log sự kiện.

## 2. Phạm vi

### In scope
- Train/fine-tune detector và segmenter trên BDD100K (có thể dùng subset do giới hạn GPU).
- Đánh giá định lượng (mAP, IoU) và định tính (video demo).
- ADAS logic rule-based dựa trên output hai model.
- Đánh giá cross-dataset trên Cityscapes val.
- Báo cáo kết quả, phân tích lỗi.

### Out of scope
- Chạy trên xe thật, điều khiển phanh/lái, chứng nhận an toàn (ISO 26262).
- Ước lượng khoảng cách tuyệt đối chính xác (cần calibration camera/LiDAR).
- Phân loại chi tiết loại biển báo (cần dataset khác — xem phần mở rộng).
- Train từ đầu backbone (luôn dùng pretrained).

## 3. Giả định

| # | Giả định |
|---|---|
| A1 | Camera đặt giữa kính chắn gió, nhìn thẳng phía trước (giống BDD100K). Tâm ảnh ≈ tâm xe. Có thể bù bằng `adas.ldw.camera_offset_ratio`. |
| A2 | Tốc độ xe không có (không có CAN/GPS) → FCW dùng proxy hình học + tốc độ tăng kích thước bbox, không phải m/s thật. |
| A3 | GPU: Google Colab T4 (16GB) cho train; máy cá nhân Windows để code/inference. |
| A4 | Người dùng đã có kiến thức ML, đang học Deep Learning, ngôn ngữ chính Python. |
| A5 | Dữ liệu BDD100K được tải thủ công sau khi đăng ký trang chính thức. |

## 4. Deliverables

| # | Deliverable | Mô tả |
|---|---|---|
| D1 | Repo code `adas` | Package + scripts + configs + tests |
| D2 | Weights | `weights/detector_best.pt`, `weights/segmenter_best.pt` |
| D3 | Notebook train Colab | `notebooks/colab_train.ipynb` |
| D4 | Demo | `python -m adas.cli run ...` và `streamlit run app/streamlit_app.py` |
| D5 | Video demo | ≥ 3 clip (ngày, đêm, mưa) có overlay |
| D6 | Báo cáo | Bảng metric, biểu đồ, phân tích lỗi, hạn chế |

## 5. Tiêu chí thành công (mục tiêu tham khảo, không phải cam kết)

| Hạng mục | Mục tiêu |
|---|---|
| Detection mAP@0.5 (BDD val, 9 class) | ≥ 0.40 (car ≥ 0.70) |
| Drivable direct IoU (BDD val) | ≥ 0.70 |
| Lane marking IoU (BDD val) | ≥ 0.30 (vạch mỏng nên IoU thấp là bình thường) |
| FPS end-to-end trên T4, video 720p | ≥ 15 FPS |
| LDW trên 5 clip tự chọn | Cảnh báo đúng khi lệch rõ, ít báo giả khi đi thẳng |
| Test | `pytest` pass 100% cho phần ADAS logic và data conversion |

## 6. Rủi ro chính (chi tiết ở `16`)

- Domain gap: BDD100K chủ yếu đường Mỹ, ít xe máy → hiệu năng trên đường Việt Nam kém hơn.
- Vạch kẻ đường mỏng, dễ bị mất khi resize.
- FCW không có khoảng cách thật.
