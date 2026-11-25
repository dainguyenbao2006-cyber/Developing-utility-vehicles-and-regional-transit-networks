# 10 — Training & Evaluation

## 1. Quy trình tổng thể

```text
M1 data ─► M3 train detector ─► eval det
      └──► M4 train segmenter ─► eval seg
                         └─► M5 ADAS ─► M6 pipeline ─► M8 eval tổng thể + báo cáo
```

## 2. Chạy trên Google Colab (GPU T4)

Khuyến nghị: **không đọc dataset trực tiếp từ Google Drive khi train** (rất chậm). Quy trình:

1. Lưu `bdd100k_subset.zip` (đã convert, ~ vài GB) trên Drive.
2. Trong notebook:
```python
from google.colab import drive; drive.mount('/content/drive')
!cp /content/drive/MyDrive/adas/bdd100k_subset.zip /content/ && unzip -q /content/bdd100k_subset.zip -d /content/data
!git clone <repo> /content/adas-vision && cd /content/adas-vision && pip install -r requirements.txt && pip install -e .
```
3. Dùng `configs/colab.yaml` (override `paths.processed_root: /content/data/processed`, `runs_root: /content/drive/MyDrive/adas/runs` để **checkpoint lưu về Drive**).
4. Train với `--resume` được vì Colab ngắt phiên bất kỳ lúc nào; lưu `last.pt` mỗi epoch.
5. Kiểm tra GPU: `!nvidia-smi`; chọn Runtime → T4 GPU.

Ước lượng thời gian (tham khảo, phụ thuộc phiên bản):
| Việc | Cấu hình | Thời gian xấp xỉ |
|---|---|---|
| Detector | yolo11s, imgsz 960, 10K ảnh, 50 epoch | vài giờ trên T4 |
| Segmenter | DeepLabV3+ r34, 384×640, 10K ảnh, 40 epoch | 1–3 giờ trên T4 |

→ Chia nhỏ: train detector một phiên, segmenter phiên khác; luôn lưu checkpoint lên Drive.

## 3. Thứ tự thí nghiệm khuyến nghị

| # | Thí nghiệm | Mục đích |
|---|---|---|
| E0 | Overfit 50 ảnh (cả det và seg) | Sanity check pipeline |
| E1 | Baseline: yolo11n @640, 2K ảnh, 10 epoch | Chạy nhanh end-to-end |
| E2 | yolo11s @960, 10K ảnh | Baseline chính |
| E3 | + oversample class hiếm | Cải thiện motorcycle/rider |
| E4 | imgsz 1280 | Vật nhỏ (sign/light) |
| S1 | DeepLabV3+ r34 @384×640 | Baseline seg |
| S2 | Unet r34 @512×896 | Vạch kẻ mỏng |
| S3 | Loss có Tversky/Focal cho lane | Tăng recall vạch |
| X1 | Eval Cityscapes (cross-domain) | Domain gap |
| X2 | Fine-tune thêm dữ liệu tự quay VN (nếu có) | Giảm domain gap |

Mỗi thí nghiệm: ghi vào `runs/experiments.csv` (id, config hash, metric chính, ghi chú) — để báo cáo có bảng ablation.

## 4. Metric & cách báo cáo

### Detection
- mAP@0.5, mAP@0.5:0.95, Precision, Recall — **tổng và từng class**.
- AP theo kích thước (small/medium/large) nếu có (COCO-style: dùng `pycocotools` hoặc kết quả Ultralytics).
- Confusion matrix (normalized).
- Theo điều kiện: `timeofday`, `weather`.

### Segmentation
- IoU từng kênh (direct, alt, lane) + mIoU.
- Precision/Recall/F1 kênh lane.
- Boundary F-score (tuỳ chọn) cho vạch mỏng.
- Theo điều kiện: `timeofday`, `weather`, `scene`.

### Hệ thống
- FPS end-to-end, thời gian từng khối (ms): detect, seg, adas, render.
- VRAM peak.
- ADAS: không có ground truth đầy đủ → **đánh giá định tính + bảng sự kiện**: với N clip tự chọn gán nhãn tay "có lệch làn/không" theo khoảng thời gian, tính precision/recall của alert (mức sự kiện) — `scripts/eval_alerts.py` (so `events.jsonl` với `annotations.json`).

## 5. Phân tích lỗi (bắt buộc cho báo cáo)
1. Lấy 50 ảnh val có lỗi lớn nhất (det: nhiều FN/FP; seg: IoU thấp nhất) → lưu `runs/error_analysis/`.
2. Phân loại nguyên nhân: đêm, mưa, che khuất, vật nhỏ, nhãn nhiễu, domain.
3. Đề xuất cải thiện cho từng nhóm.

## 6. Mẫu bảng kết quả

| Model | imgsz | mAP50 | mAP50-95 | car | pedestrian | motorcycle | traffic_sign | FPS |
|---|---|---|---|---|---|---|---|---|
| yolo11n | 640 | | | | | | | |
| yolo11s | 960 | | | | | | | |

| Model | size | IoU direct | IoU alt | IoU lane | FPS |
|---|---|---|---|---|---|
| DeepLabV3+ r34 | 384×640 | | | | |
| Unet r34 | 512×896 | | | | |

## 7. Tái lập
- Seed cố định (`project.seed`).
- Lưu: config YAML đầy đủ, git commit, phiên bản thư viện (`pip freeze > runs/<name>/env.txt`).
- Không đổi `val` giữa các thí nghiệm.
