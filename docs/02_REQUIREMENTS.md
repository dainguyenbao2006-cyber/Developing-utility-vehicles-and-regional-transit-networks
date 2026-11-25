# 02 — Yêu cầu hệ thống

## 1. Yêu cầu chức năng (FR)

### Dữ liệu
| ID | Yêu cầu | Acceptance criteria |
|---|---|---|
| FR-D1 | Convert nhãn detection BDD100K → định dạng YOLO theo mapping ở `03` | Script chạy xong, in thống kê số box/class; sample 20 ảnh vẽ lại box khớp trực quan; không có box ngoài [0,1] |
| FR-D2 | Tạo mask segmentation 3 kênh dạng bitmask PNG | Mask cùng kích thước ảnh gốc, giá trị trong [0,7]; sample overlay đúng |
| FR-D3 | Tạo subset theo số lượng ảnh + seed | Cùng seed → cùng danh sách ảnh; lưu `subset_{split}.txt` |
| FR-D4 | Convert Cityscapes val → format detection YOLO và mask drivable (road) để đánh giá cross-dataset | Có thống kê class; mapping theo `03` |
| FR-D5 | Script inspect mask để xác minh giá trị pixel trước khi convert | In `np.unique` và tỉ lệ pixel từng giá trị |

### Model
| ID | Yêu cầu | Acceptance criteria |
|---|---|---|
| FR-M1 | Train detector YOLO 9 class | Có `best.pt`, `results.csv`, confusion matrix; mAP per class |
| FR-M2 | Train segmenter 3 kênh multi-label | Có checkpoint `best.pt` chứa state_dict + config + metric; log IoU từng kênh mỗi epoch |
| FR-M3 | Detector & Segmenter đóng gói thành class có API rõ ràng (`Detector.predict`, `Segmenter.predict`) | Test chạy trên ảnh mẫu, trả đúng dataclass |
| FR-M4 | Tracking đa đối tượng (ByteTrack) | Mỗi detection có `track_id` ổn định qua frame |

### ADAS
| ID | Yêu cầu | Acceptance criteria |
|---|---|---|
| FR-A1 | **LDW**: tính offset xe so với tâm làn ego, cảnh báo trái/phải | Mask giả lập: đi giữa → không cảnh báo; lệch ≥ ngưỡng ≥ N frame → cảnh báo đúng hướng; có hysteresis |
| FR-A2 | **FCW proxy**: xe/vật trong làn ego, mức nguy hiểm theo vị trí + TTC proxy | Test với chuỗi bbox tăng kích thước → CRITICAL; bbox ngoài làn → không cảnh báo |
| FR-A3 | **Pedestrian in path**: người/rider/xe đạp có điểm chân nằm trong vùng drivable direct | Test với box giả lập |
| FR-A4 | **Traffic light**: phát hiện màu đèn (HSV); cảnh báo đèn đỏ khi đèn nằm vùng trung tâm đủ lớn | Test crop màu tổng hợp |
| FR-A5 | **Traffic sign ahead**: thông báo INFO khi có biển báo đủ lớn gần trung tâm | Có cooldown, không spam |
| FR-A6 | Alert engine: severity, debounce, cooldown, log JSONL | Không phát cùng loại alert quá `cooldown_s` |

### Ứng dụng
| ID | Yêu cầu | Acceptance criteria |
|---|---|---|
| FR-U1 | CLI: ảnh, thư mục ảnh, video, webcam | Xuất video overlay + `events.jsonl` + `summary.json` |
| FR-U2 | Overlay: box + id + class, mask drivable/lane, thanh cảnh báo, FPS | Theo bảng màu ở `11` |
| FR-U3 | Streamlit: upload video, chọn ngưỡng, xem kết quả | Chạy được `streamlit run app/streamlit_app.py` |

## 2. Yêu cầu phi chức năng (NFR)

| ID | Yêu cầu |
|---|---|
| NFR-1 | Tái lập: seed cố định, config YAML, lưu config vào checkpoint |
| NFR-2 | Hiệu năng: ≥ 15 FPS trên T4 (720p, FP16); có chế độ `--half` và `--imgsz` |
| NFR-3 | Chạy được trên Windows (pathlib, không dùng symlink bắt buộc, `num_workers` cấu hình được) |
| NFR-4 | Tách biệt: ADAS logic không import torch (thuần NumPy) để test nhanh |
| NFR-5 | Logging rõ ràng bằng `logging`, không dùng `print` trong thư viện |
| NFR-6 | Type hints + docstring ngắn cho mọi hàm public |
| NFR-7 | Mọi ngưỡng/tham số trong YAML, không hard-code |
| NFR-8 | Không commit dữ liệu/weights lớn (`.gitignore`) |
