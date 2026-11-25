# 13 — Kế hoạch triển khai (Implementation Plan)

Làm **tuần tự từng milestone**. Mỗi milestone có: mục tiêu, file cần tạo, **Definition of Done (DoD)**, prompt mẫu cho AI agent. Chỉ sang milestone kế khi DoD đạt và `pytest` pass.

Ước lượng thời gian (1 người, ~4 giờ/ngày): tổng ~ 3–5 tuần, tuỳ thời gian train.

---

## M0 — Skeleton & môi trường (0.5 ngày)

**File**: `pyproject.toml`, `requirements.txt`, `.gitignore`, `configs/default.yaml` (từ `17`), `src/adas/{__init__,constants,types,config}.py`, `tests/test_constants.py`, `scripts/check_env.py`, cây thư mục rỗng theo `05`.

**DoD**
- [ ] `pip install -e .` OK; `python -c "import adas"` OK.
- [ ] `adas.config.load_config("configs/default.yaml")` trả object truy cập kiểu `cfg.adas.ldw.warn_offset`; thiếu key → lỗi rõ ràng.
- [ ] `test_constants.py`: số class = 9, thứ tự đúng `05`.
- [ ] `ruff check .` sạch.

**Prompt mẫu**
```text
Thực hiện Milestone M0 trong docs/13. Tạo skeleton đúng docs/05, constants đúng nội dung trong docs/05,
types.py đúng docs/04 mục 2, config.py load YAML (docs/17) thành dataclass lồng nhau có validate.
Viết test. Không thêm tính năng ngoài docs.
```

---

## M1 — Data detection (1–2 ngày)

**File**: `scripts/inspect_bdd.py`, `src/adas/data/bdd.py`, `scripts/convert_bdd_det.py`, `scripts/make_subset.py`, `tests/test_data_bdd.py`.

**DoD**
- [ ] Inspect chạy trên 5 ảnh, sinh overlay + `report.md`.
- [ ] Convert train/val (subset) xong; `--validate` pass (5 cột, giá trị ∈ [0,1]).
- [ ] `data.yaml` sinh đúng 9 class.
- [ ] 20 ảnh mẫu vẽ box khớp bằng mắt.
- [ ] Test: hàm `bdd_label_to_yolo_lines` với frame giả (box ngoài ảnh, box nhỏ, category `train`).
- [ ] Thống kê `stats/det_*.json`.

**Prompt mẫu**
```text
Thực hiện M1 theo docs/06 mục 1,2,4 và docs/03. Viết hàm thuần trong src/adas/data/bdd.py,
script chỉ là lớp mỏng. Đừng đoán định dạng mask; script inspect phải in np.unique.
Viết unit test cho chuyển đổi box (clip, min_box_px, category bị bỏ).
```

---

## M2 — Data segmentation (1–2 ngày)

**File**: `src/adas/data/masks.py` (`pack_mask`, `unpack_mask`), `scripts/build_bdd_seg.py`, `src/adas/segmentation/dataset.py`, `tests/test_masks.py`.

**DoD**
- [ ] Người dùng xác nhận `value_map` từ báo cáo inspect và cập nhật config.
- [ ] Build mask train/val subset; thống kê tỉ lệ pixel mỗi kênh.
- [ ] Test round-trip pack/unpack; test `resize_pair` giữ được vạch mỏng (vạch 2 px rộng vẫn còn sau resize 0.5×).
- [ ] Dataset trả `(3,H,W)` ảnh float, `(3,H,W)` mask {0,1}; visual check 10 mẫu có augment.

**Prompt mẫu**
```text
Thực hiện M2 theo docs/06 mục 3,6 và docs/08. Bitmask: bit0 direct, bit1 alt, bit2 lane.
Resize mask bằng INTER_AREA + ngưỡng riêng từng kênh. Viết dataset + test.
Dừng và hỏi tôi nếu giá trị pixel trong report không khớp giả định.
```

---

## M3 — Detector (1 ngày code + train)

**File**: `src/adas/detection/{detector,traffic_light}.py`, `scripts/train_detector.py`, `scripts/eval_detector.py`, `tests/test_traffic_light_color.py`, `notebooks/colab_train.ipynb` (phần det).

**DoD**
- [ ] E0 overfit 50 ảnh: mAP50 cao (> 0.9) → pipeline đúng.
- [ ] E1/E2 train xong, có `best.pt`, bảng mAP từng class.
- [ ] `Detector.predict/track` trả `Detection` đúng; `track_id` ổn định trên video 10 giây.
- [ ] Test màu đèn trên ảnh tổng hợp (ô đỏ/xanh/vàng) pass.
- [ ] Eval trên Cityscapes val (nếu đã convert).

**Prompt mẫu**
```text
Thực hiện M3 theo docs/07. Viết Detector bọc Ultralytics (predict, track, reset_tracker),
kiểm tra model.names khớp DET_CLASSES, conf theo class. Viết classify_light_color theo docs/07 mục 5 + test.
Viết script train đọc detection.train.* trong config.
```

---

## M4 — Segmenter (1 ngày code + train)

**File**: `src/adas/segmentation/{model,losses,metrics,segmenter,postprocess}.py`, `scripts/train_segmenter.py`, `scripts/eval_segmenter.py`.

**DoD**
- [ ] `--overfit 50` loss → rất thấp, IoU > 0.9.
- [ ] Train S1 xong; `best.pt` chứa đủ metadata (docs/08 mục 4b).
- [ ] `Segmenter.predict` trả mask bool `(3,H,W)` cùng kích thước ảnh; preprocess khớp train.
- [ ] IoU từng kênh + ảnh so sánh GT/pred.
- [ ] Test: `IoUMeter` với tensor biết trước; `build_model` forward shape đúng.

**Prompt mẫu**
```text
Thực hiện M4 theo docs/08. smp DeepLabV3Plus resnet34, 3 kênh sigmoid, loss BCE(pos_weight)+Dice,
AMP, IoU tích luỹ, checkpoint kèm cfg/thresholds. Có cờ --overfit N và --resume.
Preprocess inference phải dùng chung hàm với train.
```

---

## M5 — ADAS logic (1–2 ngày)

**File**: `src/adas/adas/{lane_geometry,ldw,fcw,pedestrian,traffic,alerts,engine}.py`, các `tests/test_*.py` tương ứng.

**DoD**
- [ ] Mọi test ở `15` mục "ADAS" pass.
- [ ] Module ADAS **không import torch/ultralytics**.
- [ ] Mọi ngưỡng đọc từ config.
- [ ] Cooldown/hysteresis/debounce có test.

**Prompt mẫu**
```text
Thực hiện M5 theo docs/09. Chỉ NumPy + stdlib. Viết test TRƯỚC với mask hình thang tổng hợp
và chuỗi bbox giả (h tăng dần). Đảm bảo test_ldw, test_fcw, test_pedestrian, test_traffic, test_alerts pass.
```

---

## M6 — Pipeline, IO, Renderer, CLI (1–2 ngày)

**File**: `src/adas/{io,render,pipeline,cli}.py`, `tests/test_pipeline_smoke.py`.

**DoD**
- [ ] `adas run --source samples/clip.mp4` tạo `out.mp4`, `events.jsonl`, `summary.json`.
- [ ] Hoạt động với ảnh, thư mục ảnh, video, webcam.
- [ ] Ctrl+C vẫn đóng file đúng.
- [ ] Smoke test chạy với model giả (mock) không cần GPU.
- [ ] FPS + timings in ra; ≥ mục tiêu trên T4 (hoặc ghi lại con số thực).

**Prompt mẫu**
```text
Thực hiện M6 theo docs/11 và docs/04 mục 4. Pipeline nhận Detector/Segmenter qua dependency injection
để test bằng mock. Renderer không sửa frame gốc. Text tiếng Việt dùng PIL font.
```

---

## M7 — Streamlit demo (0.5–1 ngày)

**File**: `app/streamlit_app.py`.

**DoD**
- [ ] Upload video → chạy → xem kết quả + bảng sự kiện + tải file.
- [ ] Đổi ngưỡng ở sidebar có tác dụng.
- [ ] Model cache bằng `st.cache_resource`.

---

## M8 — Đánh giá tổng thể & báo cáo (1–2 ngày)

**File**: `scripts/eval_alerts.py`, `notebooks/01_eda_bdd100k.ipynb`, `docs/REPORT.md` (tạo sau).

**DoD**
- [ ] Bảng kết quả det/seg/FPS (mẫu ở `10`).
- [ ] Phân tích lỗi theo điều kiện (đêm/mưa).
- [ ] ≥ 3 video demo (ngày, đêm, mưa).
- [ ] Mục hạn chế + hướng phát triển (VN, sign classifier).

---

## M9 — Mở rộng (tuỳ chọn)

| Mở rộng | Mô tả |
|---|---|
| Sign classifier | Crop biển → MobileNet (GTSRB / dữ liệu VN) |
| Dữ liệu Việt Nam | Quay + gán nhãn 300–500 ảnh, fine-tune |
| Lane polynomial + curvature | Fit đường làn cong |
| Solid/dashed + white/yellow | Thêm kênh/thuộc tính → cảnh báo vượt vạch liền |
| Depth/khoảng cách | Ước lượng từ chiều cao bbox + giả định camera, hoặc depth model |
| ONNX/TensorRT | Tăng tốc |
| Multi-task model (YOLOP/HybridNets) | So sánh với 2-model |

---

## Quy trình mỗi phiên làm việc với AI agent
1. Nhắc agent đọc `docs/14_AGENT_RULES.md` + file docs liên quan milestone.
2. Yêu cầu agent **liệt kê kế hoạch + file sẽ đổi** trước khi code.
3. Code → chạy test → báo cáo kết quả thực (lệnh + output).
4. Commit nhỏ: `feat(m3): detector wrapper`, `test(m5): ldw hysteresis`.
5. Cập nhật docs nếu quyết định thay đổi.
