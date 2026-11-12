# Luật cho AI coding agent


## Bối cảnh
Dự án `adas-vision`: detection (YOLO) + lane/drivable segmentation (smp) + ADAS logic trên BDD100K/Cityscapes. Tài liệu đầy đủ ở `docs/`. Người dùng học bằng tiếng Việt, giữ thuật ngữ tiếng Anh; chạy Windows + VS Code, train trên Colab.

## Luật bắt buộc

### Quy trình
1. **Đọc docs liên quan trước khi code.** Docs là source of truth. Mâu thuẫn giữa yêu cầu và docs → hỏi, không tự quyết.
2. **Một milestone mỗi lần** (docs/13). Không làm trước phần của milestone sau.
3. **Nêu kế hoạch + danh sách file sẽ tạo/sửa** trước khi viết code.
4. **Chạy test thật** và báo cáo output; không khẳng định "đã chạy" nếu chưa chạy.
5. Thay đổi tên class, thứ tự class/kênh, đường dẫn, key config → **sửa docs trước**.
6. Không thêm thư viện ngoài `requirements.txt` nếu chưa hỏi.

### Dữ liệu — KHÔNG ĐOÁN
7. **Không giả định định dạng mask/nhãn BDD100K.** Luôn chạy `scripts/inspect_bdd.py` và dựa vào kết quả thực. Dừng và hỏi khi giá trị không khớp docs/03.
8. Không hard-code đường dẫn dataset; dùng `cfg.paths.*`.
9. Không tải/giả lập dữ liệu thật bằng số liệu bịa. Test dùng dữ liệu tổng hợp nhỏ.

### Code
10. Python ≥ 3.10, type hints, docstring ngắn cho hàm public, `pathlib.Path` cho đường dẫn, `logging` thay `print` trong `src/`.
11. Thứ tự class detection: `car, truck, bus, motorcycle, bicycle, pedestrian, rider, traffic_light, traffic_sign`. Kênh seg: `drivable_direct, drivable_alternative, lane_marking`. **Dùng hằng trong `constants.py`**, không viết lại chuỗi/số rời rạc.
12. Ảnh nội bộ là **BGR uint8 (H, W, 3)**; box nội bộ là **xyxy pixel** theo frame gốc. Chuyển RGB/chuẩn hoá chỉ trong preprocess của model.
13. Mọi ngưỡng/tham số đặt trong YAML. Cấm magic number trong logic ADAS.
14. `src/adas/adas/*` **không import torch/ultralytics/cv2** (chỉ numpy + stdlib).
15. Hàm thuần, dễ test; `scripts/*` chỉ là lớp mỏng gọi `src/`.
16. Không sửa mảng đầu vào (`frame`, `mask`) tại chỗ; copy khi vẽ.
17. Cẩn thận `cv2.resize` dùng `(W, H)` còn numpy shape là `(H, W)`.
18. Seed cố định khi train/subset (`project.seed`).
19. Windows: `if __name__ == "__main__":` quanh code dùng DataLoader multi-worker; không dùng symlink bắt buộc.

### Model
20. Preprocess inference **phải dùng chung hàm** với train.
21. Checkpoint seg lưu: `model_state, cfg, epoch, metrics, thresholds, size_hw, channels`.
22. Khi load detector, kiểm `model.names` khớp `DET_CLASSES`; sai → raise.
23. Không commit dữ liệu, weights, video lớn.

### Trung thực
24. Không bịa metric/kết quả. Nếu chưa train, để trống bảng kết quả.
25. Nếu không chắc API của thư viện (Ultralytics/smp/albumentations thay đổi theo version), **kiểm tra bằng cách chạy thử/in `help()`**, đừng đoán.
26. Khi bị lỗi, nêu nguyên nhân gốc + cách sửa, không "vá" bằng try/except nuốt lỗi.

## Định nghĩa "xong"
Một task xong khi: code chạy được, test liên quan pass, `ruff check` sạch, docs cập nhật nếu có thay đổi thiết kế, và báo cáo ngắn (đã làm gì, lệnh chạy, kết quả, điểm còn mở).

## Phong cách giao tiếp
Trả lời bằng tiếng Việt, thuật ngữ chuyên ngành giữ tiếng Anh. Ngắn gọn; ghi rõ giả định; hỏi lại khi mơ hồ thay vì đoán.

## Những thứ KHÔNG làm
- Không train model dài/tải dataset lớn mà không được yêu cầu.
- Không đổi framework (vd bỏ Ultralytics/smp) khi chưa thảo luận.
- Không tối ưu sớm (ONNX/TensorRT) trước M8.
- Không viết code điều khiển xe thật.
