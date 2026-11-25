# 11 — Inference, CLI & Demo app

## 1. CLI (`src/adas/cli.py`)

Dùng `argparse` (hoặc `typer`). Entry point: `python -m adas.cli ...` (và `adas` qua `pyproject.toml [project.scripts]`).

```text
adas run   --source <path|0> --config configs/default.yaml
           [--det-weights weights/detector_best.pt] [--seg-weights weights/segmenter_best.pt]
           [--out runs/demo/out.mp4] [--events runs/demo/events.jsonl]
           [--device 0] [--half/--no-half] [--imgsz 960] [--seg-every 1]
           [--show] [--max-frames N] [--no-seg] [--no-det]
adas eval-det  ...   (wrapper scripts/eval_detector.py)
adas eval-seg  ...
adas convert-bdd / convert-cityscapes ...
```

`--source`: đường dẫn ảnh, thư mục ảnh, video (`.mp4/.avi/.mov`), hoặc số nguyên (webcam index).

Output của `run`:
- `out.mp4` video overlay (codec `mp4v`; fps = fps nguồn).
- `events.jsonl` (xem `09`).
- `summary.json`: số frame, FPS trung bình, timings trung bình mỗi khối, số alert theo loại, thời lượng.

Yêu cầu:
- Xử lý Ctrl+C: đóng `VideoWriter`, ghi `summary.json`.
- `--show` dùng `cv2.imshow` (thoát bằng `q`).
- Hiển thị `tqdm` khi nguồn là file.
- Báo lỗi rõ khi thiếu weights/nguồn.

## 2. Pipeline (`pipeline.py`)

```python
class AdasPipeline:
    def __init__(self, cfg, det_weights, seg_weights, device): ...
    def process_frame(self, frame, frame_idx, ts) -> FrameResult: ...
    def run_source(self, source, writer=None, logger=None, show=False) -> dict: ...  # trả summary
```
- Seg chạy mỗi `seg_every_n_frames` frame; frame bỏ qua tái sử dụng `SegResult` trước (flag `seg_stale=True` trong `FrameResult.timings_ms` để debug).
- Với ảnh tĩnh: không tracking (`Detector.predict`), LDW state reset mỗi ảnh; FCW chỉ dùng proximity (không TTC).
- Reset tracker/engine khi bắt đầu nguồn mới.
- Timing: bọc từng bước bằng context manager `timer("detect")`.

## 3. Overlay (`render.py`)

Bảng màu (BGR) — cố định để demo nhất quán:

| Đối tượng | Màu |
|---|---|
| car | (255, 128, 0) |
| truck / bus | (255, 0, 128) |
| motorcycle / bicycle | (0, 200, 255) |
| pedestrian / rider | (0, 0, 255) |
| traffic_light | (0, 255, 255) |
| traffic_sign | (255, 255, 0) |
| drivable_direct (mask) | xanh lá (0, 200, 0), alpha 0.35 |
| drivable_alternative | xanh dương (200, 100, 0), alpha 0.25 |
| lane_marking | vàng/trắng (0, 255, 255), alpha 0.8 |

Thành phần vẽ:
1. Mask: blend theo thứ tự alt → direct → lane marking (lane trên cùng).
2. Box + nhãn `cls conf #track_id` (và màu đèn cho traffic_light).
3. Vạch ROI + đường tâm làn + tâm xe + offset (debug bật bằng `render.debug: true`).
4. **Thanh cảnh báo** phía trên: nền theo severity (INFO xanh dương, WARNING cam, CRITICAL đỏ nhấp nháy 2 Hz), chữ message (dùng tiếng Việt: `cv2.putText` không hỗ trợ Unicode → dùng **PIL `ImageFont`** với font TTF có dấu, vd `assets/fonts/NotoSans-Regular.ttf`; hoặc hiển thị không dấu nếu không có font).
5. Góc trái-dưới: FPS và timings.
6. Hàm `draw(frame, result, cfg) -> np.ndarray` **không sửa frame gốc** (copy).

## 4. Streamlit demo (`app/streamlit_app.py`)

Bố cục:
- Sidebar: chọn weights (mặc định `weights/`), `imgsz`, `conf`, bật/tắt seg/ADAS, các ngưỡng chính (`warn_offset`, `warn_ttc`...), `seg_every_n_frames`, `max_frames`.
- Main: upload video/ảnh → nút **Run** → progress bar → hiển thị video kết quả (`st.video`) + bảng sự kiện (`st.dataframe` từ `events.jsonl`) + biểu đồ offset theo thời gian + nút tải `out.mp4`, `events.jsonl`.
- Cache model: `@st.cache_resource` cho `Detector`/`Segmenter`.
- Video kết quả: ghi H.264 nếu có (nhiều trình duyệt không phát `mp4v`); fallback: chuyển bằng `ffmpeg` nếu có trong PATH, ngược lại cảnh báo người dùng tải file về xem.

## 5. Tối ưu hiệu năng
| Kỹ thuật | Tác dụng |
|---|---|
| FP16 (`half=True`) | ~1.5–2× nhanh hơn trên T4 |
| `seg_every_n_frames=2` | Giảm ~ nửa chi phí seg |
| Giảm imgsz detect 960 → 800/640 | Nhanh hơn, giảm AP vật nhỏ |
| Đọc/ghi frame trên thread riêng (queue) | Che I/O |
| ONNX/TensorRT | Mở rộng, cho triển khai |
| Upsample mask trên GPU | Tránh `cv2.resize` trên CPU với 3 kênh |

## 6. Gỡ lỗi thường gặp
- Video đen / 0 frame: kiểm tra codec & đường dẫn (Windows: dùng `pathlib`, tránh dấu `\` trong chuỗi thường).
- Mask lệch so với ảnh: kiểm tra thứ tự `(H, W)` vs `(W, H)` khi `cv2.resize`.
- Box lệch: toạ độ phải theo frame gốc (Ultralytics đã scale lại; nếu tự xử lý ONNX phải tự scale).
- Class sai tên: `model.names` khác `DET_CLASSES` → detector raise lỗi lúc load.
