# 11 — Data Pipeline (TV2): Hướng dẫn chạy & Handoff Contract

Tài liệu này mô tả cách dùng code TV2. Đặc tả gốc nằm ở `docs/03_dataset_and_bdd100k_spec.md`, tên lớp và kênh lấy từ `src/utils/constants.py` (docs/02 §2).

## 1. File sở hữu

| File | Vai trò |
|---|---|
| `data/dowload_data_bdd100k.py` | Giải nén zip BDD100K / tải qua mirror / `kagglehub`, sửa layout, xác minh cấu trúc |
| `scripts/inspect_bdd.py` → `src/data/inspection.py` | Xác minh cấu trúc thư mục, tên category JSON gốc, giá trị pixel mask drivable/lane |
| `scripts/prepare_data.py` → `src/data/prepare.py`, `src/data/splits.py` | `subset` (seed=42), `detection` (YOLO), `segmentation` (mask 3 kênh) |
| `src/data/converters.py` | JSON → YOLO 9 lớp; mask BDD → `(drivable_direct, drivable_alternative, lane_marking)` |
| `src/data/transforms.py` | Albumentations train / eval; `preprocess_inference` gọi lại đúng pipeline eval |
| `src/data/bdd_dataset.py` | `DetectionDataset`, `SegmentationDataset`, `build_dataloader` |
| `src/data/cityscapes_dataset.py` | `CityscapesRoadDataset`, `RoadIoU` (road = direct OR alternative) |
| `src/data/bdd_layout.py` | Đọc `configs/paths.yaml`, dò đường dẫn bên trong `bdd_root` |
| `tests/test_data_conversion.py` | Test bằng dữ liệu tổng hợp nhỏ (không cần BDD thật) |

## 2. Cấu hình (`configs/paths.yaml`)

Code dùng các key sau (theo docs/02 §4.1):

```yaml
project:
  seed: 42
paths:
  bdd_root: "D:/datasets/bdd100k"
  cityscapes_root: "D:/datasets/cityscapes"
  processed_root: "data/processed"   # ĐỀ XUẤT thêm — tuỳ chọn, mặc định data/processed
  splits_dir: "data/splits"          # ĐỀ XUẤT thêm — tuỳ chọn, mặc định data/splits
```

> `processed_root` và `splits_dir` là **key đề xuất mới**, cần TV1 duyệt rồi thêm vào docs/02 §4.1. Thiếu hai key này thì code dùng giá trị mặc định ghi bên cạnh. Mọi đường dẫn đều ghi đè được bằng tham số dòng lệnh (`--bdd-root`, `--out`, `--splits-dir`, `--seed`).

## 3. Quy trình (M2)

```powershell
# 0) Tải dữ liệu: BDD100K bắt buộc đăng nhập tại bdd-data.berkeley.edu
#    Tải 5 zip vào data/raw/:
#    100k_images_train.zip, 100k_images_val.zip, bdd100k_det_20_labels_trainval.zip,
#    bdd100k_drivable_labels_trainval.zip, bdd100k_lane_labels_trainval.zip
python data/dowload_data_bdd100k.py --zips-dir data/raw --out D:/datasets
# hoặc: --kaggle <owner>/<dataset>   (bản Kaggle bên thứ ba, cấu trúc phải kiểm tra lại)

# 1) BẮT BUỘC: xác minh định dạng thật (CLAUDE.md #7)
python scripts/inspect_bdd.py --split train val --out outputs/bdd_inspect.json
#    Đọc phần "Cảnh báo". Pixel mask drivable phải là {0,1,2}, nền lane phải là 0 hoặc 255.
#    Nếu khác: DỪNG, báo nhóm, cập nhật docs/03, rồi truyền --drivable-ids / --lane-background.

# 2) Chia subset cố định (seed = project.seed = 42), phân tầng weather × timeofday
python scripts/prepare_data.py subset                                    # dev: 5000 / 1000
python scripts/prepare_data.py subset --tag main --train-size 20000 --val-size 2000

# 3) Chuyển đổi đúng các ảnh trong split
python scripts/prepare_data.py all --use-splits                 # bộ dev
python scripts/prepare_data.py all --use-splits --tag main      # bộ main

# 4) Test
pytest tests/test_data_conversion.py -q
ruff check src/data scripts tests data/dowload_data_bdd100k.py
```

Ảnh được đặt vào `data/processed` bằng **hardlink**, nếu không được thì tự **copy**. Cách này không cần symlink và không cần quyền admin trên Windows. `data/splits/*.txt` phải **commit lên git** để cả nhóm dùng chung một tập val.

## 4. Handoff Contract

### → TV3 (YOLO11)

- `data/processed/bdd100k_det/data.yaml`: `names` lấy đúng `DET_CLASSES` (0 car … 8 traffic_light/traffic_sign). TV3 trỏ `configs/bdd_yolo.yaml` vào file này hoặc chép lại nội dung.
- Category BDD được ánh xạ như sau: `motor→motorcycle`, `bike→bicycle`, `person→pedestrian`, `traffic light→traffic_light`, `traffic sign→traffic_sign`. Bỏ `train`, `other vehicle`, `trailer`, `other person`.
- Box được clip vào ảnh, box nào có width ≤ 0 hoặc height ≤ 0 thì bị loại.
- `DetectionDataset` trả `image` dạng BGR uint8, `boxes` dạng xyxy pixel float32, `labels` int64.

### → TV4 (DeepLabV3+)

- `data/processed/bdd100k_seg/{images,masks}/<split>`: mask PNG 3 kênh, lưu dạng 0/255. `SegmentationDataset` trả mask 0/1, thứ tự kênh đúng `SEG_CHANNELS`.
- Các kênh độc lập, có thể chồng lên nhau, nên dùng sigmoid + BCE/Dice.
- Train dùng `build_train_transforms(640, 384)`, gồm Resize, HorizontalFlip 0.5, RandomBrightnessContrast 0.4, HueSaturationValue 0.3, RandomShadow 0.2, MotionBlur 0.2 và Normalize. Val dùng `build_eval_transforms(640, 384)`. Thêm `to_tensor=True` thì trả tensor CHW, tương đương `ToTensorV2`.
- Cross-dataset:

```python
from src.data import CityscapesRoadDataset, RoadIoU, build_eval_transforms
ds = CityscapesRoadDataset(cfg.paths.cityscapes_root, "val", transforms=build_eval_transforms(640, 384))
meter = RoadIoU(threshold=(thr_direct, thr_alt))
for item in ds:
    meter.update(probs_3ch, item["road"])   # road (H, W) {0,1}
print(meter.compute())                       # IoU của (direct OR alternative) so với GT road (labelId 7)
```

### → TV1 / TV6 (Inference)

```python
from src.data import preprocess_inference, denormalize
x = preprocess_inference(frame_bgr, 640, 384)   # frame BGR từ OpenCV -> (3, H, W) float32
```

`preprocess_inference` gọi đúng `build_eval_transforms`, nên kết quả trùng từng giá trị với validation (có test kiểm tra). Dùng `denormalize` để đưa tensor về ảnh RGB khi vẽ HUD hoặc debug.

## 5. Điểm còn mở

- [ ] Chạy `inspect_bdd.py` trên dữ liệu thật và dán kết quả vào docs/03, bỏ các chú thích "cần xác minh".
- [ ] TV1 duyệt hai key `paths.processed_root` và `paths.splits_dir`. Khi `src/utils/config.py` xong, chuyển `bdd_layout.load_data_paths` sang dùng loader chung.
- [ ] `src/utils/constants.py` hiện do TV2 điền theo đúng docs/02 §2 để không bị chặn việc. Nếu TV1 đã có bản riêng thì giữ bản của TV1.
- [ ] Chọn handle Kaggle chính thức cho cả nhóm, hoặc thống nhất tải zip từ trang BDD.
