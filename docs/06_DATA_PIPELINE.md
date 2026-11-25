# 06 — Data pipeline

Thứ tự thực hiện: **inspect → convert detection → build seg → subset → (cityscapes) → validate**.

> Quy tắc: logic nằm trong `src/adas/data/*.py` (có test), `scripts/*.py` chỉ parse argument rồi gọi hàm.

## 1. `scripts/inspect_bdd.py` (BẮT BUỘC chạy đầu tiên)

**Mục đích**: xác minh định dạng thực tế của mask trước khi viết converter.

Input: `--root`, `--n 5`. Với mỗi ảnh mẫu in ra:
- kích thước ảnh, kích thước mask, dtype
- `np.unique(mask, return_counts=True)` của drivable mask và lane mask (tỉ lệ pixel)
- lưu `runs/inspect/{name}_overlay.png`: ảnh + mask tô màu ngẫu nhiên theo giá trị

Output bắt buộc: file `runs/inspect/report.md` mô tả giá trị pixel → ý nghĩa **do người dùng xác nhận bằng mắt**. Sau đó **mới** cập nhật `configs/default.yaml` mục `segmentation.drivable.value_map` và `segmentation.lane.*`.

## 2. Convert detection: `scripts/convert_bdd_det.py`

```text
python scripts/convert_bdd_det.py --config configs/default.yaml --split train --split val
```

Thuật toán (`src/adas/data/bdd.py`):

```python
def bdd_label_to_yolo_lines(frame: dict, W: int, H: int, cfg) -> list[str]:
    lines = []
    for lab in frame.get("labels", []):
        box = lab.get("box2d")
        if box is None:
            continue
        name = BDD_DET_MAP.get(lab["category"])
        if name is None:                         # train hoặc category lạ
            continue
        if cfg.merge_rider_into_pedestrian and name == "rider":
            name = "pedestrian"
        x1 = max(0.0, min(W, box["x1"])); x2 = max(0.0, min(W, box["x2"]))
        y1 = max(0.0, min(H, box["y1"])); y2 = max(0.0, min(H, box["y2"]))
        bw, bh = x2 - x1, y2 - y1
        if bw < cfg.min_box_px or bh < cfg.min_box_px:
            continue
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        lines.append(f"{cls_id(name)} {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
    return lines
```

Yêu cầu:
- W, H lấy từ ảnh thật (`cv2.imread` hoặc đọc header bằng PIL) — mặc định 1280×720, **không hard-code**; cache kích thước nếu ảnh đồng nhất.
- Ảnh không có box nào → vẫn tạo file `.txt` rỗng (background image hợp lệ).
- Ảnh trong JSON mà không tồn tại trên đĩa → bỏ qua + ghi warning, đếm lại.
- Copy/hardlink ảnh vào `data/processed/detection/images/{split}/`:
  ```python
  try: os.link(src, dst)           # NTFS hardlink (cùng ổ đĩa)
  except OSError: shutil.copy2(src, dst)
  ```
- Chạy song song bằng `concurrent.futures.ThreadPoolExecutor` (I/O bound) với `tqdm`.
- Sinh `data/processed/detection/data.yaml`:
  ```yaml
  path: <abs path>/data/processed/detection
  train: images/train
  val: images/val
  names:
    0: car
    1: truck
    ...
  ```
  (đọc từ `DET_CLASSES`, đổi theo `merge_rider_into_pedestrian`).
- Ghi `data/processed/stats/det_{split}.json`: số ảnh, số box mỗi class, phân bố diện tích box (percentile), số ảnh rỗng, số ảnh theo `timeofday`/`weather`.

**Kiểm tra sau convert (`--validate`)**: duyệt mọi `.txt`: 5 cột, class ∈ [0, nc), mọi giá trị ∈ [0, 1], không trùng dòng. Vẽ 20 ảnh ngẫu nhiên có box vào `runs/inspect/det_samples/`.

## 3. Build segmentation masks: `scripts/build_bdd_seg.py`

Output: mỗi ảnh một PNG **uint8 một kênh**, cùng kích thước ảnh gốc:

| bit | Ý nghĩa |
|---|---|
| bit0 (1) | drivable_direct |
| bit1 (2) | drivable_alternative |
| bit2 (4) | lane_marking |

```python
def pack_mask(direct: np.ndarray, alt: np.ndarray, lane: np.ndarray) -> np.ndarray:
    out = (direct.astype(np.uint8) << 0) | (alt.astype(np.uint8) << 1) | (lane.astype(np.uint8) << 2)
    return out.astype(np.uint8)

def unpack_mask(packed: np.ndarray) -> np.ndarray:      # -> (3, H, W) uint8 {0,1}
    return np.stack([(packed >> i) & 1 for i in range(3)], axis=0).astype(np.uint8)
```

Quy trình:
1. Đọc drivable mask gốc → áp `value_map` trong config (xác minh ở bước inspect) → `direct`, `alt`.
2. Đọc lane mask gốc → áp rule trong config → `lane` (loại crosswalk/curb theo config).
3. Nếu thiếu mask cho một ảnh → bỏ ảnh khỏi tập seg (ghi `missing.txt`), **không** tạo mask rỗng giả.
4. Ghi PNG bằng `cv2.imwrite` (PNG lossless; không dùng JPEG cho mask).
5. Thống kê: tỉ lệ pixel trung bình mỗi kênh (để tính `pos_weight`), ghi `stats/seg_{split}.json`.

Nếu chỉ có JSON poly2d: dùng `python -m bdd100k.label.to_mask` (toolkit chính thức) sinh mask gốc trước, rồi chạy bước trên.

Test bắt buộc: `unpack_mask(pack_mask(a,b,c))` trả lại đúng a,b,c.

## 4. Subset: `scripts/make_subset.py`

```text
python scripts/make_subset.py --split train --n 10000 --seed 42 --stratify timeofday
```
- Đọc danh sách ảnh có **cả nhãn det lẫn seg** (giao của hai tập) để dùng chung subset.
- Stratify theo `timeofday` (tỉ lệ giữ nguyên); nếu thiếu thuộc tính → random thuần.
- Ghi `data/processed/subset_{split}_{n}_seed{seed}.txt` (mỗi dòng 1 tên ảnh).
- Các script convert nhận `--subset-file` để chỉ xử lý ảnh trong list.

## 5. Cityscapes: `scripts/convert_cityscapes.py`

```text
python scripts/convert_cityscapes.py --root data/raw/cityscapes --split val --out data/processed/cityscapes_det
```
- Duyệt `gtFine/{split}/*/*_gtFine_polygons.json`; ảnh tương ứng `leftImg8bit/{split}/{city}/{name}_leftImg8bit.png`.
- JSON có `imgWidth`, `imgHeight`, `objects: [{label, polygon: [[x,y],...]}]`.
- Với mỗi object có label trong `CITYSCAPES_DET_MAP`: bbox = min/max polygon, clip vào ảnh, chuẩn hoá theo `imgWidth/imgHeight`, bỏ box nhỏ.
- Bỏ label kết thúc bằng `group` hoặc không có trong map.
- Tuỳ chọn `--road-mask`: dùng `cityscapesscripts` tạo `labelTrainIds` rồi lưu `road` làm mask nhị phân cho đánh giá drivable.
- Sinh `data.yaml` riêng cho eval (`val: images/val`).

## 6. Dataloader segmentation: `segmentation/dataset.py`

```python
class BDDSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, size_hw=(384, 640), augment=False, subset_file=None): ...
    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        packed = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)          # (H, W) uint8
        mask = unpack_mask(packed).astype(np.float32)                  # (3, H, W)
        img, mask = resize_pair(img, mask, size_hw)                    # xem bên dưới
        if self.augment: img, mask = self.aug(img, mask)
        img = normalize_imagenet(img)                                  # float32, CHW
        return torch.from_numpy(img), torch.from_numpy(mask)
```

**Resize mask giữ vạch mỏng** (quan trọng):

```python
def resize_pair(img, mask3, size_hw, thr=(0.5, 0.5, 0.2)):
    H, W = size_hw
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    out = np.zeros((3, H, W), np.float32)
    for c in range(3):
        m = cv2.resize(mask3[c], (W, H), interpolation=cv2.INTER_AREA)  # float, mượt
        out[c] = (m > thr[c]).astype(np.float32)                        # lane: ngưỡng thấp để giữ vạch
    return img, out
```

**Augmentation** (albumentations, áp dụng cùng lúc ảnh+mask; mask 3 kênh truyền dạng `HWC`, chuyển trước/sau):
- `HorizontalFlip(p=0.5)` — an toàn cho 3 kênh (không dùng nhãn trái/phải).
- `RandomBrightnessContrast(0.3, 0.3)`, `HueSaturationValue(10, 20, 20)`.
- `RandomGamma`, `GaussNoise`, `MotionBlur(p=0.1)`, `RandomRain/RandomFog(p=0.05)` (mô phỏng thời tiết).
- `ShiftScaleRotate(shift=0.05, scale=0.1, rotate=5, p=0.5)` — **không xoay mạnh** vì làm sai hình học làn.
- **Không** dùng `VerticalFlip`, `Cutout` lớn che mất làn.

## 7. Kiểm tra chất lượng dữ liệu (checklist)
- [ ] Số ảnh det ↔ số file txt khớp.
- [ ] Số ảnh seg ↔ số mask khớp.
- [ ] Phân bố class không có class = 0 box ở val.
- [ ] Overlay mẫu: box bám đối tượng, mask bám làn.
- [ ] Có ảnh ban đêm / mưa trong subset.
- [ ] Hold-out `qual_set.txt` đã tạo và không nằm trong train.
