# 03 — Datasets

> ⚠️ Tên gói tải, cấu trúc thư mục và định dạng mask có thể thay đổi theo phiên bản. **Luôn chạy script inspect (`scripts/inspect_bdd.py`) và đối chiếu tài liệu chính thức trước khi tin vào giả định** trong file này. Các mục đánh dấu **[VERIFY]** là chỗ phải kiểm tra.

## 1. BDD100K (dataset chính)

- Nguồn: trang chính thức BDD100K (Berkeley DeepDrive), cần đăng ký để tải. Toolkit: repo `bdd100k` trên GitHub (có `bdd100k.label.to_mask`, `bdd100k.eval`...).
- 100.000 video clip, mỗi clip có 1 keyframe (ảnh JPG **1280×720**) → "100K Images": ~70K train / 10K val / 20K test (test không có nhãn công khai).
- Cùng bộ ảnh này có nhãn cho: **object detection (2D box)**, **drivable area**, **lane marking**. (Bộ "10K Images" có semantic/instance segmentation kiểu Cityscapes, **không dùng** trong dự án này trừ khi mở rộng.)
- Thuộc tính ảnh: `weather` (clear, overcast, rainy, snowy, foggy...), `scene` (city street, highway, residential...), `timeofday` (daytime, night, dawn/dusk). Dùng để **phân tích lỗi theo điều kiện**.

### 1.1 Gói cần tải [VERIFY tên gói trên trang tải]
1. `100K Images` (train/val/test).
2. Nhãn detection (bản "Detection 2020 Labels" `det_train.json`, `det_val.json` hoặc bản cũ `bdd100k_labels_images_*.json`).
3. `Drivable Area` (masks/ids hoặc nhãn JSON).
4. `Lane Markings` (masks hoặc nhãn JSON).

Nếu chỉ có nhãn JSON (poly2d) mà không có mask, dùng toolkit chính thức để render, **không tự vẽ bezier**:
```bash
python -m bdd100k.label.to_mask --help   # xem mode: drivable / lane_mark ...
```

### 1.2 Định dạng nhãn detection (JSON)
```json
{
  "name": "0000f77c-6257be58.jpg",
  "attributes": {"weather": "clear", "scene": "city street", "timeofday": "daytime"},
  "labels": [
    {"id": "0", "category": "traffic sign",
     "attributes": {"occluded": false, "truncated": false, "trafficLightColor": "none"},
     "box2d": {"x1": 1000.5, "y1": 281.9, "x2": 1054.8, "y2": 314.6}}
  ]
}
```
Chỉ các label có `box2d` mới dùng cho detection. Category gốc: `pedestrian, rider, car, truck, bus, train, motor, bike, traffic light, traffic sign`.

### 1.3 Mapping class detection (BDD → dự án)

| BDD category | Project class | idx |
|---|---|---|
| car | car | 0 |
| truck | truck | 1 |
| bus | bus | 2 |
| motor | motorcycle | 3 |
| bike | bicycle | 4 |
| pedestrian | pedestrian | 5 |
| rider | rider | 6 |
| traffic light | traffic_light | 7 |
| traffic sign | traffic_sign | 8 |
| train | **bỏ** | — |

Tuỳ chọn trong config: `detection.merge_rider_into_pedestrian: false` (nếu true → rider gộp vào pedestrian, còn 8 class và đánh lại index; **chỉ đổi qua config**, code sinh `data.yaml` tự động).

Tuỳ chọn `detection.min_box_px: 4` — bỏ box nhỏ hơn 4 px mỗi cạnh (nhiễu nhãn). `detection.ignore_occluded: false`, `ignore_truncated: false` (giữ hết mặc định).

### 1.4 Drivable area
- 3 loại vùng: **direct drivable** (làn xe đang chạy — ego lane), **alternative drivable** (làn khác có thể chuyển sang), background.
- Mask dạng id PNG: giá trị phân biệt direct / alternative / background (có thể có giá trị ignore như 255). **[VERIFY]** bằng `inspect_bdd.py` — in `np.unique` + tỉ lệ pixel + ảnh overlay.
- Mapping sang kênh dự án:
  - kênh 0 `drivable_direct` = pixel direct
  - kênh 1 `drivable_alternative` = pixel alternative

### 1.5 Lane marking
- Nhãn vạch kẻ: category kiểu `lane/single white`, `lane/double yellow`, `lane/road curb`, `lane/crosswalk`..., kèm thuộc tính **direction** (parallel/vertical) và **style** (solid/dashed).
- Mask lane có thể mã hoá category + thuộc tính trong các bit của pixel **[VERIFY cách mã hoá bằng inspect, KHÔNG đoán bit]**.
- Mặc định dự án: kênh 2 `lane_marking` = **mọi pixel thuộc vạch kẻ** (nhị phân), gồm cả `road curb`? → mặc định **loại `crosswalk` và `road curb`**, chỉ giữ vạch kẻ làn: `single/double × white/yellow/other`. Điều khiển bằng `segmentation.lane.include_crosswalk` và `include_curb`.
- Mở rộng (không bắt buộc): tách thêm kênh white/yellow, solid/dashed để cảnh báo "vượt vạch liền".

### 1.6 Cấu trúc thư mục mong đợi sau khi giải nén [VERIFY]
```text
data/raw/bdd100k/
├── images/100k/{train,val,test}/*.jpg
├── labels/
│   ├── det_20/det_train.json, det_val.json
│   ├── lane/{masks,colormaps,polygons...}/{train,val}/
│   └── drivable/{masks,colormaps,polygons...}/{train,val}/
```
Đường dẫn thực tế đặt trong `configs/default.yaml` (`paths.bdd.*`), **script không được hard-code**.

### 1.7 Split
- Dùng split chính thức: train 70K, val 10K. Test không nhãn → **không dùng**.
- Khi dùng subset (vì GPU): lấy ngẫu nhiên có seed từ train/val (`subset.train=10000`, `subset.val=2000`), **stratify theo `timeofday`** nếu có thể để giữ ảnh đêm.
- Hold-out riêng ~200 ảnh val làm **qualitative set** (cố định, lưu `data/processed/qual_set.txt`) để so sánh các lần train.

## 2. Cityscapes (dataset phụ)

- 5.000 ảnh fine annotation (2975 train / 500 val / 1525 test — test không công khai nhãn), ảnh **2048×1024**, 50 thành phố Đức.
- Cần đăng ký tải: `leftImg8bit_trainvaltest.zip` và `gtFine_trainvaltest.zip`. Cài `cityscapesscripts`.
- 19 lớp train (trainId): road 0, sidewalk 1, building 2, wall 3, fence 4, pole 5, traffic light 6, traffic sign 7, vegetation 8, terrain 9, sky 10, person 11, rider 12, car 13, truck 14, bus 15, train 16, motorcycle 17, bicycle 18.
- **Không có nhãn vạch kẻ đường** → không dùng cho kênh `lane_marking`.

### 2.1 Vai trò trong dự án
| Vai trò | Cách làm |
|---|---|
| (a) **Đánh giá cross-dataset** detection | Convert `gtFine/val` → YOLO box, eval detector train trên BDD → đo domain gap |
| (b) **Đánh giá cross-dataset** drivable | Gộp `road` (trainId 0) làm "drivable" → so IoU với kênh `drivable_direct ∪ drivable_alternative` (lưu ý: định nghĩa khác nhau, chỉ để tham khảo) |
| (c) Tuỳ chọn trộn vào train detection | Convert `train` sang YOLO, cờ `detection.use_cityscapes: false` mặc định |

### 2.2 Tạo box từ Cityscapes
- Dùng `*_gtFine_polygons.json`: mỗi object có `label` và `polygon` → bbox = min/max toạ độ polygon.
- Lý do không dùng `instanceIds.png`: `traffic sign`/`traffic light` **không có instance** trong Cityscapes; polygons JSON thì có đủ.
- Mapping:

| Cityscapes label | Project class |
|---|---|
| car | car |
| truck | truck |
| bus | bus |
| motorcycle | motorcycle |
| bicycle | bicycle |
| person | pedestrian |
| rider | rider |
| traffic light | traffic_light |
| traffic sign | traffic_sign |
| train, caravan, trailer, *group, ego vehicle, out of roi... | **bỏ** |

- Ảnh 2048×1024 → giữ nguyên khi eval (YOLO tự letterbox) hoặc resize về 1280×640; **toạ độ chuẩn hoá theo kích thước ảnh thực dùng**.
- Bỏ box có cạnh < `min_box_px` sau chuẩn hoá theo kích thước ảnh eval.

## 3. Domain gap & dữ liệu Việt Nam
- BDD100K / Cityscapes: đường Mỹ/Đức, ít xe máy dày đặc, vạch kẻ khác Việt Nam.
- Xe máy trong BDD (`motor`) rất ít so với thực tế VN → AP motorcycle thường thấp.
- Phần mở rộng: tự quay dashcam, gán nhãn ~300–500 ảnh (Roboflow / CVAT / Label Studio), fine-tune thêm. Đưa vào `data/raw/custom_vn/` với cùng mapping class.

## 4. Giấy phép
- BDD100K và Cityscapes có điều khoản riêng (chủ yếu phi thương mại/nghiên cứu). Không redistribute dữ liệu. Ghi trích dẫn trong báo cáo.
- Ultralytics YOLO: AGPL-3.0 (ghi chú trong README nếu công bố).

## 5. Trích dẫn
- Yu et al., "BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning", CVPR 2020.
- Cordts et al., "The Cityscapes Dataset for Semantic Urban Scene Understanding", CVPR 2016.
