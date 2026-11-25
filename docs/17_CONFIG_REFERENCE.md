# 17 — Tham chiếu cấu hình (`configs/default.yaml`)

> Đây là file cấu hình mặc định. `config.py` phải load thành dataclass lồng nhau (truy cập `cfg.adas.ldw.warn_offset`), validate kiểu + khoảng giá trị, hỗ trợ override bằng file thứ hai (`colab.yaml`) và `--set key=value` trên CLI.
> Các giá trị mang tính **khởi điểm**, cần tinh chỉnh bằng thực nghiệm.

```yaml
project:
  name: adas-vision
  seed: 42

paths:
  bdd:
    root: data/raw/bdd100k
    images: "{root}/images/100k"            # [VERIFY] xác nhận bằng inspect
    det_labels:
      train: "{root}/labels/det_20/det_train.json"
      val:   "{root}/labels/det_20/det_val.json"
    drivable_masks: "{root}/labels/drivable/masks"   # [VERIFY]
    lane_masks:     "{root}/labels/lane/masks"       # [VERIFY]
  cityscapes:
    root: data/raw/cityscapes
  processed_root: data/processed
  runs_root: runs
  weights_root: weights

subset:
  train: 10000
  val: 2000
  stratify: timeofday          # timeofday | weather | none
  boost_classes: []            # vd [motorcycle, rider, bus]
  qual_set_size: 200

detection:
  classes: [car, truck, bus, motorcycle, bicycle, pedestrian, rider, traffic_light, traffic_sign]
  merge_rider_into_pedestrian: false
  min_box_px: 4
  split_traffic_light_color: false
  use_cityscapes: false
  model: yolo11s.pt
  weights: weights/detector_best.pt
  imgsz: 960
  conf: 0.25
  iou: 0.6
  half: true
  tracker: bytetrack.yaml
  conf_per_class:              # lọc sau predict (model dùng conf thấp nhất)
    traffic_sign: 0.35
    pedestrian: 0.25
  train:
    epochs: 50
    batch: 16
    workers: 4
    patience: 15
    cos_lr: true
    close_mosaic: 10
    cache: false
    project: runs/detect
    name: yolo11s_bdd_960
  traffic_light:
    min_ratio: 0.02
    vote_frames: 5

segmentation:
  arch: DeepLabV3Plus          # DeepLabV3Plus | Unet | FPN
  encoder: resnet34
  encoder_weights: imagenet
  size_hw: [384, 640]
  weights: weights/segmenter_best.pt
  half: true
  thresholds: [0.5, 0.5, 0.5]
  drivable:
    value_map: {direct: 0, alternative: 1, ignore: 255}   # [VERIFY] bằng inspect_bdd.py
  lane:
    include_crosswalk: false
    include_curb: false
    # rule trích pixel vạch kẻ từ mask gốc: điền sau khi inspect (không đoán bit)
  resize_thr: [0.5, 0.5, 0.2]
  loss:
    pos_weight: [1.0, 1.0, 3.0]
    w_bce: 1.0
    w_dice: 1.0
  train:
    epochs: 40
    batch: 16
    lr: 3.0e-4
    weight_decay: 1.0e-4
    scheduler: onecycle
    amp: true
    grad_clip: 1.0
    num_workers: 4
    early_stop_patience: 8
    freeze_encoder_epochs: 0
    name: deeplab_r34_384x640
  postprocess:
    min_area: [500, 500, 20]   # theo kênh
    temporal_ema: 0.0          # 0 = tắt

adas:
  camera_offset_ratio: 0.0     # lệch camera so với tâm xe (tỉ lệ chiều rộng ảnh)
  ego_dilate_px: 6
  fallback_corridor:           # hình thang dùng khi mất mask làn
    enabled: true
    top_y: 0.55
    top_half_w: 0.08
    bottom_half_w: 0.30
  lane_geometry:
    band: [0.70, 0.90]
    row_step: 2
    min_row_pixels_ratio: 0.05
    min_rows: 5
    min_width_ratio: 0.15
    max_width_ratio: 0.90
  ldw:
    warn_offset: 0.25
    critical_offset: 0.38
    exit_factor: 0.8
    min_frames: 5
    ema_alpha: 0.4
    invalid_grace_frames: 10
  fcw:
    min_conf: 0.4
    min_track_age: 3
    track_ttl: 15
    ttc_window: 8
    warn_proximity: 0.78
    crit_proximity: 0.88
    warn_ttc: 3.0
    crit_ttc: 1.5
    lane_margin_px: 20
  pedestrian:
    min_conf: 0.35
    ped_dilate_px: 12
    warn_proximity: 0.70
    crit_proximity: 0.82
    min_frames: 3
  traffic:
    drive_on_right: true
    red_light:
      min_conf: 0.4
      rel_x: 0.3
      max_cy_ratio: 0.6
      min_h_px: 12
    sign:
      min_conf: 0.5
      min_h_px: 16
      region_x: [0.35, 1.0]
  alerts:
    cooldown_s:
      LANE_DEPARTURE: 2.0
      FORWARD_COLLISION: 1.0
      PEDESTRIAN_IN_PATH: 1.5
      TRAFFIC_LIGHT_RED: 3.0
      TRAFFIC_SIGN_AHEAD: 5.0
    display_s: 1.5

pipeline:
  seg_every_n_frames: 1
  device: 0
  max_frames: null

render:
  debug: false
  font_path: assets/fonts/NotoSans-Regular.ttf
  mask_alpha: {direct: 0.35, alternative: 0.25, lane: 0.8}
  show_fps: true

io:
  video_codec: mp4v
```

## Ghi chú các key quan trọng

| Key | Ý nghĩa / ảnh hưởng |
|---|---|
| `paths.bdd.*` | Mọi đường dẫn dataset; script chỉ dùng qua config. Placeholder `{root}` được thay khi load. |
| `detection.classes` | Thứ tự **phải** bằng `constants.DET_CLASSES` (config validate). |
| `detection.imgsz` | 960 mặc định; 640 nhanh hơn, 1280 tốt hơn cho vật nhỏ. |
| `segmentation.size_hw` | Phải chia hết cho 32. Train và inference **dùng chung**. |
| `segmentation.drivable.value_map` | **Điền sau khi inspect** — đây là điểm dễ sai nhất. |
| `segmentation.resize_thr` | Ngưỡng sau INTER_AREA từng kênh; lane thấp (0.2) để giữ vạch mỏng. |
| `adas.camera_offset_ratio` | >0: camera lệch phải tâm xe → bù để `offset` đúng. |
| `adas.lane_geometry.band` | Dải hàng (tỉ lệ chiều cao) đo biên làn; gần xe hơn → ổn định hơn nhưng ít "nhìn xa". |
| `adas.ldw.warn_offset` | Tăng → ít báo giả, chậm cảnh báo. |
| `adas.fcw.*_proximity` | Dựa vị trí chân bbox/H; phụ thuộc góc camera → hiệu chỉnh theo clip thật. |
| `adas.fcw.*_ttc` | Giây, tính từ tốc độ phình bbox; nhiễu → dùng `ttc_window` ≥ 6. |
| `pipeline.seg_every_n_frames` | 2–3 để tăng FPS nếu cần. |

## Override ví dụ (`configs/colab.yaml`)
```yaml
paths:
  processed_root: /content/data/processed
  runs_root: /content/drive/MyDrive/adas/runs
detection:
  train: {workers: 2, batch: 16}
segmentation:
  train: {num_workers: 2}
```

## Validate bắt buộc trong `config.py`
- `detection.classes == constants.DET_CLASSES` (hoặc dạng đã merge rider nếu bật cờ).
- `segmentation.size_hw` chia hết cho 32.
- `0 < ldw.warn_offset < ldw.critical_offset < 0.6`.
- `fcw.crit_ttc < fcw.warn_ttc`; `fcw.warn_proximity < fcw.crit_proximity ≤ 1`.
- `thresholds` có đúng 3 phần tử ∈ (0,1).
