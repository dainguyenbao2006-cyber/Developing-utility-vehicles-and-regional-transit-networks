# 05 — Cấu trúc dự án

```text
adas-vision/
├── CLAUDE.md                     # copy từ docs/14_AGENT_RULES.md
├── README.md
├── pyproject.toml                # package adas, dependencies, tool config (ruff, pytest)
├── requirements.txt
├── .gitignore
├── configs/
│   ├── default.yaml              # xem docs/17
│   ├── colab.yaml                # override cho Colab (đường dẫn, batch)
│   └── eval_cityscapes.yaml
├── docs/                         # bộ tài liệu này
├── data/                         # KHÔNG commit
│   ├── raw/
│   │   ├── bdd100k/
│   │   └── cityscapes/
│   └── processed/
│       ├── detection/
│       │   ├── images/{train,val}/       # hardlink/copy ảnh
│       │   ├── labels/{train,val}/       # YOLO txt
│       │   └── data.yaml                 # sinh tự động
│       ├── segmentation/
│       │   ├── images/{train,val}/
│       │   └── masks/{train,val}/        # PNG uint8 bitmask (bit0 direct, bit1 alt, bit2 lane)
│       ├── cityscapes_det/{images,labels}/val
│       ├── qual_set.txt
│       └── stats/*.json
├── weights/                      # KHÔNG commit (trừ file nhỏ nếu muốn)
├── runs/                         # log, checkpoint, kết quả eval
├── notebooks/
│   ├── 01_eda_bdd100k.ipynb
│   └── colab_train.ipynb
├── scripts/
│   ├── inspect_bdd.py            # in np.unique mask, vẽ overlay mẫu
│   ├── convert_bdd_det.py        # BDD json → YOLO txt + data.yaml
│   ├── build_bdd_seg.py          # tạo bitmask seg
│   ├── make_subset.py
│   ├── convert_cityscapes.py
│   ├── train_detector.py         # wrapper Ultralytics
│   ├── train_segmenter.py
│   ├── eval_detector.py
│   ├── eval_segmenter.py
│   └── export_onnx.py            # tuỳ chọn
├── src/adas/
│   ├── __init__.py
│   ├── types.py
│   ├── config.py
│   ├── constants.py              # DET_CLASSES, SEG_CHANNELS, nhóm VEHICLES/VRU
│   ├── io.py
│   ├── render.py
│   ├── pipeline.py
│   ├── cli.py
│   ├── detection/{__init__,detector,traffic_light}.py
│   ├── segmentation/{__init__,dataset,model,losses,metrics,segmenter,postprocess}.py
│   ├── adas/{__init__,lane_geometry,ldw,fcw,pedestrian,traffic,alerts,engine}.py
│   └── data/{__init__,bdd,cityscapes,masks,subset}.py   # hàm dùng chung cho scripts
├── app/
│   └── streamlit_app.py
└── tests/
    ├── conftest.py
    ├── test_constants.py
    ├── test_data_bdd.py
    ├── test_masks.py
    ├── test_lane_geometry.py
    ├── test_ldw.py
    ├── test_fcw.py
    ├── test_pedestrian.py
    ├── test_traffic.py
    ├── test_alerts.py
    ├── test_traffic_light_color.py
    └── test_pipeline_smoke.py
```

## `constants.py` (nội dung bắt buộc)

```python
DET_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle",
               "pedestrian", "rider", "traffic_light", "traffic_sign"]
DET_ID = {n: i for i, n in enumerate(DET_CLASSES)}

VEHICLES = {"car", "truck", "bus", "motorcycle"}
VRU = {"pedestrian", "rider", "bicycle"}          # vulnerable road users

SEG_CHANNELS = ["drivable_direct", "drivable_alternative", "lane_marking"]
SEG_ID = {n: i for i, n in enumerate(SEG_CHANNELS)}

BDD_DET_MAP = {"car": "car", "truck": "truck", "bus": "bus", "motor": "motorcycle",
               "bike": "bicycle", "pedestrian": "pedestrian", "rider": "rider",
               "traffic light": "traffic_light", "traffic sign": "traffic_sign"}
CITYSCAPES_DET_MAP = {"car": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle",
                      "bicycle": "bicycle", "person": "pedestrian", "rider": "rider",
                      "traffic light": "traffic_light", "traffic sign": "traffic_sign"}
```

## Quy ước đặt tên
- Module/hàm: `snake_case`; class: `PascalCase`; hằng: `UPPER_SNAKE`.
- Script trong `scripts/` chỉ là lớp mỏng gọi hàm trong `src/adas/data/...` (để test được).
- File processed: giữ nguyên tên ảnh gốc BDD (vd `0000f77c-6257be58.jpg` → `.txt`, `.png`).

## `.gitignore` tối thiểu
```text
data/
weights/
runs/
__pycache__/
*.pt
*.pth
*.onnx
.venv/
.ipynb_checkpoints/
*.mp4
```
