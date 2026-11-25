"""Data Layer (TV2): đọc BDD100K / Cityscapes, chuyển nhãn, augmentation & preprocess.

Module:
  * ``converters``         — Detection JSON -> YOLO; mask BDD -> mask 3 kênh
  * ``bdd_layout``         — đọc configs/paths.yaml, dò đường dẫn bên trong BDD100K
  * ``inspection``         — kiểm tra dữ liệu thực tế (dùng bởi scripts/inspect_bdd.py)
  * ``splits``             — subset cố định seed=42 (data/splits/*.txt)
  * ``prepare``            — chạy chuyển đổi (dùng bởi scripts/prepare_data.py)
  * ``transforms``         — Albumentations train / eval; ``preprocess_inference`` = eval
  * ``bdd_dataset``        — DetectionDataset, SegmentationDataset, build_dataloader
  * ``cityscapes_dataset`` — CityscapesRoadDataset, RoadIoU (road = direct OR alternative)
"""
from .bdd_dataset import DetectionDataset, SegmentationDataset, build_dataloader
from .cityscapes_dataset import CityscapesRoadDataset, RoadIoU, road_from_prediction
from .transforms import (
    DEFAULT_INPUT_SIZE,
    build_eval_transforms,
    build_train_transforms,
    denormalize,
    preprocess_inference,
)

__all__ = [
    "DEFAULT_INPUT_SIZE",
    "build_train_transforms",
    "build_eval_transforms",
    "preprocess_inference",
    "denormalize",
    "DetectionDataset",
    "SegmentationDataset",
    "build_dataloader",
    "CityscapesRoadDataset",
    "RoadIoU",
    "road_from_prediction",
]
