"""Unit tests for Milestone 2: data converters, transforms, datasets, and inspection."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from data.bdd_dataset import BDDSegmentationDataset
from data.cityscapes_dataset import CityscapesRoadDataset
from data.converters import (
    convert_bdd_frame_labels_to_yolo,
    convert_bdd_json_to_yolo_dir,
    create_subset_split,
    encode_bdd_masks_to_3ch,
    extract_cityscapes_road_mask,
    read_split_file,
    write_split_file,
)
from data.transforms import (
    get_seg_train_transforms,
    preprocess_for_segmentation,
)
from scripts.inspect_bdd import run_inspection
from utils.constants import DET_CLASS_TO_ID, SEG_CHANNELS


def test_convert_bdd_frame_labels_to_yolo() -> None:
    labels = [
        {"category": "car", "box2d": {"x1": 100.0, "y1": 100.0, "x2": 300.0, "y2": 200.0}},
        {"category": "motor", "box2d": {"x1": -20.0, "y1": 0.0, "x2": 100.0, "y2": 100.0}},
        {"category": "traffic sign", "box2d": {"x1": 500.0, "y1": 50.0, "x2": 540.0, "y2": 90.0}},
        {"category": "train", "box2d": {"x1": 10.0, "y1": 10.0, "x2": 50.0, "y2": 50.0}},
        {"category": "car", "box2d": {"x1": 10.0, "y1": 10.0, "x2": 10.0, "y2": 50.0}},
        {"category": "person"},  # missing box2d
    ]
    lines = convert_bdd_frame_labels_to_yolo(labels, img_width=1000, img_height=500)
    assert len(lines) == 3

    cls0, cx0, cy0, w0, h0 = (float(v) for v in lines[0].split())
    assert int(cls0) == DET_CLASS_TO_ID["car"]
    assert (cx0, cy0, w0, h0) == pytest.approx((0.2, 0.3, 0.2, 0.2), abs=1e-5)

    cls1, cx1, cy1, w1, h1 = (float(v) for v in lines[1].split())
    assert int(cls1) == DET_CLASS_TO_ID["motorcycle"]
    assert (cx1, cy1, w1, h1) == pytest.approx((0.05, 0.1, 0.1, 0.2), abs=1e-5)

    cls2 = int(lines[2].split()[0])
    assert cls2 == DET_CLASS_TO_ID["traffic_sign"]


def test_convert_bdd_json_to_yolo_dir(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    lbl_out = tmp_path / "labels"
    img_dir.mkdir()

    # Create one synthetic 200x100 image
    dummy_img = np.zeros((100, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "frame_001.jpg"), dummy_img)

    json_path = tmp_path / "det_val.json"
    payload = [
        {
            "name": "frame_001.jpg",
            "labels": [
                {"category": "person", "box2d": {"x1": 20, "y1": 10, "x2": 60, "y2": 90}}
            ],
        },
        {
            "name": "missing_frame.jpg",
            "labels": [{"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 50, "y2": 50}}],
        },
    ]
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    stats = convert_bdd_json_to_yolo_dir(json_path, img_dir, lbl_out)
    assert stats == {"frames_written": 1, "boxes_written": 1, "skipped_missing_images": 1}

    txt_content = (lbl_out / "frame_001.txt").read_text(encoding="utf-8").strip()
    parts = [float(x) for x in txt_content.split()]
    assert int(parts[0]) == DET_CLASS_TO_ID["pedestrian"]
    assert parts[1:] == pytest.approx([0.2, 0.5, 0.2, 0.8], abs=1e-5)


def test_encode_bdd_masks_to_3ch_and_cityscapes_road() -> None:
    drivable = np.full((16, 16), 2, dtype=np.uint8)
    drivable[8:, 4:12] = 0   # direct
    drivable[8:, 0:4] = 1    # alternative

    lane = np.full((16, 16), 255, dtype=np.uint8)
    lane[8:, 4] = 1          # lane marking

    encoded = encode_bdd_masks_to_3ch(drivable, lane)
    assert encoded.shape == (len(SEG_CHANNELS), 16, 16)
    assert encoded.dtype == np.uint8
    assert encoded[0, 10, 8] == 1
    assert encoded[1, 10, 2] == 1
    assert encoded[2, 10, 4] == 1
    assert encoded[2, 0, 0] == 0

    cs_mask = np.zeros((16, 16), dtype=np.uint8)
    cs_mask[10:, :] = 7
    road = extract_cityscapes_road_mask(cs_mask, road_id=7)
    assert road.shape == (16, 16)
    assert int(road.sum()) == 6 * 16


def test_subset_split_determinism_and_io(tmp_path: Path) -> None:
    stems = [f"img_{i:03d}" for i in range(20)]
    split_a = create_subset_split(stems, max_samples=5, seed=42)
    split_b = create_subset_split(list(reversed(stems)), max_samples=5, seed=42)
    assert split_a == split_b
    assert len(split_a) == 5

    split_file = write_split_file(split_a, tmp_path / "val.txt")
    assert read_split_file(split_file) == split_a


def test_segmentation_preprocess_and_datasets(tmp_path: Path) -> None:
    # Synthetic BGR frame -> inference tensor
    frame_bgr = np.full((72, 128, 3), 128, dtype=np.uint8)
    tensor = preprocess_for_segmentation(frame_bgr, size_hw=(32, 64))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 32, 64)
    assert tensor.dtype == torch.float32

    # Synthetic BDD dataset directory
    img_dir = tmp_path / "bdd" / "images"
    drv_dir = tmp_path / "bdd" / "drivable"
    lane_dir = tmp_path / "bdd" / "lane"
    for d in (img_dir, drv_dir, lane_dir):
        d.mkdir(parents=True)

    cv2.imwrite(str(img_dir / "sample1.jpg"), frame_bgr)
    cv2.imwrite(str(drv_dir / "sample1.png"), np.zeros((72, 128), dtype=np.uint8))
    cv2.imwrite(str(lane_dir / "sample1.png"), np.full((72, 128), 255, dtype=np.uint8))

    ds = BDDSegmentationDataset(
        images_dir=img_dir,
        drivable_dir=drv_dir,
        lane_dir=lane_dir,
        size_hw=(32, 64),
        transform=get_seg_train_transforms((32, 64)),
    )
    assert len(ds) == 1
    img_t, mask_t = ds[0]
    assert img_t.shape == (3, 32, 64)
    assert mask_t.shape == (3, 32, 64)

    # Synthetic Cityscapes dataset directory
    cs_img = tmp_path / "cs" / "leftImg8bit" / "val" / "frankfurt"
    cs_gt = tmp_path / "cs" / "gtFine" / "val" / "frankfurt"
    cs_img.mkdir(parents=True)
    cs_gt.mkdir(parents=True)
    cv2.imwrite(str(cs_img / "frankfurt_000001_leftImg8bit.png"), frame_bgr)
    cs_gt_mask = np.full((72, 128), 7, dtype=np.uint8)
    cv2.imwrite(str(cs_gt / "frankfurt_000001_gtFine_labelIds.png"), cs_gt_mask)

    cs_ds = CityscapesRoadDataset(tmp_path / "cs", split="val", size_hw=(32, 64))
    assert len(cs_ds) == 1
    c_img_t, c_mask_t = cs_ds[0]
    assert c_img_t.shape == (3, 32, 64)
    assert c_mask_t.shape == (32, 64)


def test_inspect_bdd_on_synthetic_root(tmp_path: Path) -> None:
    bdd_root = tmp_path / "bdd100k"
    img_val = bdd_root / "images" / "100k" / "val"
    drv_val = bdd_root / "drivable" / "val"
    lbl_dir = bdd_root / "labels"
    for d in (img_val, drv_val, lbl_dir):
        d.mkdir(parents=True)

    cv2.imwrite(str(img_val / "a.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
    cv2.imwrite(str(drv_val / "a.png"), np.array([[0, 1], [2, 2]], dtype=np.uint8))
    sample_json = [
        {
            "name": "a.jpg",
            "attributes": {"weather": "clear"},
            "labels": [{"category": "car"}],
        }
    ]
    (lbl_dir / "det_val.json").write_text(json.dumps(sample_json), encoding="utf-8")

    report = run_inspection(bdd_root=bdd_root, sample_masks=5, max_json_frames=10)
    assert report["image_counts"]["images/100k/val"] == 1
    assert report["json_reports"][0]["categories"]["car"] == 1
    assert report["mask_reports"][0]["unique_gray_values"] == [0, 1, 2]
