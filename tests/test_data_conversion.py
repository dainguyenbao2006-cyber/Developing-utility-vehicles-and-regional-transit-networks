"""Unit test cho module dữ liệu (TV2). Chạy: ``pytest tests/test_data_conversion.py -q``.

Không cần tải BDD100K: các test tự sinh bộ dữ liệu giả lập nhỏ theo đúng layout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import splits as split_lib  # noqa: E402
from src.data import transforms as T  # noqa: E402
from src.data.bdd_dataset import DetectionDataset, SegmentationDataset  # noqa: E402
from src.data.cityscapes_dataset import (  # noqa: E402
    CityscapesRoadDataset,
    RoadIoU,
    labelids_to_road,
    road_from_prediction,
)
from src.data.converters import (  # noqa: E402
    CH_ALTERNATIVE,
    CH_DRIVABLE,
    CH_LANE,
    DrivableIds,
    YoloBox,
    box2d_to_yolo,
    build_mask,
    frame_to_yolo,
    load_mask,
    parse_yolo_line,
    save_mask,
    yolo_lines,
    yolo_to_xyxy,
)
from src.data.inspection import inspect_split  # noqa: E402
from src.utils.constants import DET_CLASS_TO_ID, DET_CLASSES, NUM_DET_CLASSES, SEG_CHANNELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ["cv2"] + (["albumentations"] if T.HAS_ALBUMENTATIONS else [])
IGNORE = 255  # giá trị lạ trong mask để kiểm tra resize nearest không sinh nhãn mới
W, H = 1280, 720


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------
def test_seg_channel_names():
    assert SEG_CHANNELS == ("drivable_direct", "drivable_alternative", "lane_marking")


def test_nine_classes_match_spec_order():
    assert DET_CLASSES == ("car", "truck", "bus", "motorcycle", "bicycle",
                           "pedestrian", "rider", "traffic_light", "traffic_sign")
    assert NUM_DET_CLASSES == 9 and "train" not in DET_CLASSES


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def test_box2d_to_yolo_basic():
    yb = box2d_to_yolo({"x1": 100, "y1": 200, "x2": 300, "y2": 400}, 2, W, H)
    assert yb == YoloBox(2, 200 / W, 300 / H, 200 / W, 200 / H)


def test_box2d_clipped_and_swapped():
    yb = box2d_to_yolo({"x1": 1300, "y1": 700, "x2": 1200, "y2": 800}, 0, W, H)
    assert yb is not None
    x1, y1, x2, y2 = yolo_to_xyxy(yb, W, H)
    assert (x1, y1, x2, y2) == pytest.approx((1200, 700, 1280, 720))


def test_degenerate_box_dropped():
    assert box2d_to_yolo({"x1": 10, "y1": 10, "x2": 10, "y2": 50}, 0, W, H) is None  # width = 0
    assert box2d_to_yolo({"x1": -50, "y1": 10, "x2": -10, "y2": 50}, 0, W, H) is None  # ngoài ảnh
    assert box2d_to_yolo({"x1": 10, "y1": 10, "x2": 11, "y2": 50}, 0, W, H) is not None  # 1px vẫn giữ


def test_frame_to_yolo_filters_classes_and_aliases():
    frame = {
        "name": "a.jpg",
        "labels": [
            {"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 64, "y2": 64}},
            {"category": "train", "box2d": {"x1": 0, "y1": 0, "x2": 64, "y2": 64}},  # bị bỏ
            {"category": "person", "box2d": {"x1": 0, "y1": 0, "x2": 64, "y2": 64}},  # alias
            {"category": "traffic light", "box2d": {"x1": 0, "y1": 0, "x2": 64, "y2": 64}},
            {"category": "motor", "box2d": {"x1": 0, "y1": 0, "x2": 64, "y2": 64}},
            {"category": "other vehicle", "box2d": {"x1": 0, "y1": 0, "x2": 64, "y2": 64}},  # bị bỏ
            {"category": "lane/single white", "poly2d": [{"vertices": [[0, 0], [1, 1]]}]},  # không có box
        ],
    }
    boxes = frame_to_yolo(frame)
    assert [b.class_id for b in boxes] == [0, 5, 7, 3]  # car, pedestrian, traffic_light, motorcycle


def test_yolo_roundtrip():
    boxes = [YoloBox(1, 0.5, 0.5, 0.25, 0.1), YoloBox(8, 0.1, 0.9, 0.05, 0.05)]
    text = yolo_lines(boxes)
    parsed = [parse_yolo_line(ln) for ln in text.splitlines()]
    for a, b in zip(boxes, parsed):
        assert a.class_id == b.class_id
        assert (a.xc, a.yc, a.w, a.h) == pytest.approx((b.xc, b.yc, b.w, b.h), abs=1e-6)
    assert yolo_lines([]) == ""


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
def test_mask_from_drivable_png_and_lanes():
    drv = np.full((H, W), 2, np.uint8)
    drv[500:, 400:880] = 0  # direct
    drv[500:, 100:400] = 1  # alternative
    lane = {"labels": [{"category": "lane/single white", "poly2d": [{"vertices": [[640, 719], [640, 400]], "types": "LL"}]}]}
    m = build_mask(W, H, drivable_png=drv, lane_frame=lane, lane_thickness=8)
    assert m.shape == (H, W, 3) and m.dtype == np.uint8
    assert set(np.unique(m)) <= {0, 255}
    assert m[600, 600, CH_DRIVABLE] == 255 and m[600, 600, CH_ALTERNATIVE] == 0
    assert m[600, 200, CH_ALTERNATIVE] == 255 and m[600, 200, CH_DRIVABLE] == 0
    assert m[100, 100].sum() == 0
    # multi-label: vạch lane nằm đè lên vùng drivable
    assert m[600, 640, CH_LANE] == 255 and m[600, 640, CH_DRIVABLE] == 255
    assert m[600, 660, CH_LANE] == 0


def test_mask_from_polygons_old_format():
    frame = {
        "labels": [
            {"category": "drivable area", "attributes": {"areaType": "direct"},
             "poly2d": [{"vertices": [[0, 700], [600, 700], [600, 500], [0, 500]], "closed": True}]},
            {"category": "area/alternative",
             "poly2d": [{"vertices": [[700, 700], [1200, 700], [1200, 500], [700, 500]], "closed": True}]},
        ]
    }
    m = build_mask(W, H, drivable_frame=frame)
    assert m[600, 300, CH_DRIVABLE] == 255
    assert m[600, 900, CH_ALTERNATIVE] == 255
    assert m[600, 900, CH_DRIVABLE] == 0


def test_mask_from_lane_png_autodetects_background():
    for bg in (255, 0):
        lane = np.full((H, W), bg, np.uint8)
        lane[600:610, 300:900] = 3 if bg == 255 else 200  # vạch kẻ
        m = build_mask(W, H, lane_png=lane)
        assert m[605, 500, CH_LANE] == 255 and m[100, 100, CH_LANE] == 0
        assert m[..., CH_LANE].sum() == 255 * 10 * 600


def test_mask_save_load_preserves_channel_order(tmp_path):
    m = np.zeros((20, 30, 3), np.uint8)
    m[..., CH_DRIVABLE] = 255
    m[0, 0, CH_LANE] = 255
    p = tmp_path / "m.png"
    save_mask(str(p), m)
    back = load_mask(str(p))
    assert back.max() == 1
    assert back[..., CH_DRIVABLE].all()
    assert back[0, 0, CH_LANE] == 1 and back[..., CH_ALTERNATIVE].sum() == 0


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def _inputs():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W, 3), np.uint8)
    mask[:, :100, 0] = 1
    mask[:10, :10, 2] = IGNORE
    return img, mask, np.array([[10, 20, 110, 220]], np.float32), np.array([2], np.int64)


@pytest.mark.parametrize("backend", BACKENDS)
def test_eval_transforms_resize_and_normalize(backend):
    img, mask, boxes, labels = _inputs()
    out = T.build_eval_transforms(640, 384, backend=backend)(image=img, mask=mask, boxes=boxes, labels=labels)
    assert out["image"].shape == (384, 640, 3) and out["image"].dtype == np.float32
    assert abs(float(out["image"].mean())) < 1.5  # đã chuẩn hoá
    assert out["mask"].shape == (384, 640, 3)
    assert set(np.unique(out["mask"])) <= {0, 1, IGNORE}  # nearest: không sinh giá trị mới
    assert out["boxes"][0].tolist() == pytest.approx([5, 20 * 384 / 720, 55, 220 * 384 / 720], abs=0.5)
    assert out["labels"].tolist() == [2]


@pytest.mark.parametrize("backend", BACKENDS)
def test_preprocess_inference_equals_validation(backend):
    """Yêu cầu cốt lõi: tiền xử lý khi chạy thật == tiền xử lý validation."""
    frame_bgr, *_ = _inputs()  # BGR như cv2.imread / VideoCapture
    val = T.build_eval_transforms(640, 384, backend=backend)(image=frame_bgr)["image"]
    inf = T.preprocess_inference(frame_bgr, 640, 384, backend=backend)
    assert inf.shape == (3, 384, 640) and inf.dtype == np.float32
    np.testing.assert_array_equal(inf, val.transpose(2, 0, 1))


@pytest.mark.parametrize("backend", BACKENDS)
def test_eval_converts_bgr_to_rgb(backend):
    bgr = np.zeros((16, 16, 3), np.uint8)
    bgr[..., 2] = 255  # đỏ thuần trong BGR
    out = T.build_eval_transforms(16, 16, backend=backend)(image=bgr)["image"]
    rgb = T.denormalize(out)
    assert rgb[0, 0].tolist() == [255, 0, 0]  # sau BGR2RGB: kênh 0 (R) = 255


def test_denormalize_roundtrip():
    img, *_ = _inputs()
    back = T.denormalize(T.normalize_image(img))
    assert np.abs(back.astype(int) - img.astype(int)).max() <= 1
    chw = T.normalize_image(img).transpose(2, 0, 1)
    assert T.denormalize(chw).shape == img.shape


@pytest.mark.parametrize("backend", BACKENDS)
def test_train_transforms_shapes_and_valid_boxes(backend):
    t = T.build_train_transforms(512, 288, backend=backend, seed=0)
    for _ in range(15):
        img, mask, boxes, labels = _inputs()
        out = t(image=img, mask=mask, boxes=boxes, labels=labels)
        assert out["image"].shape == (288, 512, 3) and out["image"].dtype == np.float32
        assert out["mask"].shape == (288, 512, 3)
        assert set(np.unique(out["mask"])) <= {0, 1, IGNORE}
        assert len(out["boxes"]) == len(out["labels"])
        if len(out["boxes"]):
            b = out["boxes"]
            assert (b[:, [0, 2]] >= 0).all() and (b[:, [0, 2]] <= 512).all()
            assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()


def test_cv2_hflip_moves_boxes_and_mask():
    img, mask, boxes, labels = _inputs()
    out = T.Cv2Pipeline([T.HorizontalFlip(p=1.0)])(image=img, mask=mask, boxes=boxes, labels=labels)
    assert out["boxes"][0].tolist() == pytest.approx([W - 110, 20, W - 10, 220])
    assert out["mask"][:, -100:, 0].all() and out["mask"][:, :-100, 0].sum() == 0


# ---------------------------------------------------------------------------
# Cityscapes
# ---------------------------------------------------------------------------
def test_cityscapes_road_mapping():
    lid = np.full((4, 4), 8, np.uint8)  # sidewalk
    lid[2:] = 7  # road
    road = labelids_to_road(lid)
    assert road.tolist() == [[0] * 4, [0] * 4, [1] * 4, [1] * 4]


def test_road_from_prediction_is_direct_or_alternative():
    pred = np.zeros((3, 2, 3), np.float32)
    pred[0, 0, 0] = 0.9  # direct
    pred[1, 0, 1] = 0.8  # alternative
    pred[2, 1, 2] = 0.9  # lane -> không tính là road
    assert road_from_prediction(pred).tolist() == [[1, 1, 0], [0, 0, 0]]
    assert road_from_prediction(pred, threshold=(0.95, 0.5)).tolist() == [[0, 1, 0], [0, 0, 0]]


def test_road_iou_accumulates():
    m = RoadIoU()
    gt = np.array([[1, 1], [0, 0]], np.uint8)
    pred = np.zeros((3, 2, 2), np.float32)
    pred[0, 0, 0] = 1
    pred[1, 1, 0] = 1
    m.update(pred, gt)  # inter 1, union 3
    m.update(np.stack([gt, gt * 0, gt * 0]).astype(np.float32), gt)  # inter 2, union 2
    assert m.compute() == pytest.approx(3 / 5)


def test_cityscapes_road_dataset(tmp_path):
    cs = tmp_path / "cityscapes"
    (cs / "leftImg8bit" / "val" / "frankfurt").mkdir(parents=True)
    (cs / "gtFine" / "val" / "frankfurt").mkdir(parents=True)
    stem = "frankfurt_000000_000294"
    cv2.imwrite(str(cs / "leftImg8bit" / "val" / "frankfurt" / f"{stem}_leftImg8bit.png"), np.zeros((64, 128, 3), np.uint8))
    lid = np.zeros((64, 128), np.uint8)
    lid[40:] = 7
    cv2.imwrite(str(cs / "gtFine" / "val" / "frankfurt" / f"{stem}_gtFine_labelIds.png"), lid)

    ds = CityscapesRoadDataset(cs, "val")
    assert len(ds) == 1
    road = ds[0]["road"]
    assert road.shape == (64, 128) and road[50, 5] == 1 and road[5, 5] == 0

    for backend in BACKENDS:
        it = CityscapesRoadDataset(cs, "val", transforms=T.build_eval_transforms(64, 32, backend=backend))[0]
        assert it["image"].shape == (32, 64, 3) and it["road"].shape == (32, 64)
        assert set(np.unique(it["road"])) <= {0, 1}


# ---------------------------------------------------------------------------
# End-to-end: BDD giả lập -> prepare_data -> Dataset
# ---------------------------------------------------------------------------
@pytest.fixture()
def fake_bdd(tmp_path):
    root = tmp_path / "bdd100k"
    names = ["0000f77c-6257be58.jpg", "0001542f-7c670be8.jpg", "00067cfb-e535423e.jpg"]
    img_dir = root / "images" / "100k" / "val"
    img_dir.mkdir(parents=True)
    mask_dir = root / "labels" / "drivable" / "masks" / "val"
    mask_dir.mkdir(parents=True)
    frames, lanes = [], []
    for i, n in enumerate(names):
        cv2.imwrite(str(img_dir / n), np.full((H, W, 3), 40 * i, np.uint8))
        drv = np.full((H, W), 2, np.uint8)
        drv[450:, 300:980] = 0
        cv2.imwrite(str(mask_dir / n.replace(".jpg", ".png")), drv)
        labels = [{"category": "car", "box2d": {"x1": 100, "y1": 300, "x2": 300, "y2": 450}}]
        if i == 1:
            labels = []  # ảnh không có box
        frames.append({"name": n, "attributes": {"weather": "rainy" if i else "clear", "timeofday": "night"}, "labels": labels})
        lanes.append({"name": n, "labels": [{"category": "lane/single white", "poly2d": [{"vertices": [[640, 719], [600, 450]], "types": "LL", "closed": False}]}]})
    (root / "labels" / "det_20").mkdir(parents=True)
    (root / "labels" / "det_20" / "det_val.json").write_text(json.dumps(frames))
    (root / "labels" / "lane" / "polygons").mkdir(parents=True)
    (root / "labels" / "lane" / "polygons" / "lane_val.json").write_text(json.dumps(lanes))
    return root


def _load(name: str, rel: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _prepare():
    return _load("prepare_data", "scripts/prepare_data.py")


def test_inspect_script_cli(fake_bdd, capsys):
    inspect_bdd = _load("inspect_bdd", "scripts/inspect_bdd.py")
    assert inspect_bdd.main(["--bdd-root", str(fake_bdd), "--split", "val"]) == 0
    assert "structure_ok" in capsys.readouterr().out


def test_inspect_bdd(fake_bdd):
    r = inspect_split(fake_bdd, "val", sample_masks=3)
    assert r["num_frames"] == 3 and r["frames_missing_image"] == 0
    assert r["class_counts"] == {"car": 2}
    assert r["frames_without_box"] == 1
    assert r["num_drivable_masks"] == 3 and r["lane_frames_with_lane"] == 3
    assert r["weather"] == {"rainy": 2, "clear": 1}
    assert r["drivable_pixel_values"] == [0, 2]
    assert r["drivable_mask_shapes"] == {"1280x720": 3}
    assert r["raw_det_categories"] == {"car": 2}
    assert "lane/single white" in r["raw_lane_categories"]
    assert r["structure_ok"] is True


def test_inspect_bdd_flags_bad_mask_values(fake_bdd):
    mdir = fake_bdd / "labels" / "drivable" / "masks" / "val"
    cv2.imwrite(str(next(mdir.glob("*.png"))), np.full((H, W), 128, np.uint8))  # mask "màu"
    r = inspect_split(fake_bdd, "val", sample_masks=3)
    assert 128 in r["drivable_pixel_values"]
    assert any("giá trị lạ" in w for w in r["warnings"])


def test_prepare_detection_end_to_end(fake_bdd, tmp_path):
    prepare_data = _prepare()
    out = tmp_path / "processed"
    prepare_data.main(["detection", "--bdd-root", str(fake_bdd), "--out", str(out), "--splits", "val", "--link-mode", "copy"])
    det = out / "bdd100k_det"
    assert len(list((det / "labels" / "val").glob("*.txt"))) == 3
    assert len(list((det / "images" / "val").glob("*.jpg"))) == 3
    yaml_text = (det / "data.yaml").read_text()
    assert "nc: 9" in yaml_text and "val: images/val" in yaml_text

    import yaml

    cfg = yaml.safe_load(yaml_text)
    assert cfg["names"][0] == "car" and cfg["names"][7] == "traffic_light"

    ds = DetectionDataset(det, "val")
    item = ds[0]
    assert item["image"].shape == (H, W, 3)
    assert item["boxes"][0].tolist() == pytest.approx([100, 300, 300, 450], abs=0.01)
    assert item["labels"].tolist() == [DET_CLASS_TO_ID["car"]]
    assert len(ds[1]["boxes"]) == 0


def test_prepare_detection_skip_empty_and_limit(fake_bdd, tmp_path):
    prepare_data = _prepare()
    out = tmp_path / "p2"
    prepare_data.main(["detection", "--bdd-root", str(fake_bdd), "--out", str(out), "--splits", "val",
                       "--link-mode", "copy", "--skip-empty"])
    assert len(list((out / "bdd100k_det" / "labels" / "val").glob("*.txt"))) == 2
    out3 = tmp_path / "p3"
    prepare_data.main(["detection", "--bdd-root", str(fake_bdd), "--out", str(out3), "--splits", "val",
                       "--link-mode", "copy", "--limit", "1"])
    assert len(list((out3 / "bdd100k_det" / "labels" / "val").glob("*.txt"))) == 1


def test_prepare_segmentation_end_to_end(fake_bdd, tmp_path):
    prepare_data = _prepare()
    out = tmp_path / "processed"
    prepare_data.main(["segmentation", "--bdd-root", str(fake_bdd), "--out", str(out), "--splits", "val",
                       "--link-mode", "copy", "--mask-size", "640", "360"])
    saved = load_mask(str(next((out / "bdd100k_seg" / "masks" / "val").glob("*.png"))))
    assert saved.shape == (360, 640, 3)
    ds = SegmentationDataset(out / "bdd100k_seg", "val")
    assert len(ds) == 3
    item = ds[0]
    m = item["mask"]
    assert m.shape == (H, W, 3) and set(np.unique(m)) <= {0, 1}  # tự đưa về cỡ ảnh
    assert m[600, 640, CH_DRIVABLE] == 1 and m[10, 10].sum() == 0
    assert m[:, :, CH_LANE].sum() > 0

    for backend in BACKENDS:
        ds_aug = SegmentationDataset(out / "bdd100k_seg", "val",
                                     transforms=T.build_train_transforms(512, 288, backend=backend, seed=0))
        it = ds_aug[1]
        assert it["image"].shape[:2] == it["mask"].shape[:2] == (288, 512)


# ---------------------------------------------------------------------------
# Subset cố định (seed=42)
# ---------------------------------------------------------------------------
def test_subset_deterministic_and_used_by_prepare(fake_bdd, tmp_path):
    prepare_data = _prepare()
    d1, d2 = tmp_path / "s1", tmp_path / "s2"
    for d in (d1, d2):
        prepare_data.main(["subset", "--bdd-root", str(fake_bdd), "--splits-dir", str(d),
                           "--splits", "val", "--val-size", "2"])
    assert (d1 / "val.txt").read_text() == (d2 / "val.txt").read_text()  # lặp lại được
    names = split_lib.read_split(d1 / "val.txt")
    assert len(names) == 2 and "seed=42" in (d1 / "val.txt").read_text()

    out = tmp_path / "processed"
    prepare_data.main(["all", "--bdd-root", str(fake_bdd), "--out", str(out), "--splits", "val",
                       "--link-mode", "copy", "--use-splits", "--splits-dir", str(d1)])
    det_names = sorted(p.stem for p in (out / "bdd100k_det" / "labels" / "val").glob("*.txt"))
    seg_names = sorted(p.stem for p in (out / "bdd100k_seg" / "masks" / "val").glob("*.png"))
    assert det_names == seg_names == sorted(Path(n).stem for n in names)  # TV3 & TV4 cùng tập val


def test_subset_tag_file_name(fake_bdd, tmp_path):
    _prepare().main(["subset", "--bdd-root", str(fake_bdd), "--splits-dir", str(tmp_path),
                     "--splits", "val", "--tag", "main"])
    assert (tmp_path / "main_val.txt").exists()


def test_stratified_pick_keeps_proportions():
    names = [f"{i:05d}.jpg" for i in range(1000)]
    strata = {n: ("clear|daytime" if i < 600 else "rainy|night" if i < 900 else "snowy|night")
              for i, n in enumerate(names)}
    a = split_lib.stratified_pick(names, strata, 100, 42)
    assert a == split_lib.stratified_pick(list(reversed(names)), strata, 100, 42)
    counts = {k: sum(strata[n] == k for n in a) for k in set(strata.values())}
    assert counts == {"clear|daytime": 60, "rainy|night": 30, "snowy|night": 10}


def test_random_pick_is_seeded():
    names = [f"{i:05d}.jpg" for i in range(100)]
    a = split_lib.random_pick(names, 10, 42)
    assert a == split_lib.random_pick(list(reversed(names)), 10, 42) and len(a) == 10
    assert split_lib.random_pick(names, 10, 7) != a


def test_paths_yaml_is_read(tmp_path, fake_bdd):
    from src.data.bdd_layout import load_data_paths

    cfg = tmp_path / "paths.yaml"
    cfg.write_text(f"project:\n  seed: 7\npaths:\n  bdd_root: {fake_bdd.as_posix()}\n")
    dp = load_data_paths(cfg)
    assert dp.seed == 7 and dp.bdd_root == fake_bdd
    assert load_data_paths(tmp_path / "missing.yaml").bdd_root is None  # file trống/thiếu -> None
    # script dùng bdd_root từ config khi không truyền --bdd-root
    _prepare().main(["subset", "--config", str(cfg), "--splits-dir", str(tmp_path / "sp"), "--splits", "val"])
    assert "seed=7" in (tmp_path / "sp" / "val.txt").read_text()


def test_custom_drivable_ids():
    drv = np.full((4, 4), 0, np.uint8)
    drv[2:] = 1
    m = build_mask(4, 4, drivable_png=drv, drivable_ids=DrivableIds(direct=1, alternative=2, background=0))
    assert m[3, 0, CH_DRIVABLE] == 255 and m[0, 0, CH_DRIVABLE] == 0


# ---------------------------------------------------------------------------
# Download script (giải nén + xác minh cấu trúc)
# ---------------------------------------------------------------------------
def test_download_script_extracts_and_verifies(tmp_path):
    import zipfile

    dl = _load("dl", "data/dowload_data_bdd100k.py")
    raw = tmp_path / "raw"
    raw.mkdir()
    with zipfile.ZipFile(raw / "100k_images_val.zip", "w") as z:  # ảnh lồng thêm cấp, cần sửa
        z.writestr("bdd100k/images/100k/val/val/a.jpg", b"x")
    with zipfile.ZipFile(raw / "bdd100k_det_20_labels_trainval.zip", "w") as z:
        z.writestr("bdd100k/labels/det_20/det_train.json", "[]")
        z.writestr("bdd100k/labels/det_20/det_val.json", "[]")
    rc = dl.main(["--out", str(tmp_path), "--zips-dir", str(raw), "--parts", "images_val", "det"])
    assert rc == 0
    assert (tmp_path / "bdd100k" / "images" / "100k" / "val" / "a.jpg").exists()
    assert dl.main(["--out", str(tmp_path), "--parts", "lane", "--verify-only"]) == 1


def test_prepare_segmentation_with_lane_masks(fake_bdd, tmp_path):
    """Khi có labels/lane/masks (lane_maps) thì dùng mask thay cho polyline JSON."""
    prepare_data = _prepare()
    ldir = fake_bdd / "labels" / "lane" / "masks" / "val"
    ldir.mkdir(parents=True)
    for img in (fake_bdd / "images" / "100k" / "val").glob("*.jpg"):
        lane = np.full((H, W), 255, np.uint8)
        lane[700:705, 100:1100] = 0
        cv2.imwrite(str(ldir / f"{img.stem}.png"), lane)
    out = tmp_path / "processed"
    prepare_data.main(["segmentation", "--bdd-root", str(fake_bdd), "--out", str(out), "--splits", "val",
                       "--link-mode", "copy"])
    m = SegmentationDataset(out / "bdd100k_seg", "val")[0]["mask"]
    assert m[702, 500, CH_LANE] == 1
    assert m[600, 620, CH_LANE] == 0  # polyline JSON (x≈620) không còn được dùng
    assert m[..., CH_LANE].sum() == 5 * 1000
