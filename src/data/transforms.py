"""Transforms cho train / validation / inference (Albumentations).

Theo docs/03_dataset_and_bdd100k_spec.md §8:

* Train: Resize -> HorizontalFlip(0.5) -> RandomBrightnessContrast(0.4)
  -> HueSaturationValue(0.3) -> RandomShadow(0.2) -> MotionBlur(0.2)
  -> Normalize(ImageNet) [-> ToTensorV2]
* Validation / Inference: Resize -> BGR2RGB -> Normalize [-> ToTensorV2]

Đầu vào mọi pipeline là ảnh nội bộ **BGR uint8 (H, W, 3)** (docs/02 §2.3, CLAUDE.md #12).
Việc đổi sang RGB + chuẩn hoá CHỈ diễn ra ở đây.

QUAN TRỌNG: validation và inference dùng CHUNG MỘT hàm ``build_eval_transforms``.
``preprocess_inference`` chỉ gọi lại đúng pipeline đó -> không lệch train/inference.

Giao diện chung của mọi pipeline::

    out = t(image=bgr_uint8, mask=None, boxes=None, labels=None)
    out["image"]  # float32 (H, W, 3) đã chuẩn hoá  | torch (3, H, W) nếu to_tensor=True
    out["mask"]   # uint8 (H, W, C) {0,1}            | torch (C, H, W) nếu to_tensor=True
    out["boxes"]  # float32 (N, 4) xyxy pixel
    out["labels"] # int64 (N,)

Backend: ``albumentations`` (mặc định nếu đã cài) hoặc ``cv2`` (dự phòng thuần
numpy/OpenCV, cùng ngữ nghĩa — để CI/test chạy được khi chưa cài Albumentations).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Sequence

import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Kích thước mặc định cho segmentation theo spec: H×W = 384×640
DEFAULT_INPUT_SIZE = (640, 384)  # (W, H)

try:
    import albumentations as A  # type: ignore

    HAS_ALBUMENTATIONS = True
except Exception:  # pragma: no cover - phụ thuộc môi trường
    A = None
    HAS_ALBUMENTATIONS = False


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "albumentations" if HAS_ALBUMENTATIONS else "cv2"
    if backend == "albumentations" and not HAS_ALBUMENTATIONS:
        raise ImportError("Chưa cài albumentations: pip install albumentations")
    if backend not in ("albumentations", "cv2"):
        raise ValueError(f"backend không hợp lệ: {backend}")
    return backend


def _empty_boxes() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)


def _to_tensor(out: dict) -> dict:
    """HWC numpy -> CHW torch (tương đương ToTensorV2(transpose_mask=True))."""
    import torch

    out["image"] = torch.from_numpy(np.ascontiguousarray(out["image"].transpose(2, 0, 1)))
    if out.get("mask") is not None:
        m = out["mask"]
        m = m[..., None] if m.ndim == 2 else m
        out["mask"] = torch.from_numpy(np.ascontiguousarray(m.transpose(2, 0, 1)))
    return out


# ===========================================================================
# Albumentations backend
# ===========================================================================
class AlbuPipeline:
    def __init__(self, transforms: list, to_tensor: bool = False):
        self.compose = A.Compose(
            transforms, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.3)
        )
        self.to_tensor = to_tensor

    def __call__(self, image, mask=None, boxes=None, labels=None) -> dict:
        bx, lb = _empty_boxes() if boxes is None else (boxes, labels)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kw: dict = {"image": rgb, "bboxes": [list(map(float, x)) for x in bx], "labels": [int(x) for x in lb]}
        if mask is not None:
            kw["mask"] = mask
        res = self.compose(**kw)
        if len(res["bboxes"]):
            ob = np.asarray(res["bboxes"], np.float32).reshape(-1, 4)
            ol = np.asarray(res["labels"], np.int64)
        else:
            ob, ol = _empty_boxes()
        out = {"image": res["image"].astype(np.float32), "mask": res.get("mask"), "boxes": ob, "labels": ol}
        return _to_tensor(out) if self.to_tensor else out


def _albu_normalize():
    return A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0)


def _albu_eval(width: int, height: int) -> list:
    return [A.Resize(height=height, width=width), _albu_normalize()]


def _albu_train(width: int, height: int) -> list:
    return [
        A.Resize(height=height, width=width),
        A.HorizontalFlip(p=0.5),  # lật cả mask; drivable_direct vẫn là drivable_direct
        A.RandomBrightnessContrast(p=0.4),
        A.HueSaturationValue(p=0.3),
        A.RandomShadow(p=0.2),
        A.MotionBlur(p=0.2),
        _albu_normalize(),
    ]


# ===========================================================================
# cv2 backend (dự phòng, cùng ngữ nghĩa)
# ===========================================================================
@dataclass
class Sample:
    image: np.ndarray
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    mask: np.ndarray | None = None


Op = Callable[[Sample, np.random.Generator], Sample]


class Cv2Pipeline:
    def __init__(self, ops: Sequence[Op], seed: int | None = None, to_tensor: bool = False):
        self.ops = list(ops)
        self.rng = np.random.default_rng(seed)
        self.to_tensor = to_tensor

    def __call__(self, image, mask=None, boxes=None, labels=None) -> dict:
        if boxes is None:
            bx, lb = _empty_boxes()
        else:
            bx, lb = np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(labels, np.int64)
        s = Sample(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB), boxes=bx, labels=lb, mask=mask)
        for op in self.ops:
            s = op(s, self.rng)
        out = {"image": s.image, "mask": s.mask, "boxes": s.boxes, "labels": s.labels}
        return _to_tensor(out) if self.to_tensor else out


def _resize_mask(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """Resize nearest từng kênh (không sinh giá trị nhãn mới)."""
    if mask.ndim == 2:
        return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.stack(
        [cv2.resize(np.ascontiguousarray(mask[..., i]), (w, h), interpolation=cv2.INTER_NEAREST) for i in range(mask.shape[2])],
        axis=2,
    )


class Resize:
    def __init__(self, width: int, height: int):
        self.w, self.h = width, height

    def __call__(self, s: Sample, rng) -> Sample:
        h0, w0 = s.image.shape[:2]
        sx, sy = self.w / w0, self.h / h0
        s.image = cv2.resize(s.image, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        if len(s.boxes):
            s.boxes = s.boxes * np.array([sx, sy, sx, sy], np.float32)
        if s.mask is not None:
            s.mask = _resize_mask(s.mask, self.w, self.h)
        return s


class HorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, s: Sample, rng) -> Sample:
        if rng.random() >= self.p:
            return s
        w = s.image.shape[1]
        s.image = np.ascontiguousarray(s.image[:, ::-1])
        if len(s.boxes):
            b = s.boxes.copy()
            s.boxes[:, 0], s.boxes[:, 2] = w - b[:, 2], w - b[:, 0]
        if s.mask is not None:
            s.mask = np.ascontiguousarray(s.mask[:, ::-1])
        return s


class RandomBrightnessContrast:
    def __init__(self, limit: float = 0.2, p: float = 0.4):
        self.limit, self.p = limit, p

    def __call__(self, s: Sample, rng) -> Sample:
        if rng.random() >= self.p:
            return s
        alpha = 1 + rng.uniform(-self.limit, self.limit)  # contrast
        beta = rng.uniform(-self.limit, self.limit) * 255  # brightness
        s.image = (s.image.astype(np.float32) * alpha + beta).clip(0, 255).astype(np.uint8)
        return s


class HueSaturationValue:
    def __init__(self, hue: int = 20, sat: int = 30, val: int = 20, p: float = 0.3):
        self.hue, self.sat, self.val, self.p = hue, sat, val, p

    def __call__(self, s: Sample, rng) -> Sample:
        if rng.random() >= self.p:
            return s
        hsv = cv2.cvtColor(s.image, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(rng.integers(-self.hue, self.hue + 1)) // 2) % 180
        hsv[..., 1] = (hsv[..., 1] + int(rng.integers(-self.sat, self.sat + 1))).clip(0, 255)
        hsv[..., 2] = (hsv[..., 2] + int(rng.integers(-self.val, self.val + 1))).clip(0, 255)
        s.image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return s


class RandomShadow:
    """Bóng đổ đa giác ngẫu nhiên ở nửa dưới ảnh (giống A.RandomShadow mặc định)."""

    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, s: Sample, rng) -> Sample:
        if rng.random() >= self.p:
            return s
        h, w = s.image.shape[:2]
        n = int(rng.integers(3, 6))
        pts = np.stack([rng.uniform(0, w, n), rng.uniform(h / 2, h, n)], axis=1).astype(np.int32)
        shadow = np.zeros((h, w), np.uint8)
        cv2.fillPoly(shadow, [cv2.convexHull(pts)], 1)
        img = s.image.astype(np.float32)
        img[shadow == 1] *= 0.5
        s.image = img.astype(np.uint8)
        return s


class MotionBlur:
    def __init__(self, max_kernel: int = 7, p: float = 0.2):
        self.max_kernel, self.p = max_kernel, p

    def __call__(self, s: Sample, rng) -> Sample:
        if rng.random() >= self.p:
            return s
        k = int(rng.choice(np.arange(3, self.max_kernel + 1, 2)))
        kernel = np.zeros((k, k), np.float32)
        if rng.random() < 0.5:
            kernel[k // 2, :] = 1.0
        else:
            kernel[:, k // 2] = 1.0
        s.image = cv2.filter2D(s.image, -1, kernel / k)
        return s


class Normalize:
    def __call__(self, s: Sample, rng) -> Sample:
        s.image = normalize_image(s.image)
        return s


# ===========================================================================
# API công khai
# ===========================================================================
def normalize_image(image: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float32 chuẩn hoá ImageNet (giống ``A.Normalize``)."""
    x = image.astype(np.float32) / 255.0
    return (x - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)


def denormalize(image) -> np.ndarray:
    """Ngược lại ``normalize_image`` (nhận HWC/CHW, numpy/torch) -> uint8 HWC RGB. Dùng cho TV6 vẽ HUD."""
    x = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    x = x.astype(np.float32)
    if x.ndim == 3 and x.shape[0] == 3 and x.shape[2] != 3:
        x = x.transpose(1, 2, 0)
    x = x * np.asarray(IMAGENET_STD, np.float32) + np.asarray(IMAGENET_MEAN, np.float32)
    return (x * 255).round().clip(0, 255).astype(np.uint8)


def build_eval_transforms(
    width: int = DEFAULT_INPUT_SIZE[0],
    height: int = DEFAULT_INPUT_SIZE[1],
    backend: str = "auto",
    to_tensor: bool = False,
):
    """Pipeline VALIDATION = INFERENCE: Resize -> Normalize (-> ToTensor). Không ngẫu nhiên."""
    backend = _resolve_backend(backend)
    if backend == "albumentations":
        return AlbuPipeline(_albu_eval(width, height), to_tensor)
    return Cv2Pipeline([Resize(width, height), Normalize()], to_tensor=to_tensor)


def build_train_transforms(
    width: int = DEFAULT_INPUT_SIZE[0],
    height: int = DEFAULT_INPUT_SIZE[1],
    backend: str = "auto",
    seed: int | None = None,
    to_tensor: bool = False,
):
    """Pipeline TRAIN theo spec §8."""
    backend = _resolve_backend(backend)
    if backend == "albumentations":
        return AlbuPipeline(_albu_train(width, height), to_tensor)
    return Cv2Pipeline(
        [
            Resize(width, height),
            HorizontalFlip(p=0.5),
            RandomBrightnessContrast(p=0.4),
            HueSaturationValue(p=0.3),
            RandomShadow(p=0.2),
            MotionBlur(p=0.2),
            Normalize(),
        ],
        seed=seed,
        to_tensor=to_tensor,
    )


@lru_cache(maxsize=8)
def _cached_eval(width: int, height: int, backend: str):
    return build_eval_transforms(width, height, backend)


def preprocess_inference(
    frame_bgr: np.ndarray,
    width: int = DEFAULT_INPUT_SIZE[0],
    height: int = DEFAULT_INPUT_SIZE[1],
    backend: str = "auto",
) -> np.ndarray:
    """Tiền xử lý 1 frame BGR khi chạy thật (video / webcam / Streamlit).

    Gọi ĐÚNG ``build_eval_transforms`` -> trùng khớp validation (CLAUDE.md #20).
    Trả về float32 (3, H, W): ``torch.from_numpy(x)[None]`` để đưa vào model.
    """
    out = _cached_eval(width, height, _resolve_backend(backend))(image=frame_bgr)
    return np.ascontiguousarray(out["image"].transpose(2, 0, 1))


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "DEFAULT_INPUT_SIZE",
    "HAS_ALBUMENTATIONS",
    "build_train_transforms",
    "build_eval_transforms",
    "preprocess_inference",
    "normalize_image",
    "denormalize",
]
