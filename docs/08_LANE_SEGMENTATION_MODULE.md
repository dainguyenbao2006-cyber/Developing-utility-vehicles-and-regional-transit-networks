# 08 — Lane / Drivable-area segmentation module

## 1. Bài toán

Input: ảnh RGB dashcam. Output: **3 kênh nhị phân độc lập** (multi-label) cùng kích thước ảnh:

| Kênh | Tên | Ý nghĩa |
|---|---|---|
| 0 | `drivable_direct` | Làn ego (xe đang chạy) |
| 1 | `drivable_alternative` | Làn khác có thể chuyển sang |
| 2 | `lane_marking` | Vạch kẻ đường (mỏng, mất cân bằng nặng) |

Dùng **sigmoid + BCE/Dice**, không softmax, vì vạch kẻ nằm chồng trên vùng drivable.

## 2. Model (`segmentation/model.py`)

```python
import segmentation_models_pytorch as smp

def build_model(cfg) -> torch.nn.Module:
    arch = cfg.arch  # "DeepLabV3Plus" | "Unet" | "FPN"
    return getattr(smp, arch)(
        encoder_name=cfg.encoder,          # "resnet34"
        encoder_weights=cfg.encoder_weights,  # "imagenet" (None khi load ckpt)
        in_channels=3,
        classes=3,
        activation=None,                   # trả logits
    )
```

| Cấu hình | Dùng khi |
|---|---|
| `DeepLabV3Plus + resnet34` (mặc định) | Cân bằng tốt, ổn định |
| `Unet + resnet34` | Giữ chi tiết mỏng (vạch kẻ) tốt hơn |
| `FPN + mobilenet_v2`/`efficientnet-b0` | Cần nhanh/nhẹ |
| Transformer (HF `SegformerForSemanticSegmentation`, `nvidia/mit-b0`) | Mở rộng, thử nghiệm |

Input size mặc định `(H, W) = (384, 640)` (chia hết cho 32). Chế độ chi tiết cao: `(512, 896)` — cải thiện lane marking nhưng chậm hơn và tốn VRAM hơn.

## 3. Loss (`segmentation/losses.py`)

```python
class SegLoss(nn.Module):
    def __init__(self, pos_weight=(1.0, 1.0, 3.0), w_bce=1.0, w_dice=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).view(1, 3, 1, 1))
        self.dice = smp.losses.DiceLoss(mode="multilabel", from_logits=True)
        ...
    def forward(self, logits, target):     # target float (B,3,H,W)
        return self.w_bce * self.bce(logits, target) + self.w_dice * self.dice(logits, target)
```
- `pos_weight` đưa vào `.to(logits.device)` đúng lúc (đăng ký bằng `register_buffer`).
- Nếu lane marking vẫn quá thấp: thêm `FocalLoss` hoặc Tversky (`smp.losses.TverskyLoss(alpha=0.3, beta=0.7)`) cho kênh 2.
- Cho phép trọng số kênh qua config `segmentation.loss.*`.

## 4. Metrics (`segmentation/metrics.py`)

**IoU tích luỹ trên toàn bộ tập** (không trung bình theo ảnh):

```python
class IoUMeter:
    def __init__(self, n=3, thr=0.5):
        self.inter = torch.zeros(n); self.union = torch.zeros(n); self.thr = thr
    @torch.no_grad()
    def update(self, logits, target):
        pred = (logits.sigmoid() > self.thr)
        tgt = target > 0.5
        self.inter += (pred & tgt).sum(dim=(0, 2, 3)).cpu().float()
        self.union += (pred | tgt).sum(dim=(0, 2, 3)).cpu().float()
    def compute(self):
        return (self.inter / self.union.clamp(min=1)).tolist()    # IoU từng kênh
```
Báo cáo thêm Precision/Recall/F1 của kênh lane marking. Tìm ngưỡng tối ưu riêng cho mỗi kênh trên val (quét 0.3–0.7) → lưu vào checkpoint (`thresholds`).

## 4b. Huấn luyện (`scripts/train_segmenter.py`)

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW, `lr=3e-4`, `weight_decay=1e-4` |
| Scheduler | `OneCycleLR` hoặc cosine với warmup 1 epoch |
| Batch | 16 (384×640, T4, AMP) / 8 (512×896) |
| Epochs | 40 (subset 10K) |
| AMP | `torch.cuda.amp.autocast` + `GradScaler` |
| Grad clip | 1.0 |
| Seed | 42 (`torch`, `numpy`, `random`) |
| Early stopping | patience 8 theo `mean_iou(direct, alt)` + 0.5 × `iou_lane` (cấu hình được) |
| Windows | `num_workers` 2–4, `persistent_workers=True`, bảo vệ `if __name__ == "__main__":` |

Vòng lặp:
1. Mỗi epoch: train → val → in bảng `loss, iou_direct, iou_alt, iou_lane, lr`.
2. Lưu `runs/seg/<name>/last.pt` mỗi epoch, `best.pt` khi metric tốt hơn.
3. Checkpoint chứa:
```python
{"model_state": ..., "cfg": cfg_dict, "epoch": e, "metrics": {...},
 "thresholds": [0.5, 0.5, 0.5], "size_hw": (384, 640),
 "channels": SEG_CHANNELS, "git_commit": ...}
```
4. Ghi `metrics.csv` + (tuỳ chọn) TensorBoard.
5. Resume: `--resume runs/seg/<name>/last.pt`.

Mẹo:
- Epoch đầu `freeze` encoder (tuỳ chọn) 1–2 epoch để ổn định.
- Overfit thử 50 ảnh trước (`--overfit 50`) → loss phải giảm sát 0, **đây là test sanity bắt buộc** trước khi train dài.

## 5. Inference (`segmentation/segmenter.py`)

```python
class Segmenter:
    def __init__(self, ckpt: str, device="cuda", half=True, thresholds=None): ...

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> SegResult:
        H, W = frame_bgr.shape[:2]
        x = preprocess(frame_bgr, self.size_hw)           # BGR→RGB, resize INTER_AREA, normalize ImageNet, CHW
        x = torch.from_numpy(x)[None].to(self.device, dtype=self.dtype)
        logits = self.model(x)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        probs = logits.sigmoid()[0]                       # (3,H,W)
        thr = torch.tensor(self.thresholds, device=probs.device).view(3, 1, 1)
        masks = (probs > thr).cpu().numpy()               # bool (3,H,W)
        return SegResult(masks=masks, probs=None)
```

- Preprocess **phải giống hệt** train (cùng INTER_AREA, mean/std).
- `postprocess.py`: loại thành phần liên thông nhỏ (`cv2.connectedComponentsWithStats`, `min_area` theo kênh), opening/closing nhẹ cho drivable, **không** morphological mạnh cho lane marking.
- Temporal smoothing (tuỳ chọn): `probs_ema = a*probs + (1-a)*probs_prev` với `a=0.6` để giảm nhấp nháy mask.

## 6. Trích hình học làn (`adas/lane_geometry.py`)

Xem thuật toán chi tiết ở `09_ADAS_LOGIC.md` mục 2. Tóm tắt: lấy dải ngang ROI phía dưới ảnh (70–90% chiều cao), tìm biên trái/phải của **vùng liên thông của kênh `drivable_direct` chứa/gần tâm xe**, lấy median qua các hàng.

Mở rộng (tuỳ chọn, giai đoạn sau): fit đa thức bậc 2 cho biên trái/phải trên toàn mask để vẽ đường làn cong và tính độ cong; dùng `lane_marking` để xác định xe có **đè lên vạch** (so mask vạch với hình chữ nhật dưới đầu xe).

## 7. Đánh giá
- `scripts/eval_segmenter.py --ckpt ... --split val` → IoU từng kênh, P/R/F1, lưu 30 ảnh dự đoán so GT (ghép 3 cột: ảnh | GT | pred).
- Phân tích theo `timeofday`/`weather`/`scene`.
- Cross-dataset: Cityscapes `road` vs `direct ∪ alternative` (chỉ tham khảo).
- Lỗi thường gặp: vạch bị đứt ở xa, bóng cây nhầm vạch, mặt đường ướt phản chiếu, đêm tối.
