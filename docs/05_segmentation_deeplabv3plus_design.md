# 05 — Thiết kế & Huấn luyện Semantic Segmentation (DeepLabV3+)

## 1. Mục tiêu Module
Xây dựng mô hình phân vùng ngữ nghĩa (`src/models/segmenter.py`) nhằm trích xuất chính xác 3 mặt nạ không gian đường theo thứ tự `SEG_CHANNELS`:
1. `drivable_direct` (Làn đường chính xe đang di chuyển)
2. `drivable_alternative` (Làn đường lân cận có thể lưu thông)
3. `lane_marking` (Hệ thống vạch kẻ sơn phân làn)

---

## 2. Kiến trúc Mô hình: DeepLabV3+

Sử dụng thư viện `segmentation-models-pytorch` (`smp.DeepLabV3Plus`):

* **Encoder (Backbone):**
  * Lựa chọn chính: `resnet34` (pretrained `imagenet`) — tốc độ nhanh, độ chính xác ổn định trên T4.
  * Lựa chọn nhẹ thay thế: `mobilenet_v2` hoặc `timm-mobilenetv3_large_100` nếu cần tối ưu thêm FPS.
* **Module ASPP (Atrous Spatial Pyramid Pooling):**
  * Sử dụng các nhánh tích chập giãn (dilated convolution) với nhiều tỷ lệ khác nhau giúp mô hình nhìn được đồng thời:
    * Ngữ cảnh rộng của toàn mặt đường (`drivable_direct`).
    * Chi tiết mảnh, kéo dài của vạch kẻ đường (`lane_marking`).
* **Decoder:** Kết hợp đặc trưng độ phân giải cao từ tầng nông (low-level features) của Encoder để phục hồi biên sắc nét cho vạch sơn đường.
* **Head đầu ra:** `classes=3`, **không gắn Softmax**, đầu ra là logits kích thước `(B, 3, H, W)`. Khi suy luận, đưa qua hàm **Sigmoid** độc lập cho từng kênh (Multi-label).

---

## 3. Chiến lược Hàm Mất mát (Loss Function)

Bài toán có sự mất cân bằng điểm ảnh nghiêm trọng (ví dụ `lane_marking` chỉ chiếm khoảng $1\% - 3\%$ tổng số pixel khung hình). Nếu chỉ dùng Cross Entropy thông thường, mô hình dễ bỏ qua vạch kẻ đường.

**Giải pháp — Combo Loss:**
$$\mathcal{L}_{\text{total}} = w_{\text{bce}} \cdot \mathcal{L}_{\text{BCEWithLogits}} + w_{\text{dice}} \cdot \mathcal{L}_{\text{Dice}}$$

* $\mathcal{L}_{\text{BCEWithLogits}}$: Giúp tối ưu xác suất ổn định ở cấp độ từng pixel (có thể gắn `pos_weight` cho kênh `lane_marking`).
* $\mathcal{L}_{\text{Dice}}$: Tối ưu trực tiếp chỉ số chồng lấn (F1/IoU), đặc biệt hiệu quả với các đối tượng mảnh và chiếm tỷ lệ nhỏ như `lane_marking`.
* Trọng số khởi điểm đề xuất: `w_bce = 0.5`, `w_dice = 0.5` (hoặc tăng trọng số kênh `lane_marking` lên $1.5\times - 2.0\times$ nếu IoU vạch kẻ đường thấp).

---

## 4. Định dạng Checkpoint Chuẩn (`CLAUDE.md` rule #21)

Mọi checkpoint lưu trong quá trình train (`weights/segmenter_best.pt`) **bắt buộc** phải chứa đầy đủ dictionary các khóa sau:

```python
checkpoint = {
    "model_state": model.state_dict(),
    "cfg": cfg_dict,                      # Toàn bộ dict cấu hình đã dùng để train
    "epoch": epoch,                       # Epoch đạt kết quả tốt nhất
    "metrics": {                          # Metric trên tập val tại epoch này
        "iou_drivable_direct": float,
        "iou_drivable_alternative": float,
        "iou_lane_marking": float,
        "mean_iou": float,
    },
    "thresholds": [0.5, 0.5, 0.4],        # Ngưỡng Sigmoid tối ưu cho 3 kênh
    "size_hw": (384, 640),                # Kích thước ảnh đầu vào (H, W)
    "channels": SEG_CHANNELS,             # ("drivable_direct", "drivable_alternative", "lane_marking")
}
```

> [!IMPORTANT]
> Việc lưu kèm `size_hw`, `thresholds` và `channels` trong checkpoint giúp hàm suy luận tự động đồng bộ đúng kích thước tiền xử lý và thứ tự kênh với quá trình huấn luyện.

---

## 5. Thiết kế Wrapper (`src/models/segmenter.py`)

1. **Nạp checkpoint:**
   * Kiểm tra `checkpoint["channels"] == SEG_CHANNELS`. Nếu lệch thứ tự kênh, raise lỗi ngay.
   * Khởi tạo đúng kiến trúc `smp.DeepLabV3Plus` từ `checkpoint["cfg"]`, load `model_state`, chuyển sang `.eval()`.
2. **Hàm `predict(frame_bgr: np.ndarray) -> np.ndarray`:**
   * Gọi hàm tiền xử lý dùng chung (`preprocess_for_segmentation(frame_bgr, size_hw)`).
   * Chạy `with torch.inference_mode():` (kèm `torch.cuda.amp.autocast()` nếu chạy GPU).
   * Áp dụng `torch.sigmoid(logits)` $\rightarrow$ resize xác suất về kích thước `(H_orig, W_orig)` của frame gốc (chú ý `cv2.resize` nhận `(W_orig, H_orig)`).
   * Áp dụng ngưỡng `thresholds` để trả về mask nhị phân `uint8` shape `(3, H_orig, W_orig)` hoặc cả xác suất lẫn binary mask tùy chế độ.

---

## 6. Huấn luyện & Tối ưu Ngưỡng (`scripts/train_segmenter.py`)

* **Optimizer:** `AdamW` với `lr = 3e-4`, `weight_decay = 1e-4`.
* **Scheduler:** `CosineAnnealingLR` hoặc `ReduceLROnPlateau` theo `mean_iou`.
* **Mixed Precision (AMP):** Bật `torch.amp.GradScaler` để tiết kiệm VRAM và tăng tốc độ train ~1.8x trên T4.
* **Tối ưu Threshold sau Train:** Trên tập `val`, quét ngưỡng $t \in [0.2, 0.7]$ (bước $0.05$) độc lập cho từng kênh (đặc biệt là `lane_marking`) để chọn ngưỡng tối đa hóa IoU trước khi đóng gói vào checkpoint cuối cùng.
