# 16 — Rủi ro, giới hạn & lưu ý an toàn

## 1. Disclaimer an toàn
Dự án là **demo học thuật/nghiên cứu**. **Không** dùng để ra quyết định lái xe thật, không thay thế hệ thống ADAS thương mại, chưa qua kiểm chứng an toàn chức năng. Ghi rõ trong README và màn hình demo ("For research/education only").

## 2. Giới hạn kỹ thuật

| # | Giới hạn | Tác động | Cách giảm nhẹ |
|---|---|---|---|
| L1 | **Domain gap**: BDD100K/Cityscapes là đường Mỹ/Đức; VN nhiều xe máy, vạch kẻ/biển khác | AP motorcycle thấp; làn sai trên đường không vạch | Thu thập + gán nhãn dữ liệu VN, fine-tune; báo cáo cross-dataset |
| L2 | BDD100K chỉ có 1 class `traffic sign` | Không biết loại biển | Mở rộng sign classifier (GTSRB/VN) |
| L3 | Cityscapes **không có nhãn vạch kẻ** | Không dùng cho kênh lane marking | Chỉ dùng cho detection + drivable tham khảo |
| L4 | Vạch kẻ rất mỏng | IoU thấp, mất khi resize | Resize INTER_AREA + ngưỡng thấp; ảnh lớn hơn; loss Tversky/Focal |
| L5 | Không có tốc độ/khoảng cách thật | FCW chỉ là proxy | Ghi rõ; (mở rộng) calibration hoặc depth |
| L6 | `direct drivable` BDD có thể bao nhiều làn khi không có vạch | LDW sai lệch | `max_width_ratio`, fallback; đánh giá định tính |
| L7 | Camera đặt lệch / khác góc nhìn | `offset_ratio` lệch | `camera_offset_ratio`, tham số ROI |
| L8 | Đêm, mưa, chói đèn | Giảm chất lượng cả 2 model | Augmentation, phân tích theo điều kiện |
| L9 | Đường cong gắt, ngã tư | ROI dải ngang kém chính xác | Mở rộng fit đa thức |
| L10 | Màu đèn bằng HSV | Nhầm khi đèn LED chói/hậu xe | Voting theo track, ROI hợp lý; (mở rộng) classifier |
| L11 | Hiệu năng Colab T4 / máy cá nhân | Thời gian train dài, FPS thấp | Subset, FP16, `seg_every_n_frames` |
| L12 | Nhãn BDD có nhiễu | mAP bị giới hạn | Ghi nhận trong phân tích lỗi |

## 3. Rủi ro dự án

| Rủi ro | Xác suất | Ảnh hưởng | Giảm nhẹ |
|---|---|---|---|
| Tải BDD100K chậm / thiếu gói nhãn | Trung bình | Chậm M1 | Bắt đầu tải sớm; dùng subset; kiểm tra gói nhãn lane/drivable ngay |
| Định dạng mask khác giả định | Trung bình | Sai nhãn seg | Bắt buộc inspect (M1/M2), không đoán |
| AI agent sinh code lệch thiết kế | Cao | Bug tinh vi | `14_AGENT_RULES`, test, milestone nhỏ |
| API Ultralytics/smp đổi theo version | Trung bình | Lỗi chạy | Khoá version sau khi chạy ổn; kiểm tra bằng chạy thử |
| Colab ngắt phiên | Cao | Mất tiến độ | Checkpoint mỗi epoch lên Drive, `--resume` |
| Hết VRAM | Trung bình | Không train được | Giảm batch/imgsz, AMP, model nhỏ hơn |
| Alert giả nhiều (nhấp nháy) | Cao | Demo xấu | Debounce, hysteresis, EMA, cooldown, tinh chỉnh ngưỡng |

## 4. Đạo đức & quyền riêng tư
- Dữ liệu dashcam có thể chứa khuôn mặt/biển số. Khi tự thu thập: làm mờ trước khi công bố.
- Tôn trọng giấy phép dataset (nghiên cứu/phi thương mại); không phân phối lại dữ liệu.
- Trích dẫn BDD100K, Cityscapes, Ultralytics, smp trong báo cáo.

## 5. Hướng phát triển
1. Dữ liệu Việt Nam + fine-tune.
2. Sign classifier (loại biển, tốc độ).
3. Làn cong, solid/dashed, white/yellow, cảnh báo đè vạch liền.
4. Ước lượng khoảng cách/depth; fusion IMU/GPS nếu có.
5. Tối ưu triển khai: ONNX/TensorRT, Jetson.
6. Thử multi-task (YOLOP/HybridNets) và Transformer segmenter (SegFormer).
