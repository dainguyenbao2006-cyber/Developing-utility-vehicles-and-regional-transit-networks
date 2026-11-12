# 06 — Thiết kế Thuật toán Logic ADAS (`src/adas/`)

> [!IMPORTANT]
> **Ràng buộc kỹ thuật (`CLAUDE.md` rule #14):** Toàn bộ module trong `src/adas/` **chỉ được import `numpy` và thư viện chuẩn Python (stdlib)**. Tuyệt đối không import `torch`, `ultralytics`, `cv2` để đảm bảo module nhẹ, chạy tức thì và test độc lập 100%.

---

## 1. Tổng quan Đầu vào / Đầu ra

* **Đầu vào mỗi frame:**
  * `detections: list[Detection]` (tọa độ `xyxy` theo pixel frame gốc `(H, W)`).
  * `seg_mask: np.ndarray` shape `(3, H, W)` kiểu `uint8` $\{0, 1\}$ (hoặc `float32`).
  * `cfg: dict` (đọc từ `configs/adas_rules.yaml`).
* **Đầu ra:**
  * `FrameAnalysis`: trạng thái tổng hợp (`ldw_status`, `fcw_status`, danh sách `ADASAlert`).

---

## 2. Lane Departure Warning — LDW (`src/adas/ldw.py`)

### 2.1. Xác định điểm tham chiếu xe (Ego Reference Region)
* Vì camera gắn sau kính chắn gió nhìn thẳng về phía trước, tâm xe được xấp xỉ ở đáy khung hình:
  $$x_{\text{ego}} = W \cdot (0.5 + \text{camera\_offset\_ratio})$$
  $$y_{\text{ego}} = H \cdot \text{ego\_y\_ratio} \quad (\text{mặc định } 0.88)$$
* Tạo dải kiểm tra vệt bánh xe (Ego Wheel Strip) quanh `(x_ego, y_ego)`:
  * Chiều ngang: $[x_{\text{ego}} - w_{\text{half}}, \, x_{\text{ego}} + w_{\text{half}}]$, với $w_{\text{half}} = W \cdot \text{wheel\_half\_width\_ratio}$.
  * Chiều cao: băng ngang cao khoảng $2\% - 4\%$ chiều cao ảnh.

### 2.2. Quy tắc phát hiện vi phạm làn đường
Trích xuất các kênh từ `seg_mask`:
* `direct = seg_mask[0]` (`drivable_direct`)
* `alt = seg_mask[1]` (`drivable_alternative`)
* `lane = seg_mask[2]` (`lane_marking`)

Tính 3 chỉ số trong dải bánh xe (và tại vùng trung tâm mũi xe):
1. `lane_overlap_ratio`: Tỷ lệ pixel `lane_marking` nằm trong dải bánh xe trái/phải.
2. `direct_coverage`: Tỷ lệ pixel `drivable_direct` bao phủ tâm mũi xe.
3. `alt_coverage`: Tỷ lệ pixel `drivable_alternative` bao phủ tâm mũi xe.

**Phân loại trạng thái tức thời:**
* **`DEPARTED` (Đỏ):** Tâm mũi xe nằm ngoài `drivable_direct` (`direct_coverage < min_direct_cov`) VÀ rơi vào `drivable_alternative` hoặc đè mạnh qua vạch kẻ.
* **`WARNING` (Vàng):** Tâm xe vẫn trong `drivable_direct` nhưng mép bánh xe trái hoặc phải chạm/đè lên `lane_marking` (`lane_overlap_ratio >= lane_overlap_threshold`).
* **`NORMAL` (Xanh):** Xe nằm gọn trong `drivable_direct` và không đè vạch.

### 2.3. Chống nhiễu giật khung hình (Temporal Hysteresis)
* Sử dụng bộ đếm trạng thái qua `hysteresis_frames` frame liên tiếp (mặc định 3 frames) để tránh cảnh báo nhấp nháy khi mask vạch kẻ đường bị đứt đoạn trong 1-2 frame.

---

## 3. Forward Collision Warning — FCW Proxy (`src/adas/fcw.py`)

Do không có cảm biến Radar/LiDAR đo khoảng cách mét thực tế, FCW hoạt động theo **Proxy Hình học + Tốc độ thay đổi kích thước (TTC Proxy)**:

### 3.1. Xác định Hành lang Di chuyển (Ego Corridor)
* Tạo đa giác hình thang phía trước mũi xe (hoặc kết hợp trực tiếp giao với mask `drivable_direct`):
  * Đáy dưới tại $y = H$, rộng `corridor_bottom_width_ratio * W`.
  * Đỉnh trên tại $y = H \cdot \text{corridor\_top\_y\_ratio}$, rộng `corridor_top_width_ratio * W`.
* Chỉ xét các đối tượng thuộc nhóm phương tiện (`car`, `truck`, `bus`, `motorcycle`) có **điểm đáy giữa** $(x_{\text{mid}}, y_2)$ nằm trong Ego Corridor.

### 3.2. Tính toán mức độ nguy hiểm
Đối với xe gần nhất (có $y_2$ lớn nhất trong corridor):
1. **Tỷ lệ diện tích (Proximity Ratio):**
   $$R_{\text{area}} = \frac{(x_2 - x_1)(y_2 - y_1)}{W \cdot H}$$
2. **Tốc độ tiến gần (Growth Rate Proxy):**
   Lưu lịch sử $R_{\text{area}}$ của đối tượng dẫn đầu gần nhất trong `history_window` frame (ghép nối đơn giản theo độ lệch tọa độ tâm hoặc IoU giữa frame $t$ và $t-1$):
   $$\Delta_{\text{growth}} = \frac{R_{\text{area}}^{(t)} - R_{\text{area}}^{(t-k)}}{R_{\text{area}}^{(t-k)} + \epsilon}$$

**Phân cấp cảnh báo:**
* **`CRITICAL` (Đỏ):** $R_{\text{area}} \ge \text{critical\_area\_ratio}$ HOẶC ($R_{\text{area}} \ge \text{caution\_area\_ratio}$ và $\Delta_{\text{growth}} \ge \text{growth\_rate\_critical}$).
* **`CAUTION` (Vàng):** $R_{\text{area}} \ge \text{caution\_area\_ratio}$.
* **`SAFE` (Xanh):** Ngược lại.

---

## 4. Pedestrian Collision Warning & Traffic Awareness (`src/adas/pcw.py`, `tsr.py`)

### 4.1. Cảnh báo Người đi bộ / Xe thô sơ (PCW)
* Lọc các đối tượng thuộc `("pedestrian", "rider", "bicycle")`.
* Kiểm tra điều kiện nguy hiểm:
  1. Nằm ở nửa dưới khung hình: $y_2 \ge H \cdot \text{danger\_y\_min\_ratio}$, VÀ
  2. Điểm đáy $(x_{\text{mid}}, y_2)$ nằm trong vùng `drivable_direct` (hoặc giao với Ego Corridor vượt ngưỡng `in_lane_iou_threshold`).
* Khi thỏa mãn $\rightarrow$ Phát cảnh báo mức **`CRITICAL` ("NGƯỜI ĐI BỘ TRONG LÀN")**.

### 4.2. Nhận diện Biển báo & Đèn tín hiệu (TSR/TL)
* Trích xuất danh sách `traffic_sign` và `traffic_light` có độ tin cậy $\ge$ ngưỡng cấu hình và kích thước đủ lớn (tránh biển báo quá xa ở làn đối diện).
* Với `traffic_light`: Có thể bổ sung heuristic màu HSV đơn giản ở tầng visualizer/helper (tùy chọn ở milestone sau) hoặc hiển thị nhắc nhở có đèn giao thông/biển báo phía trước.

---

## 5. Quản lý Cảnh báo & Event Logger (`src/adas/alert_manager.py`)

* Tổng hợp toàn bộ cảnh báo từ LDW, FCW, PCW, TSR theo thứ tự ưu tiên độ nghiêm trọng:
  `CRITICAL` > `WARNING / CAUTION` > `INFO` > `SAFE`.
* Cung cấp cơ chế **cooldown** (ví dụ không ghi log lặp cùng một sự kiện trong 1 giây liên tiếp) để log CSV gọn nhẹ và dễ phân tích.
* **Schema Event Log (CSV):**
  `timestamp_sec, frame_idx, alert_type, severity, message, extra_info`
