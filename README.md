# ADAS-Vision — Phát hiện Phương tiện, Biển báo & Phân vùng Làn đường

Hệ thống hỗ trợ lái xe nâng cao (**ADAS**) sử dụng thị giác máy tính trên video dashcam, kết hợp **Object Detection (YOLO11)**, **Semantic Segmentation (DeepLabV3+)** và bộ **ADAS Logic Engine** thuần NumPy trên tập dữ liệu **BDD100K** và **Cityscapes**.

---

## Tính năng chính

* **Object Detection (YOLO11):** Phát hiện 9 lớp đối tượng giao thông (`car`, `truck`, `bus`, `motorcycle`, `bicycle`, `pedestrian`, `rider`, `traffic_light`, `traffic_sign`) kèm benchmark so sánh mô hình.
* **Semantic Segmentation (DeepLabV3+):** Phân vùng đa nhãn 3 kênh độc lập (`drivable_direct`, `drivable_alternative`, `lane_marking`) với Combo Loss (BCE + Dice).
* **ADAS Logic Engine (Thuần NumPy):**
  * **LDW (Lane Departure Warning):** Cảnh báo đè vạch / lệch làn đường kèm temporal hysteresis.
  * **FCW Proxy (Forward Collision Warning):** Cảnh báo nguy cơ va chạm phía trước dựa trên Ego Corridor và tốc độ phình bounding box ($\Delta\text{Area}/\Delta t$).
  * **PCW (Pedestrian Collision Warning):** Cảnh báo người đi bộ / xe thô sơ trong hành lang di chuyển.
* **Demo & Trực quan hóa:** CLI xử lý video/webcam và Web App **Streamlit** hiển thị HUD overlay kèm Event Log CSV.

---

## Cài đặt nhanh (Windows / Python $\ge$ 3.11)

Khuyến nghị dùng `uv` (đã có `uv.lock`):
```powershell
uv sync
.venv\Scripts\Activate.ps1
```

Hoặc dùng `venv` + `pip`:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Sao chép file mẫu biến môi trường nếu cần:
```powershell
copy .env.example .env
```

---

## Cấu trúc thư mục

```text
.
├── app/            # Giao diện Web Demo (Streamlit)
├── configs/        # Cấu hình YAML (detector, segmenter, adas_rules, paths)
├── data/           # Script tải & dữ liệu mẫu / splits
├── docs/           # Toàn bộ tài liệu đặc tả, thiết kế & phân công nhóm
├── notebooks/      # Notebook huấn luyện trên Google Colab T4
├── scripts/        # Script kiểm tra dữ liệu, train và evaluate
├── src/            # Mã nguồn chính (adas/, data/, models/, utils/, visualization/)
├── tests/          # Unit tests & Integration tests (pytest)
└── weights/        # Trọng số checkpoint (.pt — không commit lên Git)
```

---

## Tài liệu Dự án (`docs/`)

1. [`docs/project_description.md`](docs/project_description.md) — Tài liệu Define & Design tổng quan
2. [`docs/01_environment_and_setup.md`](docs/01_environment_and_setup.md) — Cài đặt môi trường, tải dữ liệu & quy tắc Git
3. [`docs/02_architecture_and_configs.md`](docs/02_architecture_and_configs.md) — Kiến trúc chi tiết, Data Contracts & cấu hình YAML
4. [`docs/03_dataset_and_bdd100k_spec.md`](docs/03_dataset_and_bdd100k_spec.md) — Đặc tả BDD100K, Cityscapes & pipeline tiền xử lý
5. [`docs/04_detection_yolo11_design.md`](docs/04_detection_yolo11_design.md) — Thiết kế & huấn luyện Object Detection (YOLO11)
6. [`docs/05_segmentation_deeplabv3plus_design.md`](docs/05_segmentation_deeplabv3plus_design.md) — Thiết kế & huấn luyện Segmentation (DeepLabV3+)
7. [`docs/06_adas_logic_design.md`](docs/06_adas_logic_design.md) — Thiết kế thuật toán ADAS thuần NumPy
8. [`docs/07_inference_visualization_and_app.md`](docs/07_inference_visualization_and_app.md) — Pipeline suy luận, HUD & Streamlit App
9. [`docs/08_testing_and_evaluation_plan.md`](docs/08_testing_and_evaluation_plan.md) — Kế hoạch kiểm thử (`pytest`) & đánh giá mô hình
10. [`docs/09_milestones_and_risk_management.md`](docs/09_milestones_and_risk_management.md) — Lộ trình Milestone (M1–M7) & quản lý rủi ro
11. [`docs/10_team_task_allocation.md`](docs/10_team_task_allocation.md) — Phân chia nhiệm vụ cho 6 thành viên

---

## Kiểm tra chất lượng code

```powershell
ruff check .
pytest
```
