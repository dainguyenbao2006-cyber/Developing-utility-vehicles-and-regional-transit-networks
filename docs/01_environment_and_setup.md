# 01 — Thiết lập Môi trường & Quy trình Làm việc

## 1. Yêu cầu hệ thống & Công cụ

| Thành phần | Yêu cầu | Ghi chú |
| :--- | :--- | :--- |
| **Hệ điều hành (Dev/Inference)** | Windows 10/11 | Chạy IDE VS Code, phát triển code & chạy inference demo |
| **Môi trường Train (Cloud)** | Google Colab (GPU T4 16GB VRAM) | Phục vụ huấn luyện YOLO11 và DeepLabV3+ |
| **Python** | $\ge 3.11$ | Quản lý qua `.python-version` và `.venv` |
| **Package Manager** | `uv` hoặc `pip` | Đồng bộ theo `pyproject.toml` / `requirements.txt` |
| **Linter & Test** | `ruff`, `pytest` | Bắt buộc sạch `ruff check` và pass test trước khi merge |

---

## 2. Cài đặt môi trường Local (Windows PowerShell)

### Bước 1: Tạo & kích hoạt Virtual Environment
Khuyến nghị sử dụng `uv` (nhanh, khớp `uv.lock` có sẵn trong repo):
```powershell
uv sync
.venv\Scripts\Activate.ps1
```

Hoặc sử dụng `venv` + `pip` truyền thống:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Bước 2: Kiểm tra cài đặt
```powershell
python --version
pytest --version
ruff --version
```

> [!NOTE]
> Khi viết script chạy DataLoader đa luồng (`num_workers > 0`) trên Windows, bắt buộc bọc entry point trong `if __name__ == "__main__":` để tránh lỗi spawn process.

---

## 3. Thiết lập Google Colab cho Huấn luyện

Trên `notebooks/colab_train.ipynb`:
1. Chọn Runtime: **GPU (T4)**.
2. Mount Google Drive để lưu checkpoints (`weights/`) và datasets tránh mất dữ liệu khi ngắt phiên.
3. Clone repo và cài các thư viện bổ sung nếu thiếu (`ultralytics`, `segmentation-models-pytorch`, `albumentations`).
4. Không hard-code đường dẫn Colab trong `src/`; truyền đường dẫn qua file YAML config hoặc biến ghi đè CLI.

---

## 4. Tải dữ liệu (BDD100K & Cityscapes)

### 4.1. Bộ dữ liệu BDD100K
* **Cách 1 (KaggleHub — đã có sẵn script mẫu):**
  Chạy `python data/dowload_data_bdd100k.py` (sử dụng `kagglehub.dataset_download("solesensei/solesensei_bdd100k")`). Lưu ý cấu hình Kaggle API token trước khi tải.
* **Cách 2 (Trang chính thức BDD100K):**
  Đăng ký tài khoản tại trang chủ BDD100K để tải bộ `100K Images`, `Detection 2020 Labels`, `Drivable Area`, và `Lane Markings`.

### 4.2. Bộ dữ liệu Cityscapes
* Đăng ký tài khoản tại trang chủ Cityscapes để tải `leftImg8bit_trainvaltest.zip` và `gtFine_trainvaltest.zip` (chủ yếu sử dụng tập `val` để đánh giá Cross-Dataset).
* Thư viện `cityscapesscripts` đã có sẵn trong `requirements.txt` để hỗ trợ đọc nhãn.

> [!IMPORTANT]
> Không lưu toàn bộ dataset gốc vào thư mục git được track. Đảm bảo thư mục dữ liệu lớn, file video và trọng số `.pt` đã được loại trừ trong `.gitignore`.

---

## 5. Quy tắc Git & Phối hợp Nhóm

1. **Nhánh (Branching Strategy):**
   * `main`: Nhánh ổn định, chỉ merge khi đã test và review.
   * `feature/<ten-module>`: Nhánh phát triển tính năng (ví dụ: `feature/adas-ldw`, `feature/yolo11-train`).
2. **Quy tắc Commit:**
   * Format gợi ý: `[<scope>] <mô tả ngắn>` (ví dụ: `[adas] implement LDW hysteresis logic`, `[data] add BDD100K mask inspector`).
   * **Cấm commit:** Dataset thô, file weights (`.pt`, `.pth`), video demo nặng (`> 20MB`), file cache tạm.
3. **Checklist trước khi tạo Pull Request:**
   * Chạy `ruff check .` không có lỗi/cảnh báo.
   * Chạy `pytest` pass 100% các test liên quan.
   * Cập nhật tài liệu `docs/` nếu có thay đổi schema dữ liệu, config key hoặc thứ tự class.
