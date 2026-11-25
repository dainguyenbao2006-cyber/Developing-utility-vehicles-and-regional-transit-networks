# 12 — Thiết lập môi trường (Windows + VS Code + Colab)

## 1. Máy cá nhân (Windows)

Yêu cầu: Python 3.10 hoặc 3.11, Git, VS Code, (tuỳ chọn) GPU NVIDIA + driver mới, ffmpeg trong PATH.

```powershell
git clone <repo-url> adas-vision
cd adas-vision
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Cài PyTorch **theo hướng dẫn trên trang chính thức pytorch.org** (chọn đúng CUDA hoặc CPU). Ví dụ CUDA 12.1:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Sau đó:
```powershell
pip install -r requirements.txt
pip install -e .
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> Nếu PowerShell chặn script kích hoạt venv: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## 2. `requirements.txt` (khởi điểm — khoá phiên bản sau khi chạy ổn)

```text
ultralytics
segmentation-models-pytorch
albumentations
opencv-python
numpy
pandas
pyyaml
tqdm
pillow
matplotlib
scikit-learn
streamlit
pytest
ruff
tensorboard
cityscapesscripts      # chỉ cần cho convert Cityscapes
```
(torch/torchvision cài riêng như trên. Sau khi chạy được: `pip freeze > requirements.lock.txt`.)

## 3. `pyproject.toml` tối thiểu

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "adas"
version = "0.1.0"
requires-python = ">=3.10"

[project.scripts]
adas = "adas.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

## 4. VS Code

- Extension: Python, Pylance, Jupyter, Ruff, YAML, Markdown All in One.
- `.vscode/settings.json`:
```json
{
  "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"],
  "editor.formatOnSave": true,
  "[python]": {"editor.defaultFormatter": "charliermarsh.ruff"}
}
```
- `.vscode/launch.json` gợi ý: cấu hình chạy `python -m adas.cli run --source samples/test.mp4`.

## 5. Lưu ý riêng cho Windows
| Vấn đề | Cách xử lý |
|---|---|
| DataLoader `num_workers>0` lỗi / treo | Bảo vệ `if __name__ == "__main__":`; thử `num_workers=0` để debug; dùng 2–4 |
| Symlink cần quyền admin | Dùng hardlink (`os.link`) hoặc copy — đã có trong script |
| Đường dẫn dài > 260 ký tự | Đặt dự án gần gốc ổ đĩa (`D:\adas-vision`), bật long paths |
| Xuống dòng CRLF | `git config core.autocrlf true`; file `.sh` giữ LF |
| `cv2.imshow` không hiện | Cài `opencv-python` (không phải `-headless`) |
| Ultralytics tải weights lỗi | Tải tay `yolo11s.pt` về `weights/` và trỏ đường dẫn |

## 6. Google Colab
- Runtime → T4 GPU.
- Cài: `!pip -q install ultralytics segmentation-models-pytorch albumentations`.
- Colab đã có torch/opencv; **không** cài lại torch trừ khi cần.
- Đặt đường dẫn trong `configs/colab.yaml`. Xem `10` mục 2.

## 7. Kiểm tra môi trường (script `scripts/check_env.py`)
In: phiên bản Python, torch, CUDA, tên GPU, VRAM, phiên bản ultralytics/smp/opencv, ffmpeg có hay không, các đường dẫn dataset tồn tại hay không. Thoát mã ≠ 0 nếu thiếu thành phần bắt buộc.
