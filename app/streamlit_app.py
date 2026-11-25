"""Streamlit interactive dashboard for ADAS-Vision video inference and event inspection."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from adas.alert_manager import ADASAnalyzer  # noqa: E402
from models.detector import YOLO11Detector  # noqa: E402
from models.segmenter import DeepLabV3PlusSegmenter  # noqa: E402
from pipeline import ADASPipeline  # noqa: E402
from utils.config import ConfigNode, load_project_configs  # noqa: E402
from utils.types import FrameAnalysis  # noqa: E402
from visualization.hud import EventLogger, HUDVisualizer  # noqa: E402


@st.cache_resource
def load_cached_models(
    det_weights_str: str,
    seg_weights_str: str,
    img_size: int,
    seg_size_hw: tuple[int, int],
    seg_thresholds: tuple[float, ...],
    device: str,
) -> tuple[YOLO11Detector | None, DeepLabV3PlusSegmenter | None, list[str]]:
    """Load detector and segmenter once and cache across Streamlit reruns."""
    warnings: list[str] = []
    det_path = Path(det_weights_str)
    seg_path = Path(seg_weights_str)

    detector: YOLO11Detector | None = None
    if det_path.exists():
        detector = YOLO11Detector(
            weights_path=det_path,
            img_size=img_size,
            device=device,
        )
    else:
        warnings.append(
            f"Chưa tìm thấy trọng số Detector tại `{det_path}` (chạy chế độ không box)."
        )

    segmenter: DeepLabV3PlusSegmenter | None = None
    if seg_path.exists():
        segmenter = DeepLabV3PlusSegmenter(
            weights_path=seg_path,
            size_hw=seg_size_hw,
            thresholds=list(seg_thresholds),
            device=device,
        )
    else:
        warnings.append(
            f"Chưa tìm thấy trọng số Segmenter tại `{seg_path}` (chạy chế độ không mask)."
        )

    return detector, segmenter, warnings


def discover_sample_videos() -> list[Path]:
    """Find local .mp4/.avi sample videos in data/ or outputs/ directories."""
    candidates: list[Path] = []
    for folder in (_PROJECT_ROOT / "data", _PROJECT_ROOT / "outputs"):
        if folder.exists():
            for ext in ("*.mp4", "*.avi", "*.mov"):
                candidates.extend(sorted(folder.glob(ext)))
    return candidates


def render_sidebar(cfg: ConfigNode) -> dict[str, Any]:
    """Render sidebar controls and return user-selected parameters."""
    st.sidebar.header("⚙️ Cấu hình ADAS-Vision")

    source_mode = st.sidebar.radio(
        "Nguồn video đầu vào",
        options=["Upload video dashcam", "Chọn video mẫu có sẵn"],
    )

    uploaded_file = None
    selected_sample: Path | None = None
    if source_mode == "Upload video dashcam":
        uploaded_file = st.sidebar.file_uploader(
            "Chọn file video (.mp4, .avi, .mov)",
            type=["mp4", "avi", "mov", "mkv"],
        )
    else:
        samples = discover_sample_videos()
        if samples:
            chosen_str = st.sidebar.selectbox(
                "Video mẫu",
                options=[str(p.relative_to(_PROJECT_ROOT)) for p in samples],
            )
            selected_sample = _PROJECT_ROOT / chosen_str
        else:
            st.sidebar.info("Chưa có file video mẫu trong `data/` hoặc `outputs/`.")

    st.sidebar.subheader("🎨 Lớp hiển thị HUD")
    draw_boxes = st.sidebar.checkbox("Detection Boxes", value=True)
    draw_masks = st.sidebar.checkbox("Segmentation Mask", value=True)
    draw_corridor = st.sidebar.checkbox("Ego Corridor", value=True)
    draw_hud = st.sidebar.checkbox("HUD Status & Banner", value=True)

    st.sidebar.subheader("🎚️ Tinh chỉnh ngưỡng")
    conf_threshold = st.sidebar.slider(
        "Detector Confidence",
        min_value=0.10,
        max_value=0.90,
        value=float(cfg.detector.model.conf_threshold),
        step=0.05,
    )
    camera_offset = st.sidebar.slider(
        "LDW Camera Offset Ratio",
        min_value=-0.15,
        max_value=0.15,
        value=float(cfg.adas.ldw.camera_offset_ratio),
        step=0.01,
    )
    caution_area = st.sidebar.slider(
        "FCW Caution Area Ratio",
        min_value=0.01,
        max_value=0.15,
        value=float(cfg.adas.fcw.caution_area_ratio),
        step=0.01,
    )
    critical_area = st.sidebar.slider(
        "FCW Critical Area Ratio",
        min_value=max(0.05, caution_area + 0.01),
        max_value=0.30,
        value=max(float(cfg.adas.fcw.critical_area_ratio), caution_area + 0.02),
        step=0.01,
    )
    max_frames = st.sidebar.number_input(
        "Giới hạn số frame xử lý (0 = toàn bộ)",
        min_value=0,
        max_value=10000,
        value=300,
        step=50,
    )

    return {
        "uploaded_file": uploaded_file,
        "selected_sample": selected_sample,
        "draw_boxes": draw_boxes,
        "draw_masks": draw_masks,
        "draw_corridor": draw_corridor,
        "draw_hud": draw_hud,
        "conf_threshold": conf_threshold,
        "camera_offset": camera_offset,
        "caution_area": caution_area,
        "critical_area": critical_area,
        "max_frames": int(max_frames) if max_frames > 0 else None,
    }


def main() -> None:
    """Main Streamlit web application entry point."""
    st.set_page_config(page_title="ADAS-Vision Dashboard", layout="wide")
    st.title("🚗 ADAS-Vision — Phát hiện Phương tiện, Biển báo & Cảnh báo Làn đường")

    cfg = load_project_configs()
    params = render_sidebar(cfg)

    detector, segmenter, model_warnings = load_cached_models(
        det_weights_str=str(cfg.paths.detector_weights),
        seg_weights_str=str(cfg.paths.segmenter_weights),
        img_size=int(cfg.detector.model.img_size),
        seg_size_hw=tuple(cfg.segmenter.model.input_size_hw),
        seg_thresholds=tuple(cfg.segmenter.model.thresholds),
        device=str(cfg.project.device),
    )
    for msg in model_warnings:
        st.warning(msg)

    if detector is not None:
        detector.conf_threshold = float(params["conf_threshold"])

    tab_video, tab_stats = st.tabs(["🎥 Video Inference", "📊 Thống kê & Log Sự kiện"])

    if "last_run_result" not in st.session_state:
        st.session_state["last_run_result"] = None

    with tab_video:
        run_clicked = st.button("▶️ Bắt đầu phân tích ADAS", type="primary")
        frame_slot = st.empty()
        progress_text = st.empty()

        if run_clicked:
            input_path: Path | None = None
            temp_input_file: Path | None = None

            if params["uploaded_file"] is not None:
                suffix = Path(params["uploaded_file"].name).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(params["uploaded_file"].read())
                    temp_input_file = Path(tmp.name)
                input_path = temp_input_file
            elif params["selected_sample"] is not None:
                input_path = params["selected_sample"]

            if input_path is None or not input_path.exists():
                st.error(
                    "Vui lòng upload một file video hoặc chọn video mẫu hợp lệ trước khi chạy."
                )
            else:
                adas_dict = cfg.adas.to_dict()
                adas_dict["ldw"]["camera_offset_ratio"] = params["camera_offset"]
                adas_dict["fcw"]["caution_area_ratio"] = params["caution_area"]
                adas_dict["fcw"]["critical_area_ratio"] = params["critical_area"]

                analyzer = ADASAnalyzer.from_config(adas_dict)
                visualizer = HUDVisualizer.from_config(adas_dict)
                visualizer.draw_boxes = params["draw_boxes"]
                visualizer.draw_masks = params["draw_masks"]
                visualizer.draw_corridor = params["draw_corridor"]
                visualizer.draw_hud = params["draw_hud"]

                out_dir = Path(cfg.paths.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_video_path = out_dir / "streamlit_out.mp4"
                out_csv_path = out_dir / "streamlit_events.csv"
                event_logger = EventLogger(out_csv_path)

                pipeline = ADASPipeline(
                    detector=detector,
                    segmenter=segmenter,
                    analyzer=analyzer,
                    visualizer=visualizer,
                    event_logger=event_logger,
                )

                def _on_frame(
                    rendered_bgr: Any,
                    analysis: FrameAnalysis,
                    cur_fps: float,
                ) -> None:
                    if analysis.frame_idx % 2 == 0:
                        frame_slot.image(rendered_bgr, channels="BGR", width="stretch")
                        progress_text.caption(
                            f"Frame #{analysis.frame_idx} | FPS: {cur_fps:.1f} | "
                            f"LDW: {analysis.ldw_status} | FCW: {analysis.fcw_status}"
                        )

                summary = pipeline.run_source(
                    source=input_path,
                    output_video_path=out_video_path,
                    max_frames=params["max_frames"],
                    show_display=False,
                    frame_callback=_on_frame,
                )

                if temp_input_file is not None and temp_input_file.exists():
                    temp_input_file.unlink(missing_ok=True)

                st.session_state["last_run_result"] = {
                    "summary": summary,
                    "records": list(event_logger.records),
                }
                st.success(
                    f"Hoàn tất {summary.frames_processed} frames "
                    f"(Trung bình: {summary.avg_fps:.2f} FPS)."
                )

        last_res = st.session_state["last_run_result"]
        if last_res is not None:
            summary = last_res["summary"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Frames đã xử lý", summary.frames_processed)
            col2.metric("FPS trung bình", f"{summary.avg_fps:.2f}")
            col3.metric("Tổng cảnh báo ghi nhận", sum(summary.alert_counts.values()))

            if summary.fps_history:
                st.subheader("Biểu đồ FPS theo Frame")
                fps_df = pd.DataFrame({"FPS": summary.fps_history})
                st.line_chart(fps_df)

    with tab_stats:
        last_res = st.session_state["last_run_result"]
        if last_res is None:
            st.info("Hãy chạy phân tích video ở Tab 1 để xem thống kê và tải Event Log.")
        else:
            summary = last_res["summary"]
            records = last_res["records"]

            st.subheader("📦 Số lượng đối tượng phát hiện (lượt xuất hiện qua các frame)")
            if summary.class_counts:
                cls_df = pd.DataFrame(
                    [
                        {"Class": k, "Count": v}
                        for k, v in sorted(summary.class_counts.items(), key=lambda x: -x[1])
                    ]
                )
                st.dataframe(cls_df, width="stretch", hide_index=True)
            else:
                st.caption("Không có đối tượng nào được phát hiện.")

            st.subheader("⚠️ Nhật ký Sự kiện Cảnh báo (Event Log)")
            if records:
                events_df = pd.DataFrame(records)
                timeline_counts = (
                    events_df.groupby(["frame_idx", "alert_type"]).size().unstack(fill_value=0)
                )
                st.bar_chart(timeline_counts)
                st.dataframe(events_df, width="stretch", hide_index=True)
            else:
                st.caption("Không có sự kiện cảnh báo nào phát sinh trong phiên chạy này.")

            dl_col1, dl_col2 = st.columns(2)
            if summary.log_csv_path is not None and Path(summary.log_csv_path).exists():
                csv_bytes = Path(summary.log_csv_path).read_bytes()
                dl_col1.download_button(
                    label="📥 Download CSV Event Log",
                    data=csv_bytes,
                    file_name="adas_events.csv",
                    mime="text/csv",
                )
            if summary.output_video_path is not None and Path(summary.output_video_path).exists():
                vid_bytes = Path(summary.output_video_path).read_bytes()
                dl_col2.download_button(
                    label="📥 Download Output Video (.mp4)",
                    data=vid_bytes,
                    file_name="adas_demo_out.mp4",
                    mime="video/mp4",
                )


if __name__ == "__main__":
    main()
