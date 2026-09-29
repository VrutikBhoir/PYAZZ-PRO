from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path
from threading import Lock
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import yaml
from PIL import Image

from grader import OnionInput, decide
from size_estimator import (
    onion_diameter_mm,
    pixels_per_mm_from_reference,
    px_diameter_from_bbox,
)


ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.yaml"
MODEL_PATH = ROOT / "model" / "best.pt"
CONDITION_BY_CLASS = {
    "healthy_onion": "Healthy",
    "rotten_onion": "Rotten",
    "sprouted_onion": "Sprouted",
    "damaged_onion": "Damaged",
}

st.set_page_config(
    page_title="Onion Quality | Inspection Station",
    page_icon="OQ",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
    :root {
      --ink: #172b31; --muted: #607176; --paper: #f4f6f2; --white: #ffffff;
      --line: #dce4df; --teal: #197b78; --teal-dark: #125b5a;
      --leaf: #317b50; --gold: #d59a38; --red: #b5473d;
    }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    [data-testid="stHeader"] { background: rgba(244,246,242,.92); }
    [data-testid="stSidebar"] { background: #e9efeb; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-family: 'Manrope', sans-serif; }
    .block-container { max-width: 1440px; padding-top: 1.6rem; padding-bottom: 3rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--ink); letter-spacing: 0 !important; }
    .masthead { display:flex; justify-content:space-between; align-items:center; gap:20px; padding: 0 0 22px; border-bottom:1px solid var(--line); }
    .eyebrow { color:var(--teal); font:500 11px 'DM Mono',monospace; letter-spacing:1.2px; text-transform:uppercase; }
    .title { margin:5px 0 0; font:800 30px 'Manrope',sans-serif; letter-spacing:0; }
    .subline { color:var(--muted); font-size:14px; margin-top:5px; }
    .brand-mark { width:44px; height:44px; border-radius:12px; display:grid; place-items:center; background:var(--teal); color:white; font:800 18px 'Manrope',sans-serif; }
    .head-left { display:flex; align-items:center; gap:14px; }
    .section-label { margin:24px 0 11px; display:flex; align-items:center; gap:10px; font:700 14px 'Manrope',sans-serif; }
    .section-label:after { content:''; height:1px; background:var(--line); flex:1; }
    .panel { background:var(--white); border:1px solid var(--line); border-radius:6px; padding:18px; }
    .capture-panel { min-height:280px; }
    .hint { color:var(--muted); font-size:12px; line-height:1.5; }
    .metric { padding:15px 16px; border-left:3px solid var(--metric-color); background:#fff; border-top:1px solid var(--line); border-right:1px solid var(--line); border-bottom:1px solid var(--line); border-radius:3px; min-height:88px; }
    .metric-label { color:var(--muted); font:500 10px 'DM Mono',monospace; text-transform:uppercase; }
    .metric-value { margin-top:7px; font:700 20px 'Manrope',sans-serif; color:var(--ink); }
    .result-card { border:1px solid var(--line); background:#fff; border-radius:5px; padding:16px; margin-bottom:12px; }
    .result-head { display:flex; align-items:center; justify-content:space-between; gap:12px; padding-bottom:12px; border-bottom:1px solid var(--line); }
    .result-name { font:700 16px 'Manrope',sans-serif; }
    .decision { font:500 10px 'DM Mono',monospace; padding:7px 9px; border-radius:3px; white-space:nowrap; }
    .decision-yes { color:#21643c; background:#eaf5ec; }
    .decision-no { color:#9c3d35; background:#faeeec; }
    .result-grid { display:grid; grid-template-columns:repeat(6,1fr); gap:10px; padding-top:12px; }
    .result-stat-label { color:var(--muted); font:500 9px 'DM Mono',monospace; text-transform:uppercase; }
    .result-stat-value { margin-top:4px; font-weight:600; font-size:13px; }
    .reason { color:#8e4038; font-size:12px; margin:10px 0 0; }
    .calibration { border-left:3px solid var(--gold); background:#fbf6ea; padding:10px 12px; color:#71551d; font-size:11px; line-height:1.45; }
    div.stButton > button[kind="primary"] { background:var(--teal); border-color:var(--teal); color:white; font-weight:700; min-height:44px; }
    div.stButton > button[kind="primary"]:hover { background:var(--teal-dark); border-color:var(--teal-dark); }
    [data-testid="stFileUploader"] { border:1px dashed #aabbb3; border-radius:5px; background:#fbfcfa; }
    @media(max-width:600px) { .block-container { padding:1rem .8rem 2rem; } .title { font-size:24px; } .masthead { align-items:flex-start; } .result-grid { grid-template-columns:repeat(2,1fr); } }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def create_warmed_model():
    from ultralytics import YOLO

    model = YOLO(str(MODEL_PATH))
    model.predict(
        np.zeros((480, 640, 3), dtype=np.uint8),
        imgsz=640,
        verbose=False,
    )
    return model


@st.cache_resource
def get_model_executor():
    return ThreadPoolExecutor(max_workers=1)


@st.cache_resource
def load_model_async() -> Future:
    return get_model_executor().submit(create_warmed_model)


def load_model():
    return load_model_async().result()


def analyze_image(
    image: Image.Image,
    config: dict,
    confidence_threshold: float,
    calibration_confirmed: bool,
    image_size: int = 640,
    annotate: bool = True,
):
    model = load_model()
    prediction = model.predict(
        image,
        conf=confidence_threshold,
        imgsz=image_size,
        verbose=False,
    )[0]
    annotated = (
        Image.fromarray(prediction.plot()[:, :, ::-1]) if annotate else image
    )
    detections = []
    boxes = prediction.boxes
    if boxes is None or len(boxes) == 0:
        return annotated, detections, any(
            name == "sprout" for name in prediction.names.values()
        ), []

    raw = []
    for box in boxes:
        class_id = int(box.cls.item())
        label = prediction.names[class_id].lower()
        coords = box.xyxy[0].tolist()
        raw.append({
            "label": label,
            "confidence": float(box.conf.item()),
            "coords": coords,
        })

    onion_boxes = [item for item in raw if item["label"] in CONDITION_BY_CLASS]
    sprout_boxes = [item for item in raw if item["label"] == "sprout"]
    ppm = float(config["calibration"]["pixels_per_mm"])
    for item in onion_boxes:
        x1, y1, x2, y2 = item["coords"]
        diameter = (
            onion_diameter_mm(
                px_diameter_from_bbox(abs(x2 - x1), abs(y2 - y1)), ppm
            )
            if calibration_confirmed
            else None
        )
        has_sprout = any(
            max(x1, sprout["coords"][0]) < min(x2, sprout["coords"][2])
            and max(y1, sprout["coords"][1]) < min(y2, sprout["coords"][3])
            for sprout in sprout_boxes
        )
        condition = CONDITION_BY_CLASS[item["label"]]
        result = decide(
            OnionInput(
                diameter_mm=diameter,
                condition=condition,
                sprout=has_sprout,
                confidence=item["confidence"],
            ),
            config,
        )
        detections.append(result)
    return annotated, detections, bool(sprout_boxes), raw


class LiveInspectionState:
    def __init__(self):
        self._lock = Lock()
        self._results = None
        self._unmatched_sprout = False
        self._updated_at = None
        self._error = None
        self._status = "Starting camera..."
        self._boxes = []
        self._inference_future = None
        self._last_inference_started = 0.0
        self._session_records = []
        self._session_unmatched_sprout = False

    def publish(self, results, unmatched_sprout: bool, boxes: list) -> None:
        with self._lock:
            self._results = results
            self._unmatched_sprout = unmatched_sprout
            self._session_unmatched_sprout |= unmatched_sprout
            self._boxes = boxes
            now = time.monotonic()
            onion_boxes = [
                box for box in boxes if box["label"] in CONDITION_BY_CLASS
            ]
            for result, box in zip(results, onion_boxes):
                best_record = None
                best_overlap = 0.25
                for record in self._session_records:
                    if now - record["last_seen"] > 3.0:
                        continue
                    overlap = self._intersection_over_union(
                        record["coords"], box["coords"]
                    )
                    if overlap > best_overlap:
                        best_record = record
                        best_overlap = overlap
                if best_record is None:
                    self._session_records.append({
                        "result": result,
                        "coords": box["coords"],
                        "last_seen": now,
                    })
                else:
                    best_record.update(
                        result=result,
                        coords=box["coords"],
                        last_seen=now,
                    )
            self._updated_at = datetime.now().strftime("%H:%M:%S")
            self._error = None
            self._status = None

    @staticmethod
    def _intersection_over_union(first: list, second: list) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
        second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def schedule_inference(self, executor, callback, interval: float = 0.15) -> None:
        now = time.monotonic()
        with self._lock:
            if self._inference_future and not self._inference_future.done():
                return
            if now - self._last_inference_started < interval:
                return
            self._last_inference_started = now
            self._inference_future = executor.submit(callback)

    def current_boxes(self) -> list:
        with self._lock:
            return list(self._boxes)

    def finish_session(self) -> list:
        with self._lock:
            return {
                "results": [record["result"] for record in self._session_records],
                "unmatched_sprout": self._session_unmatched_sprout,
            }

    def reset_session(self) -> None:
        with self._lock:
            self._session_records.clear()
            self._results = None
            self._boxes = []
            self._updated_at = None
            self._unmatched_sprout = False
            self._session_unmatched_sprout = False
            self._error = None
            self._status = "Starting camera..."

    def publish_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def publish_error(self, error: str) -> None:
        with self._lock:
            self._error = error

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "results": self._results,
                "unmatched_sprout": self._unmatched_sprout,
                "updated_at": self._updated_at,
                "error": self._error,
                "status": self._status,
            }


def make_video_frame_callback(
    live_state: LiveInspectionState,
    config: dict,
    confidence_threshold: float,
    calibration_confirmed: bool,
    model_future: Future,
):
    def process_frame(frame: av.VideoFrame) -> av.VideoFrame:
        if not model_future.done():
            live_state.publish_status("Preparing AI detector in background...")
            return frame
        if model_future.exception() is not None:
            live_state.publish_error(str(model_future.exception()))
            return frame
        try:
            image_array = frame.to_ndarray(format="rgb24")

            def run_inference() -> None:
                try:
                    _, detections, unmatched_sprout, boxes = analyze_image(
                        Image.fromarray(image_array.copy()),
                        config,
                        confidence_threshold,
                        calibration_confirmed,
                        annotate=False,
                    )
                    live_state.publish(detections, unmatched_sprout, boxes)
                except Exception as error:
                    live_state.publish_error(str(error))

            live_state.schedule_inference(get_model_executor(), run_inference)
            annotated_frame = image_array.copy()
            for box in live_state.current_boxes():
                x1, y1, x2, y2 = (int(round(value)) for value in box["coords"])
                color = (48, 145, 89) if box["label"] in CONDITION_BY_CLASS else (226, 151, 51)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{box['label'].replace('_', ' ')} {box['confidence']:.0%}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            return av.VideoFrame.from_ndarray(annotated_frame, format="rgb24")
        except Exception as error:
            live_state.publish_error(str(error))
            return frame

    return process_frame


def make_qr_report_text(results: list, config: dict, generated_at: str) -> str:
    chosen = sum(result.decision == "CHOOSE" for result in results)
    calibration = (
        f"{config['calibration']['pixels_per_mm']} pixels/mm"
        if config["calibration"].get("confirmed", False)
        else "Not confirmed; sizes are unknown"
    )
    lines = [
        "ONION QUALITY PASSPORT",
        f"Inspection: {generated_at}",
        f"Camera calibration: {calibration}",
        f"Onions inspected: {len(results)}",
        f"CHOOSE: {chosen}",
        f"DO NOT CHOOSE: {len(results) - chosen}",
        "",
        "INDIVIDUAL RESULTS",
    ]
    for index, result in enumerate(results, start=1):
        size = f"{result.diameter_mm:.1f} mm" if result.diameter_mm is not None else "Unknown"
        result_line = (
            f"{index}. {result.condition} | {size} | {result.size_grade} | "
                f"{result.quality_grade} ({result.quality_score}/100) | {result.decision}"
        )
        if sum(len(line) + 1 for line in lines) + len(result_line) > 2000:
            lines.append(f"More results are included in the PDF: {len(results) - index + 1}.")
            break
        lines.append(result_line)
    return "\n".join(lines)


def make_quality_report(results: list, config: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Image as PdfImage
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    import qrcode

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    qr = qrcode.make(make_qr_report_text(results, config, generated_at))
    qr_buffer = BytesIO()
    qr.save(qr_buffer, format="PNG")
    qr_buffer.seek(0)

    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=letter, title="Onion Quality Passport")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("ONION QUALITY PASSPORT", styles["Title"]),
        Paragraph(f"Inspection report · {generated_at}", styles["Normal"]),
        Spacer(1, 16),
        Paragraph(
            (
                f"Camera calibration: {config['calibration']['pixels_per_mm']} pixels/mm"
                if config["calibration"].get("confirmed", False)
                else "Camera calibration: not confirmed; size-based choices are blocked"
            ),
            styles["Normal"],
        ),
        Spacer(1, 14),
    ]
    rows = [["#", "Condition", "Diameter", "Size grade", "Quality grade", "Score", "Decision"]]
    for index, result in enumerate(results, start=1):
        size = f"{result.diameter_mm:.1f} mm" if result.diameter_mm is not None else "Unknown"
        rows.append([
            str(index), result.condition, size, result.size_grade,
            result.quality_grade, f"{result.quality_score}/100", result.decision,
        ])
    table = Table(
        rows,
        colWidths=[22, 76, 62, 65, 72, 55, 105],
        repeatRows=1,
        hAlign="LEFT",
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#197b78")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dce4df")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f2")]),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.extend([table, Spacer(1, 18), PdfImage(qr_buffer, width=94, height=94)])
    story.append(Paragraph("Scan for a readable summary of this inspection.", styles["Normal"]))
    document.build(story)
    return output.getvalue()


def render_result(result, index: int) -> None:
    decision_class = "decision-yes" if result.decision == "CHOOSE" else "decision-no"
    decision_label = "CHOOSE" if result.decision == "CHOOSE" else "DO NOT CHOOSE"
    size = f"{result.diameter_mm:.1f} mm" if result.diameter_mm is not None else "Unknown"
    reason = " · ".join(result.reasons) if result.reasons else "Meets current quality rules"
    st.markdown(
        f"""
        <div class="result-card">
          <div class="result-head">
            <div class="result-name">Onion {index:02d}</div>
            <div class="decision {decision_class}">{decision_label}</div>
          </div>
          <div class="result-grid">
            <div><div class="result-stat-label">Condition</div><div class="result-stat-value">{result.condition}</div></div>
            <div><div class="result-stat-label">Diameter</div><div class="result-stat-value">{size}</div></div>
            <div><div class="result-stat-label">Size grade</div><div class="result-stat-value">{result.size_grade}</div></div>
            <div><div class="result-stat-label">Quality grade</div><div class="result-stat-value">{result.quality_grade}</div></div>
            <div><div class="result-stat-label">Confidence</div><div class="result-stat-value">{result.confidence:.0%}</div></div>
            <div><div class="result-stat-label">Quality score</div><div class="result-stat-value">{result.quality_score}/100</div></div>
          </div>
          <div class="reason">{reason}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every=0.75)
def render_live_results(live_state: LiveInspectionState) -> None:
    snapshot = live_state.snapshot()
    if snapshot["error"]:
        st.error(f"Live detection error: {snapshot['error']}")
        return
    if snapshot["updated_at"] is None:
        st.info(snapshot["status"] or "Waiting for live camera frames...")
        return
    if snapshot["unmatched_sprout"]:
        st.warning("A sprout was detected without an overlapping whole-onion box.")
    results = snapshot["results"] or []
    st.caption(f"Live prediction · updated {snapshot['updated_at']}")
    if not results:
        st.info("No whole onions detected in the current frame.")
        return
    for index, result in enumerate(results, start=1):
        render_result(result, index)


def render_inspection_report(
    results: list,
    config: dict,
    unmatched_sprout: bool = False,
) -> None:
    st.markdown("<div class='section-label'>Inspection report</div>", unsafe_allow_html=True)
    if not config["calibration"].get("confirmed", False):
        st.warning(
            "Size and grade are Unknown because camera calibration is not confirmed. "
            "Measure a ruler, confirm pixels/mm, and inspect again."
        )
    if unmatched_sprout:
        st.warning("A sprout was detected without an overlapping whole-onion box.")
    if not results:
        st.info("No whole onions were detected in this inspection.")
    else:
        choose_count = sum(result.decision == "CHOOSE" for result in results)
        metric_cols = st.columns(3)
        metrics = [
            ("ONIONS INSPECTED", str(len(results)), "#197b78"),
            ("MEET CURRENT RULES", str(choose_count), "#317b50"),
            ("REVIEW / REJECT", str(len(results) - choose_count), "#b5473d"),
        ]
        for column, (label, value, color) in zip(metric_cols, metrics):
            with column:
                st.markdown(
                    f"<div class='metric' style='--metric-color:{color}'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>",
                    unsafe_allow_html=True,
                )
        st.write("")
        for index, result in enumerate(results, start=1):
            render_result(result, index)
    try:
        pdf_bytes = make_quality_report(results, config)
        st.download_button(
            "Download quality passport · PDF + QR",
            data=pdf_bytes,
            file_name="onion-quality-passport.pdf",
            mime="application/pdf",
            type="primary",
            key=f"quality_report_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        )
    except ImportError:
        st.warning("Install report dependencies from requirements.txt to enable PDF + QR export.")


config = load_config()
config_signature = (
    float(config["calibration"]["pixels_per_mm"]),
    bool(config["calibration"].get("confirmed", False)),
    float(config["size"]["min_accept_mm"]),
    float(config["size"]["max_accept_mm"]),
    float(config["grading"].get("min_confidence", 0.30)),
)
if st.session_state.get("settings_config_signature") != config_signature:
    st.session_state["pixels_per_mm"] = config_signature[0]
    st.session_state["min_accept_mm"] = config_signature[2]
    st.session_state["max_accept_mm"] = config_signature[3]
    st.session_state["min_confidence"] = config_signature[4]
    st.session_state["confirmed_pixels_per_mm"] = (
        config_signature[0] if config_signature[1] else None
    )
    st.session_state["settings_config_signature"] = config_signature
st.session_state.setdefault("inspection_results", [])
st.session_state.setdefault("annotated_image", None)
st.session_state.setdefault("annotated_uploads", [])
st.session_state.setdefault("unmatched_sprout", False)
st.session_state.setdefault("live_camera_active", True)
st.session_state.setdefault("live_report_results", None)
if not hasattr(st.session_state.get("live_inspection_state"), "finish_session"):
    st.session_state["live_inspection_state"] = LiveInspectionState()

config["calibration"]["pixels_per_mm"] = float(st.session_state["pixels_per_mm"])
config["size"]["min_accept_mm"] = float(st.session_state["min_accept_mm"])
config["size"]["max_accept_mm"] = float(st.session_state["max_accept_mm"])
config["grading"]["min_confidence"] = float(st.session_state["min_confidence"])

with st.sidebar:
    st.markdown("<div class='eyebrow'>Inspection settings</div>", unsafe_allow_html=True)
    st.markdown("## Camera & standards")
    st.caption("Live detection uses the browser's selected camera. Allow camera access when prompted.")
    with st.expander("Fixed camera ruler calibration"):
        ruler_length_mm = st.number_input(
            "Known ruler length (mm)",
            min_value=0.1,
            value=50.0,
            step=1.0,
            key="ruler_length_mm",
        )
        ruler_length_px = st.number_input(
            "That length in the camera image (pixels)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="ruler_length_px",
        )
        if ruler_length_px > 0:
            calculated_pixels_per_mm = pixels_per_mm_from_reference(
                ruler_length_px, ruler_length_mm
            )
            st.caption(f"Calculated calibration: {calculated_pixels_per_mm:.2f} px/mm")
        else:
            st.caption("Measure the same ruler span in a frame from this fixed camera.")
        if st.button("Apply and save fixed calibration", use_container_width=True):
            if ruler_length_px <= 0:
                st.error("Enter the measured ruler length in pixels first.")
            else:
                measured_pixels_per_mm = pixels_per_mm_from_reference(
                    ruler_length_px, ruler_length_mm
                )
                st.session_state["pixels_per_mm"] = measured_pixels_per_mm
                st.session_state["confirmed_pixels_per_mm"] = measured_pixels_per_mm
                config["calibration"]["pixels_per_mm"] = measured_pixels_per_mm
                config["calibration"]["confirmed"] = True
                config["calibration"]["calibrated_at"] = datetime.now().date().isoformat()
                CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                st.rerun()
    pixels_per_mm = st.number_input(
        "Camera calibration · pixels per mm",
        min_value=0.1,
        max_value=100.0,
        step=0.05,
        format="%.2f",
        help="Measure a ruler in-frame: pixels per mm = measured pixel length / real mm length.",
        key="pixels_per_mm",
    )
    min_accept = st.number_input("Minimum accepted diameter (mm)", 1.0, 500.0, key="min_accept_mm")
    max_accept = st.number_input("Maximum accepted diameter (mm)", 1.0, 500.0, key="max_accept_mm")
    min_confidence = st.slider(
        "Minimum decision confidence",
        min_value=0.0,
        max_value=1.0,
        step=0.05,
        key="min_confidence",
        help="Detections below this threshold are rejected by the rules engine.",
    )
    if st.button("Confirm camera calibration", use_container_width=True):
        st.session_state["confirmed_pixels_per_mm"] = float(pixels_per_mm)
        config["calibration"]["pixels_per_mm"] = float(pixels_per_mm)
        config["calibration"]["confirmed"] = True
        config["calibration"]["calibrated_at"] = datetime.now().date().isoformat()
        CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    calibration_confirmed = (
        st.session_state["confirmed_pixels_per_mm"] is not None
        and abs(
            float(st.session_state["confirmed_pixels_per_mm"])
            - float(pixels_per_mm)
        ) < 0.001
    )
    if st.button("Save settings to config.yaml", use_container_width=True):
        if min_accept >= max_accept:
            st.error("Minimum size must be lower than maximum size.")
        else:
            config["calibration"]["pixels_per_mm"] = float(pixels_per_mm)
            config["calibration"]["confirmed"] = calibration_confirmed
            if calibration_confirmed:
                config["calibration"]["calibrated_at"] = datetime.now().date().isoformat()
            config["size"]["min_accept_mm"] = float(min_accept)
            config["size"]["max_accept_mm"] = float(max_accept)
            config["grading"]["min_confidence"] = float(min_confidence)
            CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            st.success("Settings saved.")
    if not calibration_confirmed:
        st.markdown(
            "<div class='calibration'><b>Calibration not confirmed.</b> Measure a ruler with this camera, set pixels/mm, then confirm. Size-based CHOOSE decisions stay blocked until confirmed.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.success(f"Calibration active · {pixels_per_mm:.2f} px/mm")

st.markdown(
    """
    <div class="masthead">
      <div class="head-left"><div class="brand-mark">OQ</div><div>
        <div class="eyebrow">Field inspection · quality control</div>
        <div class="title">Onion Quality Station</div>
        <div class="subline">Camera-led grading, backed by your trained detector and market rules.</div>
      </div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='section-label'>Capture & inspect</div>", unsafe_allow_html=True)
capture_col, output_col = st.columns([0.92, 1.08], gap="large")
with capture_col:
    st.markdown("<div class='panel capture-panel'>", unsafe_allow_html=True)
    source = st.radio(
        "Image source",
        ["Live camera", "Upload"],
        horizontal=True,
        label_visibility="collapsed",
        key="image_source",
    )
    if source == "Live camera":
        if st.session_state["live_camera_active"]:
            model_future = load_model_async()
            live_config = dict(config)
            live_config["calibration"] = {
                **config["calibration"],
                "pixels_per_mm": float(pixels_per_mm),
            }
            live_config["size"] = {
                **config["size"],
                "min_accept_mm": float(min_accept),
                "max_accept_mm": float(max_accept),
            }
            live_config["grading"] = {
                **config["grading"],
                "min_confidence": float(min_confidence),
            }
            webrtc_streamer(
                key="onion-live-camera",
                video_frame_callback=make_video_frame_callback(
                    st.session_state["live_inspection_state"],
                    live_config,
                    max(0.15, float(min_confidence) * 0.5),
                    calibration_confirmed,
                    model_future,
                ),
                media_stream_constraints={
                    "video": {"frameRate": {"ideal": 15, "max": 20}},
                    "audio": False,
                },
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                desired_playing_state=True,
                async_processing=False,
                media_toggle_controls=False,
                video_html_attrs={
                    "autoPlay": True,
                    "controls": False,
                    "muted": True,
                    "playsInline": True,
                    "style": {
                        "width": "100%",
                        "height": "auto",
                        "objectFit": "contain",
                    },
                },
            )
            st.markdown(
                "<div class='hint'>Allow camera access for continuous, on-screen detection. "
                "Frames are analyzed live and are not recorded or saved.</div>",
                unsafe_allow_html=True,
            )
            if st.button("Stop camera & create report", type="primary", use_container_width=True):
                session_report = st.session_state["live_inspection_state"].finish_session()
                st.session_state["live_report_results"] = session_report["results"]
                st.session_state["live_report_unmatched_sprout"] = session_report["unmatched_sprout"]
                st.session_state["live_camera_active"] = False
                st.rerun()
        else:
            st.success("Camera stopped. Session report is ready.")
            if st.button("Start new live inspection", use_container_width=True):
                st.session_state["live_inspection_state"].reset_session()
                st.session_state["live_report_results"] = None
                st.session_state["live_camera_active"] = True
                st.rerun()
        uploaded_files = []
        run_analysis = False
    else:
        uploaded_files = st.file_uploader(
            "Choose onion images",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )
        run_analysis = st.button(
            "Inspect selected images",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with output_col:
    if source == "Live camera":
        if st.session_state["live_camera_active"]:
            render_live_results(st.session_state["live_inspection_state"])
        elif st.session_state["live_report_results"] is not None:
            report_config = dict(config)
            report_config["calibration"] = {
                **config["calibration"],
                "pixels_per_mm": float(pixels_per_mm),
                "confirmed": calibration_confirmed,
            }
            render_inspection_report(
                st.session_state["live_report_results"],
                report_config,
                st.session_state.get("live_report_unmatched_sprout", False),
            )
    else:
        annotated_uploads = st.session_state.get("annotated_uploads", [])
        if run_analysis and uploaded_files:
            if min_accept >= max_accept:
                st.error("Minimum accepted diameter must be lower than maximum accepted diameter.")
            else:
                active_config = dict(config)
                active_config["calibration"] = {**config["calibration"], "pixels_per_mm": float(pixels_per_mm)}
                active_config["size"] = {
                    **config["size"],
                    "min_accept_mm": float(min_accept),
                    "max_accept_mm": float(max_accept),
                }
                active_config["grading"] = {
                    **config["grading"],
                    "min_confidence": float(min_confidence),
                }
                try:
                    batch_results = []
                    batch_annotated = []
                    unmatched_sprout = False
                    with st.spinner(f"Inspecting {len(uploaded_files)} images..."):
                        for uploaded_file in uploaded_files:
                            source_image = Image.open(
                                BytesIO(uploaded_file.getvalue())
                            ).convert("RGB")
                            annotated, detections, image_unmatched_sprout, _ = analyze_image(
                                source_image,
                                active_config,
                                max(0.15, float(min_confidence) * 0.5),
                                calibration_confirmed,
                            )
                            batch_annotated.append((uploaded_file.name, annotated))
                            batch_results.extend(detections)
                            unmatched_sprout |= image_unmatched_sprout
                    st.session_state["annotated_uploads"] = batch_annotated
                    st.session_state["inspection_results"] = batch_results
                    st.session_state["unmatched_sprout"] = unmatched_sprout
                    st.session_state["inspected_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
                except Exception as error:
                    st.error(f"Batch inspection failed: {error}")
                    annotated_uploads = []
        if annotated_uploads:
            for filename, annotated_image in annotated_uploads:
                st.image(
                    annotated_image,
                    caption=f"Detection result · {filename}",
                    use_container_width=True,
                )
        else:
            st.markdown(
                "<div class='panel capture-panel' style='display:grid;place-items:center;text-align:center'><div><div class='eyebrow'>Awaiting upload</div><div style='font:700 18px Manrope;margin:9px 0'>Your inspection appears here</div><div class='hint'>Upload a photo and run the quality inspection.</div></div></div>",
                unsafe_allow_html=True,
            )

results = st.session_state.get("inspection_results", [])
if source == "Upload" and st.session_state.get("inspected_at"):
    report_config = dict(config)
    report_config["calibration"] = {
        **config["calibration"],
        "pixels_per_mm": float(pixels_per_mm),
        "confirmed": calibration_confirmed,
    }
    render_inspection_report(
        results,
        report_config,
        st.session_state.get("unmatched_sprout", False),
    )