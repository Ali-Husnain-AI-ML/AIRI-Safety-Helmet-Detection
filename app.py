from pathlib import Path
import hashlib
import io
import re
import subprocess
import tempfile

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AIRI | Safety Intelligence",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best.pt"

# The trained model has class IDs: 0=Person, 1=Helmet, 2=No Helmet.
# Person is drawn as an unlabelled outline only; analytics count PPE classes.
PERSON_CLASS_ID = 0
PPE_CLASS_IDS = [1, 2]
# Run inference for all trained classes so Person boxes can be drawn,
# but only PPE_CLASS_IDS are included in counts and analytics.
INFERENCE_CLASS_IDS = [PERSON_CLASS_ID, *PPE_CLASS_IDS]


# ============================================================
# 2. CUSTOM DESIGN
# ============================================================
st.html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
:root {
    --background: #0a101c;
    --surface: #111c2d;
    --border: #26364d;
    --text: #f1f5f9;
    --muted: #91a1b7;
    --cyan: #22c5dc;
    --red: #fb6474;
    --yellow: #f6c85f;
    --green: #39c98a;
}
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: var(--background); color: var(--text); }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1500px; padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stSidebar"] { background: #0e1929; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.3rem; }
.brand-lockup { display:flex; align-items:center; gap:12px; padding:12px 4px 23px; }
.brand-icon { width:46px; height:46px; border-radius:14px; display:flex; align-items:center; justify-content:center; background:linear-gradient(135deg,#22c5dc,#2875d9); color:#07121e; font-size:23px; font-weight:800; }
.brand-name { color:#f8fafc; font-size:23px; font-weight:800; letter-spacing:2px; line-height:1.1; }
.brand-subtitle { color:#8093aa; font-size:9px; font-weight:700; letter-spacing:1.6px; margin-top:5px; }
.sidebar-line { height:1px; background:var(--border); margin:8px 0 22px; }
.nav-heading { color:#8093aa; font-size:10px; font-weight:700; letter-spacing:2px; margin:0 0 12px 5px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius:10px; padding:9px 11px; }
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#1a2b42; }
[data-testid="stSidebar"] [data-testid="stRadio"] p { color:#cbd5e1; font-size:14px; font-weight:600; }
.sidebar-footer { margin-top:34px; padding:15px 5px; color:#8293a9; font-size:10px; font-weight:700; letter-spacing:1.4px; }
.footer-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--green); margin-right:9px; }
.eyebrow { color:var(--cyan); font-size:10px; font-weight:800; letter-spacing:2.2px; margin-bottom:9px; }
.page-title { color:#f8fafc; font-size:clamp(27px,3vw,37px); font-weight:800; letter-spacing:-1.2px; line-height:1.2; margin:0; }
.page-description { color:var(--muted); font-size:14px; margin-top:9px; margin-bottom:27px; }
.metric-card { background:linear-gradient(145deg,#17263c,#111c2d); border:1px solid var(--border); border-radius:15px; padding:19px 20px; min-height:132px; }
.metric-top { display:flex; align-items:center; justify-content:space-between; }
.metric-label { color:#9aabc0; font-size:11px; font-weight:700; letter-spacing:1px; }
.metric-icon { width:32px; height:32px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:15px; }
.metric-value { color:#f8fafc; font-size:32px; font-weight:800; margin-top:14px; line-height:1; }
.metric-caption { color:#8193aa; font-size:11px; margin-top:9px; }
.panel { background:#111c2d; border:1px solid var(--border); border-radius:15px; padding:20px; height:100%; }
.panel-title { color:#f1f5f9; font-size:16px; font-weight:700; margin-bottom:5px; }
.panel-description { color:#91a1b7; font-size:12px; margin-bottom:17px; }
.info-card { background:#15243a; border:1px solid #263b55; border-radius:12px; padding:17px; margin-bottom:12px; }
.info-card-title { color:#f1f5f9; font-size:14px; font-weight:700; margin-bottom:6px; }
.info-card-text { color:#9aabc0; font-size:12px; line-height:1.6; }
.legend-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:8px; }
.empty-state { background:#111c2d; border:1px dashed #354963; border-radius:15px; padding:35px 22px; text-align:center; }
.empty-title { color:#f1f5f9; font-size:17px; font-weight:700; margin-bottom:8px; }
.empty-text { color:#91a1b7; font-size:13px; }
.stButton > button { background:linear-gradient(135deg,#22c5dc,#16a8c6); color:#06121d; border:0; border-radius:10px; min-height:43px; font-weight:800; }
.stButton > button:hover { background:#5ce0ee; color:#06121d; border:0; }
.stDownloadButton > button { background:#17283e; color:#dce8f5; border:1px solid #35516e; border-radius:10px; min-height:42px; font-weight:700; }
.stDownloadButton > button:hover { background:#203752; color:white; border-color:var(--cyan); }
[data-testid="stFileUploader"] { background:#111c2d; border:1px dashed #405773; border-radius:13px; padding:12px; }
[data-testid="stFileUploader"] section { background:transparent; }
[data-testid="stExpander"] { background:#111c2d; border:1px solid var(--border); border-radius:11px; }
.stSlider label { color:#cbd5e1 !important; font-weight:600; }
.stAlert { border-radius:11px; }
div[data-testid="stImage"] img { border-radius:10px; }
hr { border-color:var(--border); }
</style>
"""
)


# ============================================================
# 3. LOAD MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    loaded = YOLO(str(MODEL_PATH))
    names = loaded.names
    expected = {0: "Person", 1: "Helmet", 2: "No Helmet"}
    for class_id, expected_name in expected.items():
        actual = names.get(class_id) if isinstance(names, dict) else names[class_id]
        if str(actual).strip().lower().replace("_", " ") != expected_name.lower():
            raise ValueError(
                f"Unexpected model class mapping for ID {class_id}: {actual!r}. "
                "Expected class IDs 0=Person, 1=Helmet and 2=No Helmet."
            )
    return loaded


try:
    with st.spinner("Loading AI detection model..."):
        model = load_model()
except Exception as exc:
    st.error("Unable to load the detection model.")
    st.code(str(exc))
    st.info("Keep best.pt in the same folder as app.py.")
    st.stop()


# ============================================================
# 4. SESSION STATE
# ============================================================
def init_state():
    defaults = {
        "image_result": None,
        "video_result": None,
        "video_preview": None,
        "video_preview_key": None,
        "latest_summary": None,
        "latest_summary_type": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()


# ============================================================
# 5. PPE CLASS / DETECTION HELPERS
# ============================================================
def empty_counts():
    return {"Helmet": 0, "No Helmet": 0}


def get_class_name(class_id):
    names = model.names
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id))
    return str(names[int(class_id)])


def normalize_class_name(name):
    return str(name).lower().replace("_", " ").replace("-", " ").strip()


def get_display_class(name):
    """Map the model class name; Person is visual-only and excluded from PPE counts."""
    normalized = normalize_class_name(name)
    if normalized == "person":
        return "Person"
    if normalized == "helmet":
        return "Helmet"
    if normalized in {"no helmet", "nohelmet", "without helmet"}:
        return "No Helmet"
    return None


def get_counts(result, minimum_confidence=0.0):
    counts = empty_counts()
    if result.boxes is None:
        return counts
    for box in result.boxes:
        score = float(box.conf[0].item())
        if score < minimum_confidence:
            continue
        class_id = int(box.cls[0].item())
        display_name = get_display_class(get_class_name(class_id))
        if display_name in counts:
            counts[display_name] += 1
    return counts


class _ArrayValue:
    """Small adapter so tiled NumPy detections work with drawing helpers."""
    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def item(self):
        return self.value.item()


class _DetectionBox:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = [_ArrayValue(np.asarray(xyxy, dtype=np.float32))]
        self.conf = [_ArrayValue(np.asarray(confidence, dtype=np.float32))]
        self.cls = [_ArrayValue(np.asarray(class_id, dtype=np.float32))]


class _DetectionBoxes(list):
    id = None


class _DetectionResult:
    def __init__(self, detections):
        self.boxes = _DetectionBoxes(
            _DetectionBox(d["xyxy"], d["conf"], d["cls"])
            for d in detections
        )


def _tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    step = max(1, int(tile_size * (1.0 - overlap)))
    last_start = length - tile_size
    starts = list(range(0, last_start + 1, step))
    if starts[-1] != last_start:
        starts.append(last_start)
    return sorted(set(starts))


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _intersection_over_smaller(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    smaller = min(area_a, area_b)
    return intersection / smaller if smaller > 0 else 0.0


def _deduplicate_tiled_detections(detections, iou_threshold=0.50, ios_threshold=0.92):
    """Remove repeated Helmet/No Helmet boxes from overlapping image crops."""
    kept = []
    class_ids = sorted({int(d["cls"]) for d in detections})
    for class_id in class_ids:
        class_dets = [d for d in detections if int(d["cls"]) == class_id]
        class_dets.sort(key=lambda d: float(d["conf"]), reverse=True)
        selected = []
        while class_dets:
            best = class_dets.pop(0)
            selected.append(best)
            remaining = []
            for candidate in class_dets:
                overlap = _box_iou(best["xyxy"], candidate["xyxy"])
                containment = _intersection_over_smaller(best["xyxy"], candidate["xyxy"])
                if overlap < iou_threshold and containment < ios_threshold:
                    remaining.append(candidate)
            class_dets = remaining
        kept.extend(selected)
    return kept


def predict_tiled_image(image_rgb, image_size=960, tile_size=640, confidence=0.25, nms_iou=0.60):
    """Detect Person and PPE on overlapping crops; analytics remain PPE-only."""
    height, width = image_rgb.shape[:2]
    overlap = 0.25
    xs = _tile_starts(width, tile_size, overlap)
    ys = _tile_starts(height, tile_size, overlap)
    detections = []
    tile_number = 0

    for y0 in ys:
        for x0 in xs:
            x1 = min(width, x0 + tile_size)
            y1 = min(height, y0 + tile_size)
            crop = image_rgb[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            tile_number += 1
            tile_result = model.predict(
                source=crop,
                classes=INFERENCE_CLASS_IDS,
                conf=confidence,
                imgsz=image_size,
                iou=nms_iou,
                max_det=300,
                augment=False,
                verbose=False,
            )[0]

            if tile_result.boxes is None:
                continue

            for box in tile_result.boxes:
                score = float(box.conf[0].item())
                class_id = int(box.cls[0].item())
                coords = box.xyxy[0].cpu().numpy().astype(float)
                gx1 = max(0.0, min(float(width), coords[0] + x0))
                gy1 = max(0.0, min(float(height), coords[1] + y0))
                gx2 = max(0.0, min(float(width), coords[2] + x0))
                gy2 = max(0.0, min(float(height), coords[3] + y0))
                if gx2 <= gx1 or gy2 <= gy1:
                    continue
                detections.append({
                    "xyxy": [gx1, gy1, gx2, gy2],
                    "conf": score,
                    "cls": class_id,
                })

    merged = _deduplicate_tiled_detections(detections)
    return _DetectionResult(merged), tile_number


def draw_custom_boxes(frame, result, minimum_confidence=0.0):
    """Draw Person outlines without labels/counts; label PPE detections."""
    output = frame.copy()
    colors = {
        "Person": (155, 165, 175),
        "Helmet": (0, 210, 255),
        "No Helmet": (65, 65, 245),
    }
    boxes = result.boxes
    if boxes is None:
        return output

    for box in boxes:
        confidence = float(box.conf[0].item())
        if confidence < minimum_confidence:
            continue

        class_id = int(box.cls[0].item())
        label = get_display_class(get_class_name(class_id))
        if label not in colors:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        color = colors[label]

        # Person: draw a clean outline only. Do not print Person text,
        # counts, cards, or analytics anywhere in the application.
        if label == "Person":
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            continue

        # PPE detections remain fully labelled and counted.
        text = f"{label} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, 0.55, 2)
        label_top = max(0, y1 - th - baseline - 10)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(output, (x1, label_top), (x1 + tw + 12, y1), color, -1)
        cv2.putText(
            output,
            text,
            (x1 + 6, max(th + 2, y1 - 7)),
            font,
            0.55,
            (8, 15, 25),
            2,
            cv2.LINE_AA,
        )
    return output


# ============================================================
# 6. UI HELPERS
# ============================================================
def render_page_header(eyebrow, title, description):
    st.html(
        f'<div class="eyebrow">{eyebrow}</div>'
        f'<div class="page-title">{title}</div>'
        f'<div class="page-description">{description}</div>'
    )


def render_metric_card(label, value, caption, icon, accent):
    st.html(
        f'''<div class="metric-card">
            <div class="metric-top">
                <div class="metric-label">{label}</div>
                <div class="metric-icon" style="background:{accent}20;color:{accent};">{icon}</div>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>'''
    )


def render_detection_metrics(counts, caption):
    cols = st.columns(2, gap="medium")
    cards = [
        ("HELMET", counts["Helmet"], "✓", "#f6c85f"),
        ("NO HELMET", counts["No Helmet"], "!", "#fb6474"),
    ]
    for col, (label, value, icon, accent) in zip(cols, cards):
        with col:
            render_metric_card(label, value, caption, icon, accent)


def make_chart(counts):
    data = pd.DataFrame({"Class": list(counts.keys()), "Detections": list(counts.values())})
    return data.set_index("Class")


# ============================================================
# 7. FFMPEG / BROWSER-COMPATIBLE MP4 HELPERS
# ============================================================
def get_ffmpeg_encoder_candidates():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    encoder_text = (result.stdout or "") + (result.stderr or "")
    preferred = ["libx264", "h264_nvenc", "libopenh264", "h264_mf"]
    return [name for name in preferred if re.search(rf"\b{re.escape(name)}\b", encoder_text)]


def encode_h264_mp4(input_path, output_path, audio_source=None, keep_input_audio=True):
    """Encode an MP4 with H.264/yuv420p for browser playback."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    encoders = get_ffmpeg_encoder_candidates()
    if not encoders:
        raise RuntimeError(
            "The bundled FFmpeg does not expose an H.264 encoder. "
            "Install a full FFmpeg build and add it to Windows PATH, then restart the app."
        )

    errors = []
    for encoder in encoders:
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(input_path)]
        if audio_source is not None:
            command += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0?"]
        elif keep_input_audio:
            command += ["-map", "0:v:0", "-map", "0:a:0?"]
        else:
            command += ["-map", "0:v:0", "-an"]

        command += ["-c:v", encoder]
        if encoder == "libx264":
            command += ["-preset", "veryfast", "-crf", "23"]
        elif encoder == "h264_nvenc":
            command += ["-preset", "p4", "-cq", "24"]
        else:
            command += ["-b:v", "4M"]

        command += [
            "-pix_fmt", "yuv420p",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-movflags", "+faststart",
        ]
        if audio_source is not None or keep_input_audio:
            command += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
        command.append(str(output_path))

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return

        detail = (result.stderr or "Unknown FFmpeg error")[-1200:]
        errors.append(f"{encoder}: {detail}")
        if output_path.exists():
            output_path.unlink()

    raise RuntimeError("H.264 encoding failed.\n" + "\n".join(errors))


def prepare_browser_preview(video_bytes, extension):
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / f"original{extension or '.video'}"
        preview_path = Path(temp_dir) / "preview_h264.mp4"
        source_path.write_bytes(video_bytes)
        encode_h264_mp4(source_path, preview_path, keep_input_audio=True)
        return preview_path.read_bytes()


# ============================================================
# 8. VIDEO PROCESSING — PPE CLASSES ONLY
# ============================================================
def process_video(video_bytes, confidence, video_imgsz, nms_iou, progress_bar, status):
    """Draw unlabelled Person outlines and PPE labels; return PPE-only statistics."""
    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = Path(temp_dir) / "source_video.mp4"
        raw_output_path = Path(temp_dir) / "processed_raw.mp4"
        final_output_path = Path(temp_dir) / "processed_h264.mp4"
        source_path.write_bytes(video_bytes)

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise RuntimeError("OpenCV could not open this video file.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0:
            cap.release()
            raise RuntimeError("The uploaded video has invalid dimensions.")
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        writer = cv2.VideoWriter(
            str(raw_output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("OpenCV could not create the temporary video file.")

        peak_helmet_detections = 0
        peak_no_helmet_detections = 0
        processed_frames = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                prediction = model.predict(
                    source=frame,
                    classes=INFERENCE_CLASS_IDS,
                    conf=confidence,
                    iou=nms_iou,
                    imgsz=video_imgsz,
                    max_det=300,
                    verbose=False,
                )[0]

                frame_counts = get_counts(prediction, confidence)
                peak_helmet_detections = max(peak_helmet_detections, frame_counts["Helmet"])
                peak_no_helmet_detections = max(peak_no_helmet_detections, frame_counts["No Helmet"])

                annotated = draw_custom_boxes(frame, prediction, confidence)
                writer.write(annotated)

                processed_frames += 1
                if total_frames > 0:
                    progress_bar.progress(min(processed_frames / total_frames, 1.0))
                status.text(
                    f"Processing frame {processed_frames}"
                    + (f" of {total_frames}" if total_frames > 0 else "")
                )
        finally:
            cap.release()
            writer.release()

        if processed_frames == 0:
            raise RuntimeError("No readable frames were found in this video.")
        if not raw_output_path.exists() or raw_output_path.stat().st_size == 0:
            raise RuntimeError("The temporary processed video was not created.")

        encode_h264_mp4(
            raw_output_path,
            final_output_path,
            audio_source=source_path,
            keep_input_audio=False,
        )

        final_counts = {
            "Helmet": peak_helmet_detections,
            "No Helmet": peak_no_helmet_detections,
        }
        return final_output_path.read_bytes(), final_counts, processed_frames


# ============================================================
# 9. SIDEBAR
# ============================================================
with st.sidebar:
    st.html(
        """
        <div class="brand-lockup">
            <div class="brand-icon">A</div>
            <div>
                <div class="brand-name">AIRI</div>
                <div class="brand-subtitle">SAFETY INTELLIGENCE</div>
            </div>
        </div>
        <div class="sidebar-line"></div>
        <div class="nav-heading">WORKSPACE</div>
        """
    )

    page = st.radio(
        "Navigation",
        ["Dashboard", "Image Detection", "Video Detection", "About Model"],
        label_visibility="collapsed",
    )

    st.html('<div class="sidebar-line"></div>')
    with st.expander("Detection Preferences"):
        confidence = st.slider(
            "Detection confidence",
            min_value=0.10,
            max_value=0.80,
            value=0.25,
            step=0.05,
            help="Higher values filter weak detections; lower values may show more detections and false positives.",
        )
        nms_iou = st.slider(
            "Overlapping box separation (NMS IoU)",
            min_value=0.45,
            max_value=0.90,
            value=0.75,
            step=0.05,
            help="Higher values suppress fewer overlapping boxes. Too high can retain duplicates.",
        )
        image_imgsz = st.select_slider(
            "Image detection resolution",
            options=[640, 768, 960, 1280],
            value=960,
        )
        video_imgsz = st.select_slider(
            "Video detection resolution",
            options=[640, 768, 960, 1280],
            value=960,
        )
        tile_size = st.select_slider(
            "Image crop size",
            options=[480, 512, 640, 768],
            value=640,
            help="Overlapping crops can help detect smaller PPE objects; smaller crops take longer.",
        )

    st.html(
        '<div class="sidebar-footer"><span class="footer-dot"></span>AI SAFETY MONITORING</div>'
    )


# ============================================================
# 10. DASHBOARD PAGE
# ============================================================
if page == "Dashboard":
    render_page_header(
        "AIRI / OVERVIEW",
        "Safety Monitoring",
        "Helmet and no-helmet detection activity, with subtle worker outlines for context.",
    )

    latest = st.session_state.latest_summary
    if latest is None:
        shown_counts = {"Helmet": "—", "No Helmet": "—"}
        caption = "No completed analysis yet"
    else:
        shown_counts = latest
        caption = f"Latest analysis · {st.session_state.latest_summary_type}"

    render_detection_metrics(shown_counts, caption)
    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        st.html(
            """
            <div class="panel">
                <div class="panel-title">Your monitoring workspace</div>
                <div class="panel-description">Choose the type of media you want to analyze.</div>
                <div class="info-card">
                    <div class="info-card-title">01 &nbsp; Image analysis</div>
                    <div class="info-card-text">Upload a construction-site image and inspect Helmet and No Helmet detections.</div>
                </div>
                <div class="info-card">
                    <div class="info-card-title">02 &nbsp; Video analysis</div>
                    <div class="info-card-text">Preview video in the browser and watch the processed PPE detection result on this page.</div>
                </div>
                <div class="info-card" style="margin-bottom:0;">
                    <div class="info-card-title">03 &nbsp; Detection insights</div>
                    <div class="info-card-text">Review helmet-related detections from your latest analysis.</div>
                </div>
            </div>
            """
        )

    with right:
        st.html(
            """
            <div class="panel">
                <div class="panel-title">Detection classes</div>
                <div class="panel-description">The analytics report helmet-related classes only.</div>
                <div style="margin-top:22px;line-height:3;">
                    <div style="color:#dce5f0;font-size:13px;"><span class="legend-dot" style="background:#f6c85f;"></span>Helmet</div>
                    <div style="color:#dce5f0;font-size:13px;"><span class="legend-dot" style="background:#fb6474;"></span>No Helmet</div>
                </div>
                <div style="border-top:1px solid #26364d;margin-top:18px;padding-top:17px;">
                    <div style="color:#91a1b7;font-size:11px;">DETECTION ENGINE</div>
                    <div style="color:#f1f5f9;font-size:17px;font-weight:700;margin-top:5px;">YOLOv8</div>
                </div>
            </div>
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "Person boxes appear as unlabelled grey outlines only. "
        "Only Helmet and No Helmet are counted in the displayed analytics."
    )


# ============================================================
# 11. IMAGE DETECTION PAGE
# ============================================================
elif page == "Image Detection":
    render_page_header(
        "AIRI / IMAGE ANALYSIS",
        "PPE Detection",
        "Upload an image to detect PPE, with subtle worker outlines for visual context.",
    )

    uploaded_image = st.file_uploader(
        "Upload construction-site image",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_image is None:
        st.html(
            """
            <div class="empty-state">
                <div style="font-size:36px;margin-bottom:12px;">▧</div>
                <div class="empty-title">Ready for image analysis</div>
                <div class="empty-text">Choose a JPG, JPEG or PNG image using the uploader above.</div>
            </div>
            """
        )
    else:
        image_bytes = uploaded_image.getvalue()
        image_key = hashlib.sha256(image_bytes).hexdigest() + f":{confidence:.2f}:{image_imgsz}:{nms_iou:.2f}:{tile_size}:ppe_only_v1"

        try:
            original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            st.error("This image could not be opened. Please upload another image.")
            st.stop()

        st.html('<div class="panel-title">Image preview</div>')
        st.image(original_image, caption=uploaded_image.name, use_container_width=True)

        if st.button("Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Detecting Helmet and No Helmet..."):
                rgb_array = np.array(original_image)
                prediction, crop_count = predict_tiled_image(
                    image_rgb=rgb_array,
                    image_size=image_imgsz,
                    tile_size=tile_size,
                    confidence=confidence,
                    nms_iou=nms_iou,
                )
                bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                result_image = draw_custom_boxes(bgr_array, prediction, confidence)
                result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                counts = get_counts(prediction, confidence)

                st.session_state.image_result = {
                    "key": image_key,
                    "image": result_rgb,
                    "counts": counts,
                    "crop_count": crop_count,
                }
                st.session_state.latest_summary = counts
                st.session_state.latest_summary_type = "Image"

        saved_result = st.session_state.image_result
        if saved_result is not None and saved_result["key"] == image_key:
            st.markdown("---")
            render_page_header(
                "ANALYSIS COMPLETE",
                "Detection Results",
                "Worker outlines are shown for visual context; only helmet-related results appear in analytics.",
            )

            original_col, result_col = st.columns(2, gap="medium")
            with original_col:
                st.html('<div class="panel-title">Original image</div>')
                st.image(original_image, use_container_width=True)
            with result_col:
                st.html('<div class="panel-title">PPE detection output</div>')
                st.image(saved_result["image"], use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.html('<div class="panel-title">Detection summary</div>')
            render_detection_metrics(saved_result["counts"], "Detections in this image")
            st.caption(f"Image analyzed in {saved_result['crop_count']} overlapping crop(s).")

            chart_col, details_col = st.columns([1, 1], gap="large")
            with chart_col:
                st.html('<div class="panel-title">Class distribution</div>')
                st.bar_chart(make_chart(saved_result["counts"]), color="#22c5dc")
            with details_col:
                st.html('<div class="panel-title">Detection breakdown</div>')
                total = max(sum(saved_result["counts"].values()), 1)
                for name, value in saved_result["counts"].items():
                    st.write(f"**{name}**")
                    st.progress(value / total)
                    st.caption(f"{value} detected")

            output_image = Image.fromarray(saved_result["image"])
            output_buffer = io.BytesIO()
            output_image.save(output_buffer, format="PNG")
            st.download_button(
                "Download annotated image",
                data=output_buffer.getvalue(),
                file_name="AIRI_PPE_Detection.png",
                mime="image/png",
                use_container_width=True,
            )


# ============================================================
# 12. VIDEO DETECTION PAGE
# ============================================================
elif page == "Video Detection":
    render_page_header(
        "AIRI / VIDEO ANALYSIS",
        "Video PPE Detection",
        "Preview the video and process Helmet and No Helmet detections.",
    )

    uploaded_video = st.file_uploader(
        "Upload construction-site video",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        key="video_uploader",
    )

    if uploaded_video is None:
        st.html(
            """
            <div class="empty-state">
                <div style="font-size:36px;margin-bottom:12px;">▶</div>
                <div class="empty-title">Ready for video analysis</div>
                <div class="empty-text">Upload an MP4, MOV, AVI, MKV or WebM video to begin.</div>
            </div>
            """
        )
    else:
        original_video_bytes = uploaded_video.getvalue()
        extension = Path(uploaded_video.name).suffix.lower() or ".mp4"
        video_key = hashlib.sha256(original_video_bytes).hexdigest()
        analysis_key = f"{video_key}:{confidence:.2f}:{video_imgsz}:{nms_iou:.2f}"

        if st.session_state.video_preview_key != video_key:
            st.session_state.video_preview = None
            st.session_state.video_result = None
            st.session_state.video_preview_key = video_key

        if st.session_state.video_preview is None:
            try:
                with st.spinner("Preparing browser-compatible video preview..."):
                    st.session_state.video_preview = prepare_browser_preview(
                        original_video_bytes,
                        extension,
                    )
            except Exception as exc:
                st.error("Could not prepare the browser video preview.")
                st.code(str(exc))
                st.info("Video preview conversion requires an H.264-capable FFmpeg encoder.")
                st.stop()

        preview_bytes = st.session_state.video_preview
        st.html(
            '<div class="panel-title">Original video preview</div>'
            '<div class="panel-description">Play the uploaded video here before running PPE detection.</div>'
        )
        st.video(preview_bytes, format="video/mp4", width="stretch")

        if st.button("Start PPE Video Analysis", type="primary", use_container_width=True):
            progress_bar = st.progress(0.0)
            status = st.empty()
            try:
                with st.spinner("Detecting Helmet and No Helmet in video frames..."):
                    processed_bytes, counts, frames = process_video(
                        preview_bytes,
                        confidence,
                        video_imgsz,
                        nms_iou,
                        progress_bar,
                        status,
                    )

                st.session_state.video_result = {
                    "key": analysis_key,
                    "video": processed_bytes,
                    "counts": counts,
                    "frames": frames,
                }
                st.session_state.latest_summary = counts
                st.session_state.latest_summary_type = "Video"
                status.success(f"Completed successfully. {frames} frames analyzed.")
            except Exception as exc:
                status.empty()
                st.error("Video processing failed.")
                st.code(str(exc))

        saved_video = st.session_state.video_result
        if saved_video is not None and saved_video["key"] == analysis_key:
            st.markdown("---")
            render_page_header(
                "ANALYSIS COMPLETE",
                "Processed Video",
                "The annotated H.264 video is playable directly on this webpage.",
            )
            st.video(saved_video["video"], format="video/mp4", width="stretch")
            st.download_button(
                "Download processed video (optional)",
                data=saved_video["video"],
                file_name="AIRI_Processed_PPE_Detection.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            cols = st.columns(2, gap="medium")
            with cols[0]:
                render_metric_card(
                    "PEAK HELMET DETECTIONS",
                    saved_video["counts"]["Helmet"],
                    "Maximum Helmet boxes in one frame",
                    "✓",
                    "#f6c85f",
                )
            with cols[1]:
                render_metric_card(
                    "PEAK NO-HELMET DETECTIONS",
                    saved_video["counts"]["No Helmet"],
                    "Maximum No Helmet boxes in one frame",
                    "!",
                    "#fb6474",
                )

            st.caption(
                f"Frames processed: {saved_video['frames']}. "
                "Video summary values are peak detections in one frame, not totals across all frames."
            )
            st.html('<div class="panel-title">Detection summary</div>')
            st.bar_chart(make_chart(saved_video["counts"]), color="#22c5dc")


# ============================================================
# 13. ABOUT MODEL PAGE
# ============================================================
elif page == "About Model":
    render_page_header(
        "AIRI / MODEL INFORMATION",
        "About the Detection System",
        "Technical overview of the AI-powered PPE detection system.",
    )

    info_left, info_right = st.columns([1, 1], gap="large")
    with info_left:
        st.html(
            """
            <div class="panel">
                <div class="panel-title">Model overview</div>
                <div class="panel-description">Object detection for construction-site PPE monitoring.</div>
                <div class="info-card"><div class="info-card-title">Architecture</div><div class="info-card-text">YOLOv8 object detection</div></div>
                <div class="info-card"><div class="info-card-title">Inference framework</div><div class="info-card-text">Ultralytics</div></div>
                <div class="info-card"><div class="info-card-title">Model weights</div><div class="info-card-text">best.pt</div></div>
                <div class="info-card" style="margin-bottom:0;"><div class="info-card-title">Supported inputs</div><div class="info-card-text">Images and video files</div></div>
            </div>
            """
        )
    with info_right:
        st.html(
            """
            <div class="panel">
                <div class="panel-title">Active detection classes</div>
                <div class="panel-description">The application filters inference to these PPE categories.</div>
                <div class="info-card"><div class="info-card-title"><span class="legend-dot" style="background:#f6c85f;"></span>Helmet</div><div class="info-card-text">Detects helmet objects.</div></div>
                <div class="info-card" style="margin-bottom:0;"><div class="info-card-title"><span class="legend-dot" style="background:#fb6474;"></span>No Helmet</div><div class="info-card-text">Detects objects classified as no helmet.</div></div>
            </div>
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.warning(
        "This app displays only Helmet and No Helmet predictions. "
        "AI detections can be missed or incorrect and do not replace human safety inspections."
    )
    st.caption("AIRI | Safety Helmet Detection")
