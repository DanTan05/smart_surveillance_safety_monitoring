"""
Smart Surveillance & Safety Monitor
=====================================
CSCI435 – Computer Vision Algorithms and Systems
University of Wollongong in Dubai
"""

import os
import time
import tempfile

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from cv_modules.object_detection import ObjectDetector
from cv_modules.motion_detection import MotionDetector
from cv_modules.face_detection   import FaceDetector
from cv_modules.edge_detection   import EdgeDetector


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Surveillance Monitor | CSCI435",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

:root {
    --bg-primary:    #0d1117;
    --bg-secondary:  #161b22;
    --bg-card:       #1c2333;
    --border:        #30363d;
    --accent:        #2f81f7;
    --accent-dim:    #1a4a8a;
    --green:         #3fb950;
    --red:           #f85149;
    --orange:        #d29922;
    --text-primary:  #e6edf3;
    --text-secondary:#8b949e;
    --text-muted:    #484f58;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

.stApp { background-color: var(--bg-primary); color: var(--text-primary); }

[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

.sidebar-brand {
    padding: 1.5rem 0 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.sidebar-brand h1 {
    font-size: 1.05rem; font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--text-primary) !important; margin: 0;
}
.sidebar-brand p {
    font-size: 0.72rem; color: var(--text-secondary) !important;
    margin: 0.25rem 0 0; font-family: 'DM Mono', monospace;
}

.sidebar-section {
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-muted) !important; margin: 1.5rem 0 0.6rem;
}

[data-testid="stCheckbox"] label,
[data-testid="stRadio"] label {
    font-size: 0.88rem !important; color: var(--text-secondary) !important;
}
[data-testid="stCheckbox"] label:hover,
[data-testid="stRadio"] label:hover { color: var(--text-primary) !important; }

.page-header {
    padding: 2rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.page-header h1 {
    font-size: 1.6rem; font-weight: 600;
    color: var(--text-primary); margin: 0 0 0.25rem; letter-spacing: -0.02em;
}
.page-header p {
    font-size: 0.82rem; color: var(--text-secondary);
    margin: 0; font-family: 'DM Mono', monospace;
}

.pill-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 0.75rem; }
.pill {
    font-size: 0.72rem; font-family: 'DM Mono', monospace; font-weight: 500;
    padding: 3px 10px; border-radius: 4px; background: var(--accent-dim);
    color: #79c0ff; border: 1px solid #1f6feb; letter-spacing: 0.03em;
}
.pill-inactive { background: var(--bg-card); color: var(--text-muted); border-color: var(--border); }

.metric-grid { display: flex; gap: 12px; margin: 1.5rem 0 0.5rem; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 120px; background: var(--bg-card);
    border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.25rem;
}
.metric-card .label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.4rem;
}
.metric-card .value {
    font-size: 1.8rem; font-weight: 600;
    font-family: 'DM Mono', monospace; color: var(--text-primary); line-height: 1;
}
.metric-card .value.green  { color: var(--green); }
.metric-card .value.red    { color: var(--red); }
.metric-card .value.orange { color: var(--orange); }
.metric-card .value.blue   { color: var(--accent); }

.section-title {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.75rem;
}

.img-wrapper {
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 8px; overflow: hidden;
}
.img-label {
    font-size: 0.72rem; font-family: 'DM Mono', monospace;
    color: var(--text-muted); padding: 0.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
    letter-spacing: 0.05em; text-transform: uppercase;
}

.stButton > button {
    background-color: var(--accent) !important; color: white !important;
    border: none !important; border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    font-size: 0.88rem !important; padding: 0.5rem 1.5rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1px dashed var(--border) !important; border-radius: 8px !important;
}

[data-testid="stSlider"] label {
    font-size: 0.82rem !important; color: var(--text-secondary) !important;
}

[data-testid="stAlert"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important; color: var(--text-secondary) !important;
}

[data-testid="stProgressBar"] > div > div { background-color: var(--accent) !important; }

hr { border-color: var(--border) !important; }

.status-row {
    font-family: 'DM Mono', monospace; font-size: 0.8rem;
    color: var(--text-secondary); padding: 0.4rem 0;
}
.status-row span.ok   { color: var(--green); }
.status-row span.warn { color: var(--orange); }

.live-notice {
    background: var(--bg-card); border: 1px solid var(--border);
    border-left: 3px solid var(--accent); border-radius: 8px;
    padding: 1rem 1.25rem; font-size: 0.85rem;
    color: var(--text-secondary); line-height: 1.6; margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Hide sidebar collapse button via JavaScript ───────────────────────────────
components.html(
    """
    <script>
    (function() {
        function hideBtn() {
            var p = window.parent;
            var sels = [
                'button[data-testid="stSidebarCollapseButton"]',
                '[data-testid="stSidebarCollapseButton"]',
                'button[aria-label="Collapse sidebar"]',
                'button[aria-label="Close sidebar"]'
            ];
            sels.forEach(function(s) {
                p.document.querySelectorAll(s).forEach(function(el) {
                    el.style.setProperty('display', 'none', 'important');
                });
            });
        }
        hideBtn();
        setInterval(hideBtn, 300);
    })();
    </script>
    """,
    height=0,
    scrolling=False,
)


# ── WebRTC config ─────────────────────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})


# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    return {
        "object": ObjectDetector(),
        "face":   FaceDetector(),
        "edge":   EdgeDetector(),
    }

models = load_models()

if "motion_detector" not in st.session_state:
    st.session_state["motion_detector"] = MotionDetector()


# ── Core pipeline ─────────────────────────────────────────────────────────────
def process_frame(frame, *, detect, track, motion, face, edge, conf, t1, t2):
    output = frame.copy()
    stats  = {"objects": 0, "faces": 0, "motion": False}

    if edge:
        output = models["edge"].detect(output, t1, t2)
    if motion:
        output, _, stats["motion"] = st.session_state["motion_detector"].detect(output)
    if track:
        output, stats["objects"] = models["object"].track(output, conf)
    elif detect:
        output, stats["objects"] = models["object"].detect(output, conf)
    if face:
        output, stats["faces"] = models["face"].detect(output)

    return output, stats


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h1>Surveillance Monitor</h1>
        <p>CSCI435 · Computer Vision</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section">Input Mode</p>', unsafe_allow_html=True)
    mode = st.radio(
        "Input Mode",
        ["Image Upload", "Video Upload", "Live Camera"],
        label_visibility="collapsed"
    )

    st.markdown('<p class="sidebar-section">CV Modules</p>', unsafe_allow_html=True)
    enable_detection = st.checkbox("Object Detection",  value=True)
    enable_tracking  = st.checkbox("Object Tracking",   value=False)
    enable_motion    = st.checkbox("Motion Detection",  value=True)
    enable_face      = st.checkbox("Face Detection",    value=True)
    enable_edge      = st.checkbox("Edge Detection",    value=False)

    st.markdown('<p class="sidebar-section">Parameters</p>', unsafe_allow_html=True)
    confidence = st.slider("Detection Confidence", 0.10, 1.00, 0.45, 0.05)

    t1, t2 = 50, 150
    if enable_edge:
        t1 = st.slider("Canny Lower Threshold", 0, 200, 50)
        t2 = st.slider("Canny Upper Threshold", 0, 400, 150)

    st.markdown('<p class="sidebar-section">System</p>', unsafe_allow_html=True)
    model_label = (
        "custom (fine-tuned)"
        if os.path.exists("models/custom_model.pt")
        else "yolov8s  pre-trained"
    )
    st.markdown(
        f'<p style="font-size:0.75rem;font-family:\'DM Mono\',monospace;color:#484f58;">'
        f'Model: {model_label}</p>',
        unsafe_allow_html=True
    )


# ── Page header ───────────────────────────────────────────────────────────────
active_modules = [
    name for flag, name in [
        (enable_detection, "Object Detection"),
        (enable_tracking,  "Object Tracking"),
        (enable_motion,    "Motion Detection"),
        (enable_face,      "Face Detection"),
        (enable_edge,      "Edge Detection"),
    ] if flag
]

pills_html = "".join(f'<span class="pill">{m}</span>' for m in active_modules)

st.markdown(f"""
<div class="page-header">
    <h1>Smart Surveillance & Safety Monitor</h1>
    <p>University of Wollongong in Dubai &nbsp;·&nbsp; CSCI435 Computer Vision</p>
    <div class="pill-row">{pills_html or '<span class="pill pill-inactive">No modules active</span>'}</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 1 — Image Upload
# ─────────────────────────────────────────────────────────────────────────────
if mode == "Image Upload":
    st.markdown('<p class="section-title">Image Analysis</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

    if uploaded is not None:
        st.session_state["motion_detector"].reset()
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Could not decode image. Please try another file.")
            st.stop()

        col_l, col_r = st.columns(2, gap="medium")

        with col_l:
            st.markdown(
                '<div class="img-wrapper"><div class="img-label">Original</div>',
                unsafe_allow_html=True
            )
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("Processing..."):
            t_start = time.perf_counter()
            result, stats = process_frame(
                frame,
                detect=enable_detection, track=enable_tracking,
                motion=enable_motion,    face=enable_face,
                edge=enable_edge,        conf=confidence,
                t1=t1, t2=t2,
            )
            latency_ms = (time.perf_counter() - t_start) * 1000

        with col_r:
            st.markdown(
                '<div class="img-wrapper"><div class="img-label">Processed</div>',
                unsafe_allow_html=True
            )
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        motion_val   = "YES" if stats["motion"] else "NO"
        motion_color = "orange" if stats["motion"] else "green"

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="label">Objects</div>
                <div class="value blue">{stats["objects"]}</div>
            </div>
            <div class="metric-card">
                <div class="label">Faces</div>
                <div class="value blue">{stats["faces"]}</div>
            </div>
            <div class="metric-card">
                <div class="label">Motion</div>
                <div class="value {motion_color}">{motion_val}</div>
            </div>
            <div class="metric-card">
                <div class="label">Latency</div>
                <div class="value">{latency_ms:.0f}<span style="font-size:1rem;color:#8b949e"> ms</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 2 — Video Upload
# ─────────────────────────────────────────────────────────────────────────────
elif mode == "Video Upload":
    st.markdown('<p class="section-title">Video Analysis</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="collapsed"
    )

    if uploaded is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        tfile.flush()
        tfile.close()

        cap          = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.release()

        col_info, col_skip = st.columns([2, 1])
        with col_info:
            st.markdown(
                f'<p style="font-size:0.8rem;font-family:\'DM Mono\',monospace;color:#8b949e;">'
                f'{total_frames} frames &nbsp;·&nbsp; {fps:.1f} fps &nbsp;·&nbsp; '
                f'{total_frames/fps:.1f}s</p>',
                unsafe_allow_html=True
            )
        with col_skip:
            skip = st.slider(
                "Process every Nth frame", 1, 10, 1, label_visibility="visible"
            )

        if st.button("Start Processing"):
            st.session_state["motion_detector"].reset()
            models["object"].reset_tracker()

            cap         = cv2.VideoCapture(tfile.name)
            placeholder = st.empty()
            status_ph   = st.empty()
            progress    = st.progress(0.0)

            frame_idx   = 0
            proc_count  = 0
            max_objects = 0
            max_faces   = 0
            motion_cnt  = 0
            times       = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1
                if frame_idx % skip != 0:
                    continue

                t0 = time.perf_counter()
                result, stats = process_frame(
                    frame,
                    detect=enable_detection, track=enable_tracking,
                    motion=enable_motion,    face=enable_face,
                    edge=enable_edge,        conf=confidence,
                    t1=t1, t2=t2,
                )
                times.append(time.perf_counter() - t0)
                proc_count  += 1
                max_objects  = max(max_objects, stats["objects"])
                max_faces    = max(max_faces,   stats["faces"])
                if stats["motion"]:
                    motion_cnt += 1

                placeholder.image(
                    cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                    use_container_width=True,
                    caption=f"Frame {frame_idx} / {total_frames}",
                )
                m_status = "DETECTED" if stats["motion"] else "CLEAR"
                status_ph.markdown(
                    f'<p class="status-row">'
                    f'Frame {frame_idx} &nbsp;|&nbsp; '
                    f'Objects: {stats["objects"]} &nbsp;|&nbsp; '
                    f'Faces: {stats["faces"]} &nbsp;|&nbsp; '
                    f'Motion: <span class="{"warn" if stats["motion"] else "ok"}">'
                    f'{m_status}</span></p>',
                    unsafe_allow_html=True
                )
                progress.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            os.unlink(tfile.name)

            avg_ms  = (sum(times) / len(times) * 1000) if times else 0
            avg_fps = 1000.0 / avg_ms if avg_ms > 0 else 0

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="label">Frames</div>
                    <div class="value">{proc_count}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Peak Objects</div>
                    <div class="value blue">{max_objects}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Motion Frames</div>
                    <div class="value orange">{motion_cnt}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Avg FPS</div>
                    <div class="value green">{avg_fps:.1f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MODE 3 — Live Camera
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown('<p class="section-title">Live Camera</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="live-notice">
        WebRTC streams your webcam directly to the CV pipeline.
        Click <strong>START</strong> and grant camera permission when prompted.
        Use the sidebar to toggle modules in real time.
    </div>
    """, unsafe_allow_html=True)

    class LiveVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self._object = ObjectDetector()
            self._face   = FaceDetector()
            self._edge   = EdgeDetector()
            self._motion = MotionDetector()
            self.do_detect = True
            self.do_track  = False
            self.do_motion = True
            self.do_face   = True
            self.do_edge   = False
            self.conf      = 0.45
            self.t1        = 50
            self.t2        = 150

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            out = img.copy()
            if self.do_edge:
                out = self._edge.detect(out, self.t1, self.t2)
            if self.do_motion:
                out, _, _ = self._motion.detect(out)
            if self.do_track:
                out, _ = self._object.track(out, self.conf)
            elif self.do_detect:
                out, _ = self._object.detect(out, self.conf)
            if self.do_face:
                out, _ = self._face.detect(out)
            return av.VideoFrame.from_ndarray(out, format="bgr24")

    ctx = webrtc_streamer(
        key="live-surveillance",
        video_processor_factory=LiveVideoProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.do_detect = enable_detection
        ctx.video_processor.do_track  = enable_tracking
        ctx.video_processor.do_motion = enable_motion
        ctx.video_processor.do_face   = enable_face
        ctx.video_processor.do_edge   = enable_edge
        ctx.video_processor.conf      = confidence
        ctx.video_processor.t1        = t1
        ctx.video_processor.t2        = t2
