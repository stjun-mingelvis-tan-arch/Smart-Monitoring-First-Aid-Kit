"""
ESP32 UWB Indoor Position Tracker
===================================
Direct MQTT WebSocket in browser (Leaflet + MQTT.js)
4 UWB anchors (NW, NE, SW, SE) with persistent calibration via localStorage
Light, clean blueprint aesthetic
Kalman filter smoothing in JS
Dual tag support
Fast actions: Access People (ESP32-CAM + live face recognition via MQTT), Stock Tracking (placeholder)

FIXES v2:
  - Axis swap fixed: UWB real_x/real_y swapped in affine solver so NW->NW, SE->SE
  - Anchor real coords updated to match physical layout after axis correction
  - Trail removed - only live dot + pulse circle shown

FEATURE v3:
  - Fast action buttons: Access People / Stock Tracking
  - Access People tab: ESP32-CAM MJPEG stream + live face recognition results
    streamed over the same MQTT broker (topic configurable in sidebar)
  - Stock Tracking tab: placeholder

FEATURE v4:
  - Access People tab now draws a live bounding box overlay on top of the
    MJPEG stream, positioned/scaled from bbox pixel coords published over MQTT
    by the face recognition script. Box auto-clears after a short silence.
"""

import streamlit as st
import base64
import json
import threading
import time
import pickle
import os
from collections import Counter

import numpy as np

st.set_page_config(
    page_title="UWB Tracker",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_defaults = {
    "anchors": {
        "NW": {"img_x": 0.08, "img_y": 0.08, "real_x": 1.81, "real_y": 0.0,  "label": "Anchor NW", "addr": "0x86"},
        "NE": {"img_x": 0.92, "img_y": 0.08, "real_x": 1.81, "real_y": 1.81, "label": "Anchor NE", "addr": "0x87"},
        "SW": {"img_x": 0.08, "img_y": 0.92, "real_x": 0.0,  "real_y": 0.0,  "label": "Anchor SW", "addr": "0x84"},
        "SE": {"img_x": 0.92, "img_y": 0.92, "real_x": 0.0,  "real_y": 1.81, "label": "Anchor SE", "addr": "0x85"},
    },
    "floor_plan_b64":  None,
    "floor_plan_type": "image/png",
    "mqtt_broker":     "35.172.255.228",
    "mqtt_port":       8083,
    "mqtt_topic":      "my_room/esp32_tracker/position",
    "mqtt_topic2":     "my_room/esp32_tracker/position_tag2",
    "tag1_label":      "Tag 1",
    "tag2_label":      "Tag 2",
    "tag2_enabled":    False,
    "board_x":         1.81,
    "board_y":         1.81,
    "smoothing":       "kalman",
    "kalman_r":        0.08,
    "kalman_q":        0.001,
    "min_move":        0.02,
    "active_view":         "tracker",
    "esp32_cam_url":       "http://192.168.100.103:81/stream",
    "mqtt_face_topic":     "my_room/access/face_result",
    "cam_stream_width":    640,
    "cam_stream_height":   480,
    "faces_pkl_path":      "faces.pkl",
    "recognizer_running":  False,
    "recognizer_error":    None,
    "enroll_name":         "",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================================================
# Embedded face-recognition worker
# ---------------------------------------------------------------------
# Runs in a background thread inside this same process, started once
# and kept alive across Streamlit reruns via a module-level singleton
# (st.session_state alone isn't enough to guard this, since each
# browser session gets its own session_state but we only want ONE
# camera connection + model loaded per server process).
# It pulls frames from the ESP32-CAM, runs InsightFace, matches against
# faces.pkl, and publishes results to MQTT — same wire format the
# dashboard's bbox overlay already expects, so no JS changes needed.
# =====================================================================

_RECOGNIZER_STATE = {
    "thread": None,
    "stop_flag": False,
    "running": False,
    "error": None,
}


def _recognition_worker(cam_url, broker, port, topic, pkl_path,
                         stream_w, stream_h, match_threshold=0.40, buffer_size=5):
    import cv2
    import paho.mqtt.client as mqtt
    from insightface.app import FaceAnalysis

    try:
        with open(pkl_path, "rb") as f:
            face_db = pickle.load(f)
    except FileNotFoundError:
        _RECOGNIZER_STATE["error"] = f"'{pkl_path}' not found. Run your enrollment script first."
        _RECOGNIZER_STATE["running"] = False
        return

    try:
        app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        _RECOGNIZER_STATE["error"] = f"Failed to load AI model: {e}"
        _RECOGNIZER_STATE["running"] = False
        return

    cap = cv2.VideoCapture(cam_url)
    if not cap.isOpened():
        _RECOGNIZER_STATE["error"] = f"Could not connect to ESP32-CAM stream at {cam_url}"
        _RECOGNIZER_STATE["running"] = False
        return

    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="face_recognition_embedded")
    try:
        mqtt_client.connect(broker, 1883, keepalive=30)  # plain MQTT, not the browser WS port
        mqtt_client.loop_start()
    except Exception as e:
        _RECOGNIZER_STATE["error"] = f"Could not connect to MQTT broker: {e}"
        cap.release()
        _RECOGNIZER_STATE["running"] = False
        return

    frame_buffer = []
    _RECOGNIZER_STATE["running"] = True
    _RECOGNIZER_STATE["error"] = None

    try:
        while not _RECOGNIZER_STATE["stop_flag"]:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_h, frame_w = frame.shape[:2]
            faces = app.get(frame)

            if len(faces) == 0:
                frame_buffer.append("Unknown")
                if len(frame_buffer) > buffer_size:
                    frame_buffer.pop(0)
                continue

            for face in faces:
                live_embedding = face.embedding
                live_norm = live_embedding / np.linalg.norm(live_embedding)

                best_name = "Unknown"
                best_similarity = -1.0

                for name, stored_embedding in face_db.items():
                    stored_norm = stored_embedding / np.linalg.norm(stored_embedding)
                    similarity = float(np.dot(stored_norm, live_norm))
                    if similarity > best_similarity:
                        best_similarity = similarity
                        if similarity >= match_threshold:
                            best_name = name

                frame_buffer.append(best_name)
                if len(frame_buffer) > buffer_size:
                    frame_buffer.pop(0)

                majority_vote = Counter(frame_buffer).most_common(1)[0]
                matched_name = majority_vote[0] if majority_vote[1] >= 3 else "Unknown"

                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

                status = "authorized" if matched_name != "Unknown" else "unverified"

                scale_x = stream_w / frame_w
                scale_y = stream_h / frame_h

                payload = {
                    "ts": time.time(),
                    "status": status,
                    "name": matched_name,
                    "similarity": max(best_similarity, 0.0),
                    "bbox": [
                        int(x1 * scale_x),
                        int(y1 * scale_y),
                        int(x2 * scale_x),
                        int(y2 * scale_y),
                    ],
                    "stream_width": stream_w,
                    "stream_height": stream_h,
                }
                mqtt_client.publish(topic, json.dumps(payload), qos=0)
    finally:
        cap.release()
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        _RECOGNIZER_STATE["running"] = False


def start_recognizer():
    if _ENROLL_STATE["running"]:
        _RECOGNIZER_STATE["error"] = "Enrollment is currently running — stop it first to free up the camera."
        return
    if _RECOGNIZER_STATE["thread"] is not None and _RECOGNIZER_STATE["thread"].is_alive():
        return  # already running, no-op
    _RECOGNIZER_STATE["stop_flag"] = False
    _RECOGNIZER_STATE["error"] = None
    t = threading.Thread(
        target=_recognition_worker,
        args=(
            st.session_state.esp32_cam_url,
            st.session_state.mqtt_broker,
            st.session_state.mqtt_port,
            st.session_state.mqtt_face_topic,
            st.session_state.faces_pkl_path,
            st.session_state.cam_stream_width,
            st.session_state.cam_stream_height,
        ),
        daemon=True,
    )
    _RECOGNIZER_STATE["thread"] = t
    t.start()


def stop_recognizer():
    _RECOGNIZER_STATE["stop_flag"] = True


# =====================================================================
# Embedded enrollment worker
# ---------------------------------------------------------------------
# Same single-background-thread pattern as the recognizer above, and
# guarded to refuse starting if the recognizer is already using the
# camera (and vice versa) since both need exclusive access to the
# single ESP32-CAM stream on this machine.
#
# UI flow: worker continuously runs FaceMesh (for live EAR / blink
# value) and keeps the latest raw frame available. The "Capture this
# angle" button sets capture_requested=True; the worker grabs the next
# frame, runs InsightFace on it, and stores the embedding for the
# current step — this replaces the old spacebar-press logic. The final
# step (blink) is still detected automatically from live EAR, exactly
# like before, just without a keyboard window.
# =====================================================================

ENROLL_STEPS = [
    "Look straight ahead",
    "Tilt your head UP slightly",
    "Tilt your head DOWN slightly",
    "Turn your head LEFT slightly",
    "Turn your head RIGHT slightly",
    "Tilt your head UP and LEFT",
    "Tilt your head UP and RIGHT",
    "Tilt your head DOWN and LEFT",
    "Blink your eyes to finish",
]
EAR_THRESHOLD = 0.21
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

_ENROLL_STATE = {
    "thread": None,
    "stop_flag": False,
    "running": False,
    "error": None,
    "step": 0,
    "capture_requested": False,
    "last_capture_ok": None,   # True/False/None — feedback for the most recent capture attempt
    "avg_ear": 0.0,
    "saved_name": None,        # set once the final embedding is written to disk
}


def _get_ear(landmarks, indices, w, h):
    # landmarks: list of NormalizedLandmark objects (mediapipe Tasks API) —
    # same .x / .y attribute access as the legacy solutions API, so this
    # function's body is unchanged.
    pts = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    h1 = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * h1)


_FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_FACE_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task"
)


def _ensure_face_landmarker_model():
    """Downloads the FaceLandmarker model bundle once, next to this script,
    if it isn't already present. Required by the new mediapipe Tasks API —
    unlike the old `solutions` API, the model isn't bundled with the pip
    package anymore."""
    if os.path.exists(_FACE_LANDMARKER_MODEL_PATH):
        return _FACE_LANDMARKER_MODEL_PATH
    import urllib.request
    urllib.request.urlretrieve(_FACE_LANDMARKER_MODEL_URL, _FACE_LANDMARKER_MODEL_PATH)
    return _FACE_LANDMARKER_MODEL_PATH


def _make_face_landmarker():
    import mediapipe as mp
    model_path = _ensure_face_landmarker_model()
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def _enrollment_worker(cam_url, pkl_path, worker_name):
    import cv2
    import mediapipe as mp
    from insightface.app import FaceAnalysis

    try:
        app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        face_landmarker = _make_face_landmarker()
    except Exception as e:
        _ENROLL_STATE["error"] = f"Failed to load AI models: {e}"
        _ENROLL_STATE["running"] = False
        return

    cap = cv2.VideoCapture(cam_url)
    if not cap.isOpened():
        _ENROLL_STATE["error"] = f"Could not connect to ESP32-CAM stream at {cam_url}"
        _ENROLL_STATE["running"] = False
        return

    collected_embeddings = []
    is_eyes_closed = False
    _ENROLL_STATE["running"] = True
    _ENROLL_STATE["error"] = None
    _ENROLL_STATE["step"] = 0
    _ENROLL_STATE["last_capture_ok"] = None
    _ENROLL_STATE["saved_name"] = None

    try:
        while not _ENROLL_STATE["stop_flag"] and _ENROLL_STATE["step"] < len(ENROLL_STEPS):
            ret, frame = cap.read()
            if not ret:
                continue

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = face_landmarker.detect(mp_image)

            step = _ENROLL_STATE["step"]

            if not results.face_landmarks:
                continue

            landmarks = results.face_landmarks[0]
            left_ear = _get_ear(landmarks, LEFT_EYE, w, h)
            right_ear = _get_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            _ENROLL_STATE["avg_ear"] = avg_ear

            if step < 8:
                # Manual capture steps — wait for the UI button to request one
                if _ENROLL_STATE["capture_requested"]:
                    _ENROLL_STATE["capture_requested"] = False
                    faces = app.get(frame)
                    if len(faces) == 1:
                        collected_embeddings.append(faces[0].embedding)
                        _ENROLL_STATE["last_capture_ok"] = True
                        _ENROLL_STATE["step"] += 1
                    else:
                        _ENROLL_STATE["last_capture_ok"] = False
            else:
                # Step 9: automatic blink detection, same logic as before
                if avg_ear < EAR_THRESHOLD:
                    is_eyes_closed = True
                elif avg_ear >= EAR_THRESHOLD and is_eyes_closed:
                    is_eyes_closed = False
                    faces = app.get(frame)
                    if len(faces) == 1:
                        collected_embeddings.append(faces[0].embedding)
                        _ENROLL_STATE["last_capture_ok"] = True
                        _ENROLL_STATE["step"] += 1

        if not _ENROLL_STATE["stop_flag"] and len(collected_embeddings) == 9:
            mean_embedding = np.mean(np.vstack(collected_embeddings), axis=0)
            final_embedding = mean_embedding / np.linalg.norm(mean_embedding)

            try:
                with open(pkl_path, "rb") as f:
                    face_db = pickle.load(f)
            except FileNotFoundError:
                face_db = {}

            face_db[worker_name] = final_embedding
            with open(pkl_path, "wb") as f:
                pickle.dump(face_db, f)

            _ENROLL_STATE["saved_name"] = worker_name
    finally:
        cap.release()
        _ENROLL_STATE["running"] = False


def start_enrollment(name):
    if _RECOGNIZER_STATE["running"]:
        _ENROLL_STATE["error"] = "Recognition is currently running — stop it first to free up the camera."
        return False
    if _ENROLL_STATE["thread"] is not None and _ENROLL_STATE["thread"].is_alive():
        return False  # already running
    _ENROLL_STATE["stop_flag"] = False
    _ENROLL_STATE["error"] = None
    t = threading.Thread(
        target=_enrollment_worker,
        args=(st.session_state.esp32_cam_url, st.session_state.faces_pkl_path, name),
        daemon=True,
    )
    _ENROLL_STATE["thread"] = t
    t.start()
    return True


def request_capture():
    _ENROLL_STATE["capture_requested"] = True


def stop_enrollment():
    _ENROLL_STATE["stop_flag"] = True

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@400;600;700&display=swap');
html, [class*="css"] { font-family: 'Outfit', sans-serif !important; }
.stApp { background: #f0f4f8; color: #1a2333; }
h1,h2,h3 { font-family: 'Outfit', sans-serif !important; color: #0f1f35 !important; font-weight: 700 !important; }
section[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #dde3ec !important; }
section[data-testid="stSidebar"] * { color: #1a2333 !important; }
section[data-testid="stSidebar"] .stExpander { border: 1px solid #dde3ec !important; border-radius: 8px !important; background: #f8fafc !important; }
.stButton>button { background: #1a56db !important; color: #ffffff !important; border: none !important; border-radius: 7px !important; font-family: 'DM Mono', monospace !important; font-size: 12px !important; padding: 6px 14px !important; transition: all .18s !important; }
.stButton>button:hover { background: #1140a8 !important; transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(26,86,219,.3) !important; }
.stNumberInput input, .stTextInput input { background: #f8fafc !important; color: #1a2333 !important; border-color: #dde3ec !important; border-radius: 7px !important; font-family: 'DM Mono', monospace !important; }
.stSelectbox > div { background: #f8fafc !important; }
hr { border-color: #dde3ec !important; }
.block-container { padding-top: 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;gap:12px;padding-bottom:4px">
  <div style="background:#1a56db;border-radius:10px;width:40px;height:40px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0">📡</div>
  <div>
    <div style="font-family:'Outfit',sans-serif;font-size:20px;font-weight:700;color:#0f1f35;line-height:1.1">UWB Position Tracker</div>
    <div style="font-family:'DM Mono',monospace;font-size:9px;color:#8fa8c8;letter-spacing:2px;margin-top:1px">DW1000 · ESP32 · MQTT · KALMAN · 4-ANCHOR TRILATERATION</div>
  </div>
</div>
<div style="height:2px;background:linear-gradient(90deg,#1a56db,#38bdf8,transparent);border-radius:2px;margin:10px 0 16px"></div>
""", unsafe_allow_html=True)

fa1, fa2, fa3, fa4, _sp = st.columns([1, 1, 1, 1, 2])
with fa1:
    if st.button("📍 Tracker", use_container_width=True,
                  type=("primary" if st.session_state.active_view == "tracker" else "secondary")):
        st.session_state.active_view = "tracker"
        st.rerun()
with fa2:
    if st.button("🧑‍🤝‍🧑 Access People", use_container_width=True,
                  type=("primary" if st.session_state.active_view == "access_people" else "secondary")):
        st.session_state.active_view = "access_people"
        st.rerun()
with fa3:
    if st.button("📦 Stock Tracking", use_container_width=True,
                  type=("primary" if st.session_state.active_view == "stock_tracking" else "secondary")):
        st.session_state.active_view = "stock_tracking"
        st.rerun()
with fa4:
    if st.button("🪪 Enroll Employee", use_container_width=True,
                  type=("primary" if st.session_state.active_view == "enroll" else "secondary")):
        st.session_state.active_view = "enroll"
        st.rerun()

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    with st.expander("📶 MQTT / Broker", expanded=True):
        st.session_state.mqtt_broker = st.text_input("Broker IP", value=st.session_state.mqtt_broker)
        st.session_state.mqtt_port   = st.number_input("WebSocket Port", value=st.session_state.mqtt_port, min_value=1, max_value=65535)
        st.caption("Port **8083** = WS plain · Port **8084** = WSS secure")

    with st.expander("🏷️ Tags", expanded=True):
        st.markdown("**Tag 1** 🔴")
        st.session_state.tag1_label = st.text_input("Label", value=st.session_state.tag1_label, key="t1l")
        st.session_state.mqtt_topic = st.text_input("Topic", value=st.session_state.mqtt_topic, key="t1t")
        st.markdown("---")
        st.session_state.tag2_enabled = st.toggle("Enable Tag 2 🔵", value=st.session_state.tag2_enabled)
        if st.session_state.tag2_enabled:
            st.session_state.tag2_label  = st.text_input("Tag 2 Label", value=st.session_state.tag2_label)
            st.session_state.mqtt_topic2 = st.text_input("Tag 2 Topic", value=st.session_state.mqtt_topic2)

    with st.expander("📐 Board Dimensions", expanded=True):
        st.caption("Physical size of your UWB board in metres")
        st.session_state.board_x = st.number_input("Width  (m)", value=st.session_state.board_x, step=0.01, format="%.2f")
        st.session_state.board_y = st.number_input("Height (m)", value=st.session_state.board_y, step=0.01, format="%.2f")

    with st.expander("🗺️ Floor Plan", expanded=False):
        uploaded = st.file_uploader("Upload image", type=["png","jpg","jpeg","svg","webp"])
        if uploaded:
            raw = uploaded.read()
            st.session_state.floor_plan_b64  = base64.b64encode(raw).decode()
            st.session_state.floor_plan_type = uploaded.type
            st.success(f"✅ {uploaded.name}")
        elif st.session_state.floor_plan_b64:
            st.info("Using uploaded floor plan.")

    with st.expander("📍 Anchor Calibration (saved in browser)", expanded=True):
        st.caption("Set image position (0-1 fraction) and real-world position in metres. Saved in your browser.")
        for aid, a in st.session_state.anchors.items():
            with st.expander(f"{aid} · {a['addr']} · {a['label']}", expanded=False):
                a['label'] = st.text_input("Label", a['label'], key=f"lbl_{aid}")
                a['addr']  = st.text_input("Short addr", a['addr'], key=f"addr_{aid}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**On image (0-1)**")
                    a['img_x'] = st.number_input("X →", 0.0, 1.0, float(a['img_x']), 0.01, key=f"ix_{aid}")
                    a['img_y'] = st.number_input("Y ↓", 0.0, 1.0, float(a['img_y']), 0.01, key=f"iy_{aid}")
                with c2:
                    st.markdown("**Real (metres)**")
                    a['real_x'] = st.number_input("X (m)", value=float(a['real_x']), step=0.01, format="%.2f", key=f"rx_{aid}")
                    a['real_y'] = st.number_input("Y (m)", value=float(a['real_y']), step=0.01, format="%.2f", key=f"ry_{aid}")

    with st.expander("🎛️ Smoothing", expanded=False):
        st.session_state.smoothing = st.selectbox("Algorithm", ["kalman","moving_avg","none"],
            index=["kalman","moving_avg","none"].index(st.session_state.smoothing))
        st.session_state.min_move = st.slider("Min move (m)", 0.0, 0.3, st.session_state.min_move, 0.005, format="%.3f")
        if st.session_state.smoothing == "kalman":
            st.session_state.kalman_r = st.slider("R (meas. noise)", 0.001, 1.0, st.session_state.kalman_r, 0.001, format="%.3f")
            st.session_state.kalman_q = st.slider("Q (proc. noise)", 0.0001, 0.1, st.session_state.kalman_q, 0.0001, format="%.4f")

    with st.expander("🧑‍🤝‍🧑 Access People / Camera", expanded=(st.session_state.active_view == "access_people")):
        st.caption("ESP32-CAM stream URL and the MQTT topic the face-recognition worker publishes to.")
        st.session_state.esp32_cam_url   = st.text_input("ESP32-CAM stream URL", value=st.session_state.esp32_cam_url)
        st.session_state.mqtt_face_topic = st.text_input("Face result MQTT topic", value=st.session_state.mqtt_face_topic)
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.cam_stream_width = st.number_input(
                "Stream width (px)", value=st.session_state.cam_stream_width, min_value=1, step=1)
        with c2:
            st.session_state.cam_stream_height = st.number_input(
                "Stream height (px)", value=st.session_state.cam_stream_height, min_value=1, step=1)
        st.session_state.faces_pkl_path = st.text_input(
            "Faces database (.pkl) path", value=st.session_state.faces_pkl_path)
        st.caption("Stream width/height must match what the recognizer scales bbox coords to, so the box lines up.")
        st.caption("Publishes over plain MQTT (port 1883) to the same broker IP configured above, separate from the browser's WebSocket port.")

        st.markdown("---")
        is_running = _RECOGNIZER_STATE["running"]
        rc1, rc2 = st.columns(2)
        with rc1:
            if st.button("▶ Start Recognition", disabled=is_running, use_container_width=True):
                start_recognizer()
                st.rerun()
        with rc2:
            if st.button("⏹ Stop Recognition", disabled=not is_running, use_container_width=True):
                stop_recognizer()
                st.rerun()
        if is_running:
            st.success("Recognizer running in background")
        elif _RECOGNIZER_STATE["error"]:
            st.error(f"Recognizer stopped: {_RECOGNIZER_STATE['error']}")
        else:
            st.info("Recognizer not running")


def default_floorplan_svg(bx, by) -> str:
    W, H = 900, int(900 * by / bx) if bx > 0 else 720
    H = max(H, 400)
    step_m = 0.25
    step_px_x = 900 * step_m / bx if bx > 0 else 225
    step_px_y = H   * step_m / by if by > 0 else 225
    grid_lines = ""
    x = step_px_x
    while x < W:
        grid_lines += f"<line x1='{x:.1f}' y1='0' x2='{x:.1f}' y2='{H}' stroke='#ccd8e8' stroke-width='0.8'/>"
        x += step_px_x
    y = step_px_y
    while y < H:
        grid_lines += f"<line x1='0' y1='{y:.1f}' x2='{W}' y2='{y:.1f}' stroke='#ccd8e8' stroke-width='0.8'/>"
        y += step_px_y
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{H}'>
  <rect width='{W}' height='{H}' fill='#f5f8ff'/>
  {grid_lines}
  <rect x='2' y='2' width='{W-4}' height='{H-4}' fill='none' stroke='#1a56db' stroke-width='3' rx='4'/>
  <text x='{W//2}' y='{H//2-10}' text-anchor='middle' fill='#c8d8ee' font-family='monospace' font-size='14' font-weight='bold'>{bx:.2f} m × {by:.2f} m</text>
  <text x='{W//2}' y='{H//2+14}' text-anchor='middle' fill='#c8d8ee' font-family='monospace' font-size='11'>Upload floor plan in sidebar →</text>
</svg>"""
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def build_component() -> str:
    if st.session_state.floor_plan_b64:
        img_src = f"data:{st.session_state.floor_plan_type};base64,{st.session_state.floor_plan_b64}"
    else:
        img_src = default_floorplan_svg(st.session_state.board_x, st.session_state.board_y)

    cfg = {
        "broker":      st.session_state.mqtt_broker,
        "port":        st.session_state.mqtt_port,
        "topic":       st.session_state.mqtt_topic,
        "topic2":      st.session_state.mqtt_topic2 if st.session_state.tag2_enabled else "",
        "tag1Label":   st.session_state.tag1_label,
        "tag2Label":   st.session_state.tag2_label,
        "tag2Enabled": st.session_state.tag2_enabled,
        "anchors":     st.session_state.anchors,
        "boardX":      st.session_state.board_x,
        "boardY":      st.session_state.board_y,
        "smoothing":   st.session_state.smoothing,
        "kalmanR":     st.session_state.kalman_r,
        "kalmanQ":     st.session_state.kalman_q,
        "minMove":     st.session_state.min_move,
    }
    cfg_json = json.dumps(cfg)

    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/mqtt@5.3.4/dist/mqtt.min.js"></script>
<style>
* { margin:0;padding:0;box-sizing:border-box; }
body { background:#f0f4f8;font-family:'DM Mono',monospace,sans-serif;overflow:hidden; }
#map { width:100%;height:580px;background:#f5f8ff; }
.leaflet-container { background:#f5f8ff !important; }
.leaflet-control-zoom a { background:#fff !important;color:#1a56db !important;border-color:#dde3ec !important;font-weight:700; }
.leaflet-popup-content-wrapper { background:#fff;border:1px solid #dde3ec;border-radius:10px;color:#1a2333;font-family:'Outfit',sans-serif;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.1); }
.leaflet-popup-tip { background:#fff; }
#hud {
  position:absolute;top:12px;right:12px;z-index:900;
  background:rgba(255,255,255,.96);border:1px solid #dde3ec;border-radius:12px;
  padding:14px 18px;min-width:210px;box-shadow:0 4px 20px rgba(0,0,0,.1);
  pointer-events:none;backdrop-filter:blur(10px);
}
.hud-title { font-family:'Outfit',sans-serif;font-size:11px;font-weight:700;color:#0f1f35;letter-spacing:1px;text-transform:uppercase;padding-bottom:8px;margin-bottom:8px;border-bottom:2px solid #eff6ff;display:flex;align-items:center;gap:7px; }
.hud-section { border-bottom:1px solid #f0f4f8;margin-bottom:8px;padding-bottom:8px; }
.hud-section:last-child { border:none;margin:0;padding:0; }
.hud-tag { font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:5px; }
.hud-row { display:flex;justify-content:space-between;align-items:center;margin:3px 0; }
.hud-lbl { color:#8fa8c8;font-size:9px;letter-spacing:1px;text-transform:uppercase; }
.hud-val { color:#1a56db;font-size:11px;font-weight:700;font-family:'DM Mono',monospace; }
.dot { display:inline-block;width:8px;height:8px;border-radius:50%;flex-shrink:0; }
.dot-g { background:#22c55e;box-shadow:0 0 8px #22c55e88;animation:blink 1.4s ease-in-out infinite; }
.dot-r { background:#ef4444; }
.dot-y { background:#f59e0b;animation:blink 1.4s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }
#statusbar {
  position:absolute;bottom:0;left:0;right:0;z-index:900;
  background:rgba(255,255,255,.94);border-top:1px solid #dde3ec;
  padding:7px 16px;display:flex;gap:24px;align-items:center;
  backdrop-filter:blur(8px);pointer-events:none;
}
.sb-item { display:flex;flex-direction:column; }
.sb-lbl { color:#aab8cc;font-size:8px;letter-spacing:1.5px;text-transform:uppercase; }
.sb-val { color:#1a2333;font-size:11px;font-family:'DM Mono',monospace;font-weight:500;margin-top:1px; }
#legend {
  position:absolute;bottom:48px;left:12px;z-index:900;
  background:rgba(255,255,255,.94);border:1px solid #dde3ec;border-radius:10px;
  padding:10px 14px;box-shadow:0 2px 10px rgba(0,0,0,.08);pointer-events:none;
}
.leg-row { display:flex;align-items:center;gap:8px;margin:3px 0; }
.leg-txt { color:#6b7a99;font-size:10px;font-family:'Outfit',sans-serif; }
#cal-toast {
  position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:1000;
  background:#dcfce7;border:1px solid #86efac;border-radius:8px;padding:7px 18px;
  color:#166534;font-size:10px;letter-spacing:1px;display:none;box-shadow:0 2px 10px rgba(0,0,0,.1);
}
#cal-loaded {
  position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:1000;
  background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:7px 18px;
  color:#1a56db;font-size:10px;letter-spacing:1px;display:none;box-shadow:0 2px 10px rgba(0,0,0,.1);
}
</style>
</head>
<body>
<div style="position:relative">
  <div id="map"></div>

  <div id="hud">
    <div class="hud-title">
      <span id="h-status"><span class="dot dot-r"></span></span>
      UWB TRACKER
    </div>
    <div class="hud-section">
      <div class="hud-tag" style="color:#dc2626">🔴 <span id="h-t1-name">TAG 1</span></div>
      <div class="hud-row"><span class="hud-lbl">X raw</span><span class="hud-val" id="h1-rx">—</span></div>
      <div class="hud-row"><span class="hud-lbl">Y raw</span><span class="hud-val" id="h1-ry">—</span></div>
      <div class="hud-row"><span class="hud-lbl">X smooth</span><span class="hud-val" id="h1-sx">—</span></div>
      <div class="hud-row"><span class="hud-lbl">Y smooth</span><span class="hud-val" id="h1-sy">—</span></div>
      <div class="hud-row"><span class="hud-lbl">Speed</span><span class="hud-val" id="h1-spd">—</span></div>
    </div>
    <div class="hud-section" id="hud-t2" style="display:none">
      <div class="hud-tag" style="color:#0369a1">🔵 <span id="h-t2-name">TAG 2</span></div>
      <div class="hud-row"><span class="hud-lbl">X raw</span><span class="hud-val" id="h2-rx">—</span></div>
      <div class="hud-row"><span class="hud-lbl">Y raw</span><span class="hud-val" id="h2-ry">—</span></div>
      <div class="hud-row"><span class="hud-lbl">X smooth</span><span class="hud-val" id="h2-sx">—</span></div>
      <div class="hud-row"><span class="hud-lbl">Y smooth</span><span class="hud-val" id="h2-sy">—</span></div>
      <div class="hud-row"><span class="hud-lbl">Speed</span><span class="hud-val" id="h2-spd">—</span></div>
    </div>
    <div class="hud-row">
      <span class="hud-lbl">Messages</span>
      <span class="hud-val" id="h-msgs">0</span>
    </div>
  </div>

  <div id="legend">
    <div class="leg-row">
      <div style="width:14px;height:14px;border-radius:3px;background:#eff6ff;border:2px solid #1a56db;flex-shrink:0"></div>
      <span class="leg-txt">UWB Anchor</span>
    </div>
    <div class="leg-row">
      <div style="width:14px;height:14px;border-radius:50%;background:#dc2626;border:2px solid #fff;box-shadow:0 0 0 2px #dc262644;flex-shrink:0"></div>
      <span class="leg-txt" id="leg-t1">Tag 1 (live)</span>
    </div>
    <div id="leg-t2-wrap" style="display:none">
      <div class="leg-row" style="margin-top:4px">
        <div style="width:14px;height:14px;border-radius:50%;background:#0369a1;border:2px solid #fff;box-shadow:0 0 0 2px #0369a144;flex-shrink:0"></div>
        <span class="leg-txt" id="leg-t2">Tag 2 (live)</span>
      </div>
    </div>
  </div>

  <div id="statusbar">
    <div class="sb-item"><span class="sb-lbl">Broker</span><span class="sb-val" id="sb-broker">—</span></div>
    <div class="sb-item"><span class="sb-lbl">Tag 1 topic</span><span class="sb-val" id="sb-t1">—</span></div>
    <div class="sb-item" id="sb-t2-wrap"><span class="sb-lbl">Tag 2 topic</span><span class="sb-val" id="sb-t2">—</span></div>
    <div class="sb-item"><span class="sb-lbl">Msg/s</span><span class="sb-val" id="sb-rate">0</span></div>
    <div class="sb-item"><span class="sb-lbl">Last update</span><span class="sb-val" id="sb-last">—</span></div>
    <div class="sb-item" style="margin-left:auto;display:flex;gap:8px;align-items:center">
      <span id="sb-cal-src" style="color:#aab8cc;font-size:9px;font-family:monospace;letter-spacing:1px"></span>
      <button onclick="saveCal()" style="background:#1a56db;color:#fff;border:none;border-radius:6px;padding:5px 12px;font-size:10px;cursor:pointer;font-family:monospace;pointer-events:all"
        onmouseover="this.style.background='#1140a8'" onmouseout="this.style.background='#1a56db'">💾 Save Cal</button>
      <button onclick="clearCal()" style="background:#ef4444;color:#fff;border:none;border-radius:6px;padding:5px 12px;font-size:10px;cursor:pointer;font-family:monospace;pointer-events:all"
        onmouseover="this.style.background='#b91c1c'" onmouseout="this.style.background='#ef4444'">🗑 Reset Cal</button>
    </div>
  </div>

  <div id="cal-toast">✓ Calibration saved to browser</div>
  <div id="cal-loaded">📂 Calibration loaded from browser</div>
</div>

<script>
const CFG = __CFG_JSON__;
const IMG_SRC = "__IMG_SRC__";
const LS_ANCHORS = "uwb_calibration_v2";
const LS_BOARD   = "uwb_board_v1";

function loadCal() {
  let loaded = false;
  try {
    const raw = localStorage.getItem(LS_ANCHORS);
    if (raw) {
      const saved = JSON.parse(raw);
      Object.keys(saved).forEach(id => {
        if (!CFG.anchors[id]) return;
        const s = saved[id];
        ['img_x','img_y','real_x','real_y','label','addr'].forEach(k => {
          if (s[k] !== undefined) CFG.anchors[id][k] = s[k];
        });
      });
      loaded = true;
    }
    const braw = localStorage.getItem(LS_BOARD);
    if (braw) {
      const bd = JSON.parse(braw);
      if (bd.boardX) CFG.boardX = bd.boardX;
      if (bd.boardY) CFG.boardY = bd.boardY;
    }
  } catch(e) { console.warn("Cal load failed:", e); }
  if (loaded) {
    document.getElementById('sb-cal-src').textContent = '📂 localStorage';
    const b = document.getElementById('cal-loaded');
    b.style.display = 'block';
    setTimeout(() => b.style.display = 'none', 2800);
  } else {
    document.getElementById('sb-cal-src').textContent = '📋 defaults';
  }
}

function saveCal() {
  try {
    const d = {};
    Object.entries(CFG.anchors).forEach(([id, a]) => {
      d[id] = { img_x:a.img_x, img_y:a.img_y, real_x:a.real_x, real_y:a.real_y, label:a.label, addr:a.addr };
    });
    localStorage.setItem(LS_ANCHORS, JSON.stringify(d));
    localStorage.setItem(LS_BOARD, JSON.stringify({ boardX:CFG.boardX, boardY:CFG.boardY }));
    document.getElementById('sb-cal-src').textContent = '📂 localStorage';
    const t = document.getElementById('cal-toast');
    t.style.display = 'block';
    setTimeout(() => t.style.display = 'none', 2500);
  } catch(e) { console.warn("Cal save failed:", e); }
}

function clearCal() {
  localStorage.removeItem(LS_ANCHORS);
  localStorage.removeItem(LS_BOARD);
  document.getElementById('sb-cal-src').textContent = '📋 defaults';
  alert("Calibration cleared. Reload to restore defaults.");
}

loadCal();

function computeTransform(anchors) {
  const keys = Object.keys(anchors);
  if (keys.length < 2) return null;

  function ne(rows, vals) {
    let A=[[0,0,0],[0,0,0],[0,0,0]], b=[0,0,0];
    for (let i=0;i<rows.length;i++) {
      for (let r=0;r<3;r++) for (let c=0;c<3;c++) A[r][c]+=rows[i][r]*rows[i][c];
      for (let r=0;r<3;r++) b[r]+=rows[i][r]*vals[i];
    }
    for (let col=0;col<3;col++) {
      let mx=col;
      for (let r=col+1;r<3;r++) if (Math.abs(A[r][col])>Math.abs(A[mx][col])) mx=r;
      [A[col],A[mx]]=[A[mx],A[col]]; [b[col],b[mx]]=[b[mx],b[col]];
      if (Math.abs(A[col][col])<1e-12) return null;
      for (let r=col+1;r<3;r++) {
        const f=A[r][col]/A[col][col];
        for (let c=col;c<3;c++) A[r][c]-=f*A[col][c];
        b[r]-=f*b[col];
      }
    }
    const res=[0,0,0];
    for (let r=2;r>=0;r--) {
      res[r]=b[r];
      for (let c=r+1;c<3;c++) res[r]-=A[r][c]*res[c];
      res[r]/=A[r][r];
    }
    return res;
  }

  const rows = keys.map(k => [anchors[k].real_y, anchors[k].real_x, 1]);
  const xv   = keys.map(k => anchors[k].img_x);
  const yv   = keys.map(k => anchors[k].img_y);
  const xc = ne(rows, xv), yc = ne(rows, yv);
  if (!xc || !yc) return null;
  return { a:xc[0], b:xc[1], tx:xc[2], c:yc[0], d:yc[1], ty:yc[2] };
}

function applyT(T, rx, ry) {
  return { fx: T.a*ry + T.b*rx + T.tx, fy: T.c*ry + T.d*rx + T.ty };
}

function makeKalman(R, Q) {
  let x=[0,0,0,0], P=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], init=false;
  return {
    reset() { init=false; },
    update(mx, my, dt) {
      if (!init) { x=[mx,my,0,0]; init=true; return {x:mx,y:my,vx:0,vy:0}; }
      const t=Math.min(dt||0.1,2);
      const xp=[x[0]+x[2]*t,x[1]+x[3]*t,x[2],x[3]];
      const Pp=[
        [P[0][0]+t*P[2][0]+t*(P[0][2]+t*P[2][2])+Q,P[0][1]+t*P[2][1]+t*(P[0][3]+t*P[2][3]),P[0][2]+t*P[2][2],P[0][3]+t*P[2][3]],
        [P[1][0]+t*P[3][0]+t*(P[1][2]+t*P[3][2]),P[1][1]+t*P[3][1]+t*(P[1][3]+t*P[3][3])+Q,P[1][2]+t*P[3][2],P[1][3]+t*P[3][3]],
        [P[2][0],P[2][1],P[2][2]+Q,P[2][3]],
        [P[3][0],P[3][1],P[3][2],P[3][3]+Q]
      ];
      const ix=mx-xp[0], iy=my-xp[1];
      const S00=Pp[0][0]+R, S11=Pp[1][1]+R;
      const K=[[Pp[0][0]/S00,Pp[0][1]/S11],[Pp[1][0]/S00,Pp[1][1]/S11],
               [Pp[2][0]/S00,Pp[2][1]/S11],[Pp[3][0]/S00,Pp[3][1]/S11]];
      x=[xp[0]+K[0][0]*ix+K[0][1]*iy,xp[1]+K[1][0]*ix+K[1][1]*iy,
         xp[2]+K[2][0]*ix+K[2][1]*iy,xp[3]+K[3][0]*ix+K[3][1]*iy];
      P=[[Pp[0][0]-K[0][0]*Pp[0][0]-K[0][1]*Pp[1][0],Pp[0][1]-K[0][0]*Pp[0][1]-K[0][1]*Pp[1][1],Pp[0][2]-K[0][0]*Pp[0][2]-K[0][1]*Pp[1][2],Pp[0][3]-K[0][0]*Pp[0][3]-K[0][1]*Pp[1][3]],
         [Pp[1][0]-K[1][0]*Pp[0][0]-K[1][1]*Pp[1][0],Pp[1][1]-K[1][0]*Pp[0][1]-K[1][1]*Pp[1][1],Pp[1][2]-K[1][0]*Pp[0][2]-K[1][1]*Pp[1][2],Pp[1][3]-K[1][0]*Pp[0][3]-K[1][1]*Pp[1][3]],
         [Pp[2][0]-K[2][0]*Pp[0][0]-K[2][1]*Pp[1][0],Pp[2][1]-K[2][0]*Pp[0][1]-K[2][1]*Pp[1][1],Pp[2][2]-K[2][0]*Pp[0][2]-K[2][1]*Pp[1][2],Pp[2][3]-K[2][0]*Pp[0][3]-K[2][1]*Pp[1][3]],
         [Pp[3][0]-K[3][0]*Pp[0][0]-K[3][1]*Pp[1][0],Pp[3][1]-K[3][0]*Pp[0][1]-K[3][1]*Pp[1][1],Pp[3][2]-K[3][0]*Pp[0][2]-K[3][1]*Pp[1][2],Pp[3][3]-K[3][0]*Pp[0][3]-K[3][1]*Pp[1][3]]];
      return {x:x[0],y:x[1],vx:x[2],vy:x[3]};
    }
  };
}

function makeMovAvg(n=5) {
  let buf=[];
  return {
    reset() { buf=[]; },
    update(x,y) {
      buf.push({x,y}); if(buf.length>n) buf.shift();
      return { x:buf.reduce((s,p)=>s+p.x,0)/buf.length, y:buf.reduce((s,p)=>s+p.y,0)/buf.length };
    }
  };
}

const leafMap = L.map('map', {
  crs:L.CRS.Simple, minZoom:-3, maxZoom:4,
  zoomSnap:0.25, attributionControl:false
});
let mapW=900, mapH=720, T=null;
function frac2ll(fx,fy) { return [-(fy*mapH), fx*mapW]; }

const state = [0,1].map(() => ({
  kalman: makeKalman(CFG.kalmanR, CFG.kalmanQ),
  movAvg: makeMovAvg(5),
  marker:null, pulse:null, lastT:null
}));

let totalMsg=0, msgWindow=0;
setInterval(()=>{
  document.getElementById('sb-rate').textContent = msgWindow.toFixed(1);
  msgWindow=0;
}, 1000);

function addPoint(tagIdx, rx, ry) {
  if (!T) return;
  const now = Date.now();
  const s = state[tagIdx];
  const dt = s.lastT ? (now - s.lastT) / 1000 : 0.1;
  s.lastT = now;

  let sx=rx, sy=ry, vx=0, vy=0;
  if (CFG.smoothing === 'kalman') {
    const k = s.kalman.update(rx, ry, dt);
    sx=k.x; sy=k.y; vx=k.vx||0; vy=k.vy||0;
  } else if (CFG.smoothing === 'moving_avg') {
    const m = s.movAvg.update(rx, ry);
    sx=m.x; sy=m.y;
  }

  redraw(tagIdx, rx, ry, sx, sy, vx, vy, now);
  totalMsg++; msgWindow++;
  document.getElementById('h-msgs').textContent = totalMsg;
  document.getElementById('sb-last').textContent = new Date(now).toLocaleTimeString();
}

function redraw(tagIdx, rx, ry, sx, sy, vx, vy, now) {
  if (!T) return;
  const s = state[tagIdx];
  const isT1 = tagIdx === 0;
  const cDot   = isT1 ? '#dc2626' : '#0369a1';
  const cGlow  = isT1 ? 'rgba(220,38,38,.2)' : 'rgba(3,105,161,.2)';
  const label  = isT1 ? CFG.tag1Label : CFG.tag2Label;
  const hpfx   = isT1 ? '1' : '2';

  const {fx,fy} = applyT(T, sx, sy);
  const ll = frac2ll(fx, fy);
  const speed = Math.hypot(vx, vy);

  const icon = L.divIcon({className:'',iconSize:[26,26],iconAnchor:[13,13],
    html:`<div style="width:26px;height:26px;border-radius:50%;
      background:${cDot};border:3px solid #fff;
      box-shadow:0 0 0 4px ${cGlow},0 2px 12px ${cGlow}"></div>`
  });

  if (s.marker) leafMap.removeLayer(s.marker);
  if (s.pulse)  leafMap.removeLayer(s.pulse);

  s.marker = L.marker(ll, {icon, zIndexOffset:1000-tagIdx})
    .bindPopup(`<b style="color:${cDot};font-family:'Outfit',sans-serif">📍 ${label}</b><br>
      <span style="color:#6b7a99;font-family:'DM Mono',monospace;font-size:10px;line-height:1.9">
      RAW &nbsp; (${rx.toFixed(3)} m, ${ry.toFixed(3)} m)<br>
      SMOOTH (${sx.toFixed(3)} m, ${sy.toFixed(3)} m)<br>
      SPEED &nbsp;${(speed*100).toFixed(1)} cm/s<br>
      ⏱ ${new Date(now).toLocaleTimeString()}</span>`)
    .addTo(leafMap);

  s.pulse = L.circle(ll, {
    radius: Math.min(mapW,mapH)*.025,
    color:cDot, fillColor:cDot, fillOpacity:.06, weight:1.5
  }).addTo(leafMap);

  document.getElementById(`h${hpfx}-rx`).textContent  = rx.toFixed(3)+' m';
  document.getElementById(`h${hpfx}-ry`).textContent  = ry.toFixed(3)+' m';
  document.getElementById(`h${hpfx}-sx`).textContent  = sx.toFixed(3)+' m';
  document.getElementById(`h${hpfx}-sy`).textContent  = sy.toFixed(3)+' m';
  document.getElementById(`h${hpfx}-spd`).textContent = (speed*100).toFixed(1)+' cm/s';
}

const imgEl = new Image();
imgEl.onload = function() {
  mapW = imgEl.naturalWidth  || 900;
  mapH = imgEl.naturalHeight || 720;
  const bounds = [[-mapH,0],[0,mapW]];
  L.imageOverlay(IMG_SRC, bounds, {opacity:1}).addTo(leafMap);
  leafMap.invalidateSize({animate:false});
  setTimeout(()=>leafMap.fitBounds(bounds,{padding:[16,16]}), 60);

  T = computeTransform(CFG.anchors);

  Object.entries(CFG.anchors).forEach(([id, a]) => {
    const ll = frac2ll(a.img_x, a.img_y);
    const icon = L.divIcon({className:'',iconSize:[30,30],iconAnchor:[15,15],
      html:`<div style="width:30px;height:30px;border-radius:6px;background:#eff6ff;border:2.5px solid #1a56db;
        box-shadow:0 0 0 4px rgba(26,86,219,.12),0 2px 8px rgba(26,86,219,.2);
        display:flex;align-items:center;justify-content:center;
        color:#1a56db;font-size:9px;font-weight:700;font-family:'DM Mono',monospace">
        ${id}</div>`
    });
    L.marker(ll, {icon})
      .bindPopup(`<b style="color:#1a56db">${a.label}</b><br>
        <span style="color:#6b7a99;font-family:'DM Mono',monospace;font-size:10px;line-height:1.9">
        Addr: ${a.addr}<br>
        Image: (${a.img_x.toFixed(2)}, ${a.img_y.toFixed(2)})<br>
        Real: (${a.real_x.toFixed(2)} m, ${a.real_y.toFixed(2)} m)</span>`)
      .addTo(leafMap);
    L.circle(ll,{radius:Math.min(mapW,mapH)*.02,color:'#1a56db',fillColor:'#1a56db',fillOpacity:.04,weight:1,dashArray:'4,6'}).addTo(leafMap);
  });

  const akeys = Object.keys(CFG.anchors);
  for (let i=0;i<akeys.length;i++) for (let j=i+1;j<akeys.length;j++) {
    const ai=CFG.anchors[akeys[i]], aj=CFG.anchors[akeys[j]];
    L.polyline([frac2ll(ai.img_x,ai.img_y),frac2ll(aj.img_x,aj.img_y)],
      {color:'#bfdbfe',weight:1,dashArray:'4,8',opacity:.8}).addTo(leafMap);
  }

  document.getElementById('h-t1-name').textContent = CFG.tag1Label.toUpperCase();
  document.getElementById('h-t2-name').textContent = CFG.tag2Label.toUpperCase();
  document.getElementById('leg-t1').textContent    = CFG.tag1Label+' (live)';
  document.getElementById('leg-t2').textContent    = CFG.tag2Label+' (live)';
  document.getElementById('sb-broker').textContent = CFG.broker+':'+CFG.port;
  document.getElementById('sb-t1').textContent     = CFG.topic;

  if (CFG.tag2Enabled && CFG.topic2) {
    document.getElementById('hud-t2').style.display      = 'block';
    document.getElementById('leg-t2-wrap').style.display = 'block';
    document.getElementById('sb-t2').textContent         = CFG.topic2;
  } else {
    document.getElementById('sb-t2-wrap').style.display  = 'none';
  }

  startMQTT();
};
imgEl.onerror = ()=>console.error('Floor plan failed');
imgEl.src = IMG_SRC;

function startMQTT() {
  const proto = (location.protocol==='https:'||CFG.port===8084) ? 'wss' : 'ws';
  const url = `${proto}://${CFG.broker}:${CFG.port}/mqtt`;
  const client = mqtt.connect(url, {
    clientId:'uwb_'+Math.random().toString(16).slice(2,10),
    clean:true, reconnectPeriod:4000, connectTimeout:10000, keepalive:30
  });

  const setDot = cls =>
    document.getElementById('h-status').innerHTML = `<span class="dot ${cls}"></span>`;

  client.on('connect',    ()=>{ setDot('dot-g'); client.subscribe(CFG.topic); if(CFG.tag2Enabled&&CFG.topic2) client.subscribe(CFG.topic2); });
  client.on('reconnect',  ()=>setDot('dot-y'));
  client.on('disconnect', ()=>setDot('dot-r'));
  client.on('offline',    ()=>setDot('dot-r'));
  client.on('error',      ()=>setDot('dot-r'));

  client.on('message', (topic, payload) => {
    try {
      const d = JSON.parse(payload.toString());
      const rx = parseFloat(d.x), ry = parseFloat(d.y);
      if (isNaN(rx)||isNaN(ry)) return;
      if (topic===CFG.topic)       addPoint(0,rx,ry);
      else if (topic===CFG.topic2) addPoint(1,rx,ry);
    } catch(e) { console.warn('Parse:',e); }
  });

  window.clearDots = ()=>{
    state.forEach(s=>{
      s.kalman.reset(); s.movAvg.reset(); s.lastT=null;
      [s.marker,s.pulse].forEach(l=>{ if(l) leafMap.removeLayer(l); });
      s.marker=null; s.pulse=null;
    });
  };
}
</script>
</body>
</html>""".replace("__CFG_JSON__", cfg_json).replace("__IMG_SRC__", img_src)


def build_access_people_component() -> str:
    """ESP32-CAM viewer + live face-recognition results panel (MQTT-driven),
    with a bounding-box overlay drawn on top of the MJPEG stream."""
    cfg = {
        "broker":       st.session_state.mqtt_broker,
        "port":         st.session_state.mqtt_port,
        "camUrl":       st.session_state.esp32_cam_url,
        "faceTopic":    st.session_state.mqtt_face_topic,
        "streamWidth":  st.session_state.cam_stream_width,
        "streamHeight": st.session_state.cam_stream_height,
    }
    cfg_json = json.dumps(cfg)

    return """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<script src="https://unpkg.com/mqtt@5.3.4/dist/mqtt.min.js"></script>
<style>
* { margin:0;padding:0;box-sizing:border-box; }
body { background:#f0f4f8;font-family:'Outfit',sans-serif,'DM Mono',monospace;overflow:hidden; }
.wrap { display:flex; gap:16px; padding:4px; height:600px; }
.cam-panel {
  flex:1.4; background:#0f1f35; border-radius:14px; position:relative; overflow:hidden;
  display:flex; align-items:center; justify-content:center; box-shadow:0 4px 20px rgba(0,0,0,.12);
}
.cam-media {
  position:relative; width:100%; height:100%; display:flex; align-items:center; justify-content:center;
}
.cam-panel img { width:100%; height:100%; object-fit:contain; display:none; }
.cam-placeholder {
  color:#6b85ad; font-family:'DM Mono',monospace; font-size:12px; text-align:center; padding:20px;
  display:flex; flex-direction:column; align-items:center; gap:10px;
}
.cam-btn {
  background:#1a56db; color:#fff; border:none; border-radius:8px; padding:9px 20px;
  font-family:'DM Mono',monospace; font-size:12px; cursor:pointer; transition:.18s;
}
.cam-btn:hover { background:#1140a8; transform:translateY(-1px); }
.cam-tag {
  position:absolute; top:10px; left:10px; background:rgba(0,0,0,.45); color:#9fd6ff;
  font-family:'DM Mono',monospace; font-size:9px; letter-spacing:1.5px; padding:5px 10px;
  border-radius:6px; display:none; z-index:5;
}
.cam-live-dot { display:inline-block;width:7px;height:7px;border-radius:50%;background:#22c55e;margin-right:6px;animation:blink 1.4s ease-in-out infinite; vertical-align:middle;}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.25} }

.bbox-overlay {
  position:absolute; border:3px solid #22c55e; border-radius:3px;
  box-shadow:0 0 0 2px rgba(34,197,94,.18);
  pointer-events:none; display:none; z-index:4;
}
.bbox-overlay.unverified { border-color:#ef4444; box-shadow:0 0 0 2px rgba(239,68,68,.18); }
.bbox-label {
  position:absolute; bottom:100%; left:-3px; transform:translateY(1px);
  background:#22c55e; color:#fff; font-family:'DM Mono',monospace; font-size:10px; font-weight:700;
  padding:3px 8px; border-radius:4px 4px 0 0; white-space:nowrap;
}
.bbox-overlay.unverified .bbox-label { background:#ef4444; }

.results-panel {
  flex:1; background:#fff; border:1px solid #dde3ec; border-radius:14px; padding:16px;
  display:flex; flex-direction:column; box-shadow:0 4px 20px rgba(0,0,0,.06); overflow:hidden;
}
.res-title { font-family:'Outfit',sans-serif; font-size:12px; font-weight:700; color:#0f1f35;
  letter-spacing:1px; text-transform:uppercase; display:flex; align-items:center; gap:8px;
  padding-bottom:10px; margin-bottom:10px; border-bottom:2px solid #eff6ff; }
.status-dot { width:8px;height:8px;border-radius:50%; }
.status-dot.g { background:#22c55e; box-shadow:0 0 8px #22c55e88; animation:blink 1.4s ease-in-out infinite; }
.status-dot.r { background:#ef4444; }
.status-dot.y { background:#f59e0b; animation:blink 1.4s ease-in-out infinite; }

.current-result {
  border-radius:12px; padding:18px; margin-bottom:14px; text-align:center; transition:.25s;
  background:#f8fafc; border:2px solid #dde3ec;
}
.current-result.authorized { background:#f0fdf4; border-color:#86efac; }
.current-result.unverified { background:#fef2f2; border-color:#fca5a5; }
.cr-name { font-family:'Outfit',sans-serif; font-size:18px; font-weight:700; color:#0f1f35; margin-bottom:4px; }
.cr-name.authorized { color:#166534; }
.cr-name.unverified { color:#991b1b; }
.cr-sub { font-family:'DM Mono',monospace; font-size:10px; color:#8fa8c8; letter-spacing:1px; }
.cr-conf { font-family:'DM Mono',monospace; font-size:11px; color:#1a56db; font-weight:700; margin-top:6px; }

.log-title { font-family:'DM Mono',monospace; font-size:9px; color:#aab8cc; letter-spacing:1.5px;
  text-transform:uppercase; margin-bottom:8px; }
.log-list { flex:1; overflow-y:auto; display:flex; flex-direction:column; gap:6px; }
.log-row { display:flex; justify-content:space-between; align-items:center; padding:7px 10px;
  border-radius:8px; background:#f8fafc; font-family:'DM Mono',monospace; font-size:10px; }
.log-row.authorized { background:#f0fdf4; }
.log-row.unverified { background:#fef2f2; }
.log-name { font-weight:700; color:#1a2333; }
.log-name.authorized { color:#166534; }
.log-name.unverified { color:#991b1b; }
.log-time { color:#aab8cc; }
</style>
</head>
<body>
<div class="wrap">
  <div class="cam-panel">
    <div class="cam-media" id="cam-media">
      <div class="cam-tag" id="cam-tag"><span class="cam-live-dot"></span>ESP32-CAM</div>
      <img id="cam-img"/>
      <div class="bbox-overlay" id="bbox-overlay"><span class="bbox-label" id="bbox-label"></span></div>
      <div class="cam-placeholder" id="cam-placeholder">
        <div style="font-size:28px">📷</div>
        <div>Camera stream is stopped</div>
        <button class="cam-btn" id="cam-toggle" onclick="toggleCam()">▶ Open ESP32-CAM</button>
      </div>
    </div>
  </div>

  <div class="results-panel">
    <div class="res-title">
      <span class="status-dot r" id="mqtt-dot"></span>
      Face Recognition Results
    </div>

    <div class="current-result" id="current-result">
      <div class="cr-name" id="cr-name">No data yet</div>
      <div class="cr-sub" id="cr-sub">Waiting for recognition results…</div>
      <div class="cr-conf" id="cr-conf"></div>
    </div>

    <div class="log-title">Recent events</div>
    <div class="log-list" id="log-list"></div>
  </div>
</div>

<script>
const CFG = __CFG_JSON__;
let camOpen = false;

function toggleCam() {
  const img = document.getElementById('cam-img');
  const placeholder = document.getElementById('cam-placeholder');
  const tag = document.getElementById('cam-tag');
  camOpen = !camOpen;
  if (camOpen) {
    img.src = CFG.camUrl;
    img.style.display = 'block';
    placeholder.style.display = 'none';
    tag.style.display = 'block';
  } else {
    img.src = '';
    img.style.display = 'none';
    placeholder.style.display = 'flex';
    tag.style.display = 'none';
    hideBbox();
  }
}

document.getElementById('cam-img').onerror = function() {
  if (!camOpen) return;
  const placeholder = document.getElementById('cam-placeholder');
  placeholder.innerHTML = `
    <div style="font-size:28px">⚠️</div>
    <div>Could not reach camera at<br><span style="color:#9fd6ff">${CFG.camUrl}</span></div>
    <button class="cam-btn" onclick="retryCam()">↻ Retry</button>`;
  placeholder.style.display = 'flex';
  document.getElementById('cam-img').style.display = 'none';
  document.getElementById('cam-tag').style.display = 'none';
  camOpen = false;
  hideBbox();
};
function retryCam() { camOpen = false; toggleCam(); }

const MAX_LOG = 12;
let logEntries = [];

function setMqttDot(cls) {
  document.getElementById('mqtt-dot').className = 'status-dot ' + cls;
}

function renderCurrent(d) {
  const box = document.getElementById('current-result');
  const nameEl = document.getElementById('cr-name');
  const subEl = document.getElementById('cr-sub');
  const confEl = document.getElementById('cr-conf');

  box.className = 'current-result';
  nameEl.className = 'cr-name';

  if (d.status === 'authorized') {
    box.classList.add('authorized');
    nameEl.classList.add('authorized');
    nameEl.textContent = '✓ ' + d.name;
    subEl.textContent = 'Authorized · ' + new Date(d.ts*1000).toLocaleTimeString();
  } else if (d.status === 'unverified') {
    box.classList.add('unverified');
    nameEl.classList.add('unverified');
    nameEl.textContent = '? Unverified';
    subEl.textContent = 'Face detected, not matched · ' + new Date(d.ts*1000).toLocaleTimeString();
  } else {
    nameEl.textContent = 'No one detected';
    subEl.textContent = 'Last checked ' + new Date(d.ts*1000).toLocaleTimeString();
  }
  confEl.textContent = d.status === 'no_face' ? '' : ('Similarity: ' + (d.similarity*100).toFixed(1) + '%');
}

function addLogEntry(d) {
  if (d.status === 'no_face') return;
  logEntries.unshift(d);
  if (logEntries.length > MAX_LOG) logEntries.pop();
  const list = document.getElementById('log-list');
  list.innerHTML = logEntries.map(e => `
    <div class="log-row ${e.status}">
      <span class="log-name ${e.status}">${e.status === 'authorized' ? '✓' : '?'} ${e.name === 'Unknown' ? 'Unverified' : e.name}</span>
      <span class="log-time">${new Date(e.ts*1000).toLocaleTimeString()}</span>
    </div>
  `).join('');
}

// ---- Bounding box overlay ----
// The MJPEG <img> uses object-fit:contain, so the rendered image may be
// letterboxed inside the element. We compute the actual displayed image
// rect (accounting for letterboxing) and scale bbox pixel coords from the
// source stream resolution (CFG.streamWidth x CFG.streamHeight) into that rect.
let bboxClearTimer = null;
const BBOX_AUTOCLEAR_MS = 1500; // no MQTT message with a bbox for this long -> hide box

function hideBbox() {
  document.getElementById('bbox-overlay').style.display = 'none';
  if (bboxClearTimer) { clearTimeout(bboxClearTimer); bboxClearTimer = null; }
}

function getDisplayedImageRect() {
  const mediaRect = document.getElementById('cam-media').getBoundingClientRect();
  const naturalW = CFG.streamWidth, naturalH = CFG.streamHeight;
  const containerW = mediaRect.width, containerH = mediaRect.height;

  const scale = Math.min(containerW / naturalW, containerH / naturalH);
  const dispW = naturalW * scale;
  const dispH = naturalH * scale;
  const offsetX = (containerW - dispW) / 2;
  const offsetY = (containerH - dispH) / 2;

  return { offsetX, offsetY, scale };
}

function drawBbox(d) {
  if (!camOpen || !d.bbox || d.bbox.length !== 4) return;
  const [x1, y1, x2, y2] = d.bbox;
  const { offsetX, offsetY, scale } = getDisplayedImageRect();

  const overlay = document.getElementById('bbox-overlay');
  const label = document.getElementById('bbox-label');

  overlay.style.left   = (offsetX + x1 * scale) + 'px';
  overlay.style.top    = (offsetY + y1 * scale) + 'px';
  overlay.style.width  = ((x2 - x1) * scale) + 'px';
  overlay.style.height = ((y2 - y1) * scale) + 'px';
  overlay.style.display = 'block';

  overlay.classList.toggle('unverified', d.status === 'unverified');

  if (d.status === 'authorized') {
    label.textContent = `${d.name} · ${(d.similarity*100).toFixed(0)}%`;
  } else {
    label.textContent = `Verifying… ${(d.similarity*100).toFixed(0)}%`;
  }

  if (bboxClearTimer) clearTimeout(bboxClearTimer);
  bboxClearTimer = setTimeout(hideBbox, BBOX_AUTOCLEAR_MS);
}

// Reposition the box if the panel resizes (e.g. browser window resize)
window.addEventListener('resize', () => {
  const overlay = document.getElementById('bbox-overlay');
  if (overlay.style.display === 'block' && window._lastBboxData) {
    drawBbox(window._lastBboxData);
  }
});

function startMQTT() {
  const proto = (location.protocol==='https:'||CFG.port===8084) ? 'wss' : 'ws';
  const url = `${proto}://${CFG.broker}:${CFG.port}/mqtt`;
  const client = mqtt.connect(url, {
    clientId:'access_'+Math.random().toString(16).slice(2,10),
    clean:true, reconnectPeriod:4000, connectTimeout:10000, keepalive:30
  });

  client.on('connect',    ()=>{ setMqttDot('g'); client.subscribe(CFG.faceTopic); });
  client.on('reconnect',  ()=>setMqttDot('y'));
  client.on('disconnect', ()=>setMqttDot('r'));
  client.on('offline',    ()=>setMqttDot('r'));
  client.on('error',      ()=>setMqttDot('r'));

  client.on('message', (topic, payload) => {
    if (topic !== CFG.faceTopic) return;
    try {
      const d = JSON.parse(payload.toString());
      if (!d.ts) d.ts = Date.now()/1000;
      renderCurrent(d);
      addLogEntry(d);
      if (d.bbox) {
        window._lastBboxData = d;
        drawBbox(d);
      }
    } catch(e) { console.warn('Parse:', e); }
  });
}

startMQTT();
</script>
</body>
</html>""".replace("__CFG_JSON__", cfg_json)


from streamlit.components.v1 import html as st_html

if st.session_state.active_view == "tracker":
    st_html(build_component(), height=640, scrolling=False)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    c1, c2, c3, _ = st.columns([1,1,2,4])
    with c1:
        if st.button("🗑 Clear Dot"):
            st.toast("Dot cleared!", icon="🗑️")
    with c2:
        if st.button("🔄 Reload"):
            st.rerun()
    with c3:
        st.markdown(
            "<div style='padding-top:8px;color:#8fa8c8;font-size:10px;font-family:monospace'>"
            "4-anchor UWB · Kalman · Dot only · Axis-swap corrected</div>",
            unsafe_allow_html=True
        )

elif st.session_state.active_view == "access_people":
    st.markdown(
        "<div style='font-family:Outfit,sans-serif;font-size:14px;font-weight:700;color:#0f1f35;margin-bottom:8px'>"
        "🧑‍🤝‍🧑 Access People</div>",
        unsafe_allow_html=True
    )
    st_html(build_access_people_component(), height=640, scrolling=False)
    st.markdown(
        "<div style='padding-top:8px;color:#8fa8c8;font-size:10px;font-family:monospace'>"
        "ESP32-CAM stream + live face recognition, with bounding box overlay · "
        "Recognition runs as a background thread inside this app — start it from the "
        "'Access People / Camera' panel in the sidebar.</div>",
        unsafe_allow_html=True
    )

elif st.session_state.active_view == "stock_tracking":
    st.markdown(
        "<div style='font-family:Outfit,sans-serif;font-size:14px;font-weight:700;color:#0f1f35;margin-bottom:8px'>"
        "📦 Stock Tracking</div>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style="background:#fff;border:1px dashed #dde3ec;border-radius:14px;padding:60px 20px;
      text-align:center;color:#8fa8c8;font-family:'DM Mono',monospace;font-size:12px">
      📦<br><br>Stock Tracking — coming soon
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.active_view == "enroll":
    st.markdown(
        "<div style='font-family:Outfit,sans-serif;font-size:14px;font-weight:700;color:#0f1f35;margin-bottom:8px'>"
        "🪪 Enroll Employee</div>",
        unsafe_allow_html=True
    )

    is_enrolling = _ENROLL_STATE["running"]
    is_recognizing = _RECOGNIZER_STATE["running"]

    ec1, ec2 = st.columns([1.4, 1])

    with ec1:
        st.markdown(
            f"<img src='{st.session_state.esp32_cam_url}' style='width:100%;border-radius:14px;"
            f"background:#0f1f35;display:block' onerror=\"this.style.display='none'\"/>",
            unsafe_allow_html=True
        )
        st.caption("Live camera preview — position your face per the current step below.")

    with ec2:
        if is_recognizing:
            st.warning("Recognition is currently running. Stop it from the Access People sidebar panel before enrolling — both need exclusive use of the camera.")
        elif not is_enrolling:
            st.session_state.enroll_name = st.text_input(
                "Employee name", value=st.session_state.enroll_name, placeholder="e.g. Elvis")
            disabled = not st.session_state.enroll_name.strip()
            if st.button("▶ Start Enrollment", use_container_width=True, disabled=disabled, type="primary"):
                ok = start_enrollment(st.session_state.enroll_name.strip())
                if not ok and _ENROLL_STATE["error"]:
                    st.error(_ENROLL_STATE["error"])
                st.rerun()
            if disabled:
                st.caption("Enter a name to begin.")
        else:
            step = _ENROLL_STATE["step"]
            st.markdown(f"**Enrolling: {st.session_state.enroll_name}**")
            st.progress(min(step, 9) / 9)

            if step < len(ENROLL_STEPS):
                st.markdown(f"##### Step {step + 1}/9 — {ENROLL_STEPS[step]}")
            else:
                st.markdown("##### Finalizing…")

            if _ENROLL_STATE["last_capture_ok"] is True:
                st.success("Captured ✓")
            elif _ENROLL_STATE["last_capture_ok"] is False:
                st.error("No single clear face detected — reposition and try again.")

            if step < 8:
                if st.button("📸 Capture this angle", use_container_width=True, type="primary"):
                    request_capture()
                    time.sleep(0.4)  # give the worker a moment to process before rerunning
                    st.rerun()
            else:
                st.info(f"Blink to finish · live EAR: {_ENROLL_STATE['avg_ear']:.3f} (threshold {EAR_THRESHOLD})")

            if st.button("⏹ Cancel", use_container_width=True):
                stop_enrollment()
                st.rerun()

            if _ENROLL_STATE["saved_name"]:
                st.success(f"✓ {_ENROLL_STATE['saved_name']} enrolled and saved to {st.session_state.faces_pkl_path}!")
                st.session_state.enroll_name = ""
                _ENROLL_STATE["saved_name"] = None

            if not is_enrolling and step < len(ENROLL_STEPS) and _ENROLL_STATE["error"]:
                st.error(_ENROLL_STATE["error"])

            # Light auto-refresh while enrollment is active so the live EAR value
            # and step progress update without the person needing to interact.
            if is_enrolling:
                time.sleep(0.6)
                st.rerun()

st.markdown("""
<div style="margin-top:10px;padding:10px 0;border-top:1px solid #dde3ec;
  color:#aab8cc;font-size:9px;letter-spacing:2px;text-align:center;font-family:monospace">
  DW1000 UWB · ESP32 · MQTT WebSocket · Leaflet CRS.Simple · Kalman Filter ·
  4-Anchor Trilateration · Axis-swap corrected · Calibration persisted in localStorage ·
  Live bbox overlay for Access People
</div>
""", unsafe_allow_html=True)
