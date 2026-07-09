"""
Face Recognition MJPEG Backend
================================
This is your working notebook script, converted so it can feed a web page
instead of a local cv2.imshow() window. cv2.imshow() opens a native OS
window — it cannot render inside Streamlit or any browser, whether run
locally or deployed. This script keeps your exact recognition logic
(InsightFace + cosine similarity + 5-frame majority vote) but re-serves
each annotated frame as an MJPEG HTTP stream that the "Access People"
Streamlit page displays via a plain <img> tag.

Run this as its OWN process, separate from `streamlit run app.py`:

    python face_recognition_server.py

Then set the "Face-recognition backend URL" on the Access People page to:

    http://192.168.100.16:5001/video_feed

(use the machine's real IP, not 192.168.100.103 — that's your ESP32-CAM's
address, not this server's)
"""

import time
import pickle
import threading
from collections import Counter
import os
import cv2
import numpy as np
from flask import Flask, Response, jsonify
from insightface.app import FaceAnalysis

# ── Config ───────────────────────────────────────────────────
ESP32_STREAM_URL = "http://192.168.100.103:81/stream"  # ESP32-CAM MJPEG stream


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_DB_PATH = os.path.join(BASE_DIR, "faces.pkl")
BUFFER_SIZE        = 5
MATCH_THRESHOLD    = 0.40
HOST, PORT         = "0.0.0.0", 5001

flask_app = Flask(__name__)

# ── Load model + database once at startup ──────────────────────
print("Loading AI Model...")
face_app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['detection', 'recognition'],
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
face_app.prepare(ctx_id=0, det_size=(640, 640))

try:
    with open(FACE_DB_PATH, "rb") as f:
        face_db = pickle.load(f)
    print(f"[OK] Loaded database with workers: {list(face_db.keys())}")
except FileNotFoundError:
    print(f"[!] '{FACE_DB_PATH}' not found — run your enrollment script first.")
    face_db = {}

# ── Shared state written by the recognition thread, read by Flask routes ──
_lock = threading.Lock()
latest_frame_jpeg = None
latest_status = {"name": "Unknown", "similarity": 0.0, "ts": 0}
frame_buffer = []


def recognition_loop():
    global latest_frame_jpeg, latest_status, frame_buffer

    cap = cv2.VideoCapture(ESP32_STREAM_URL)
    if not cap.isOpened():
        print("[!] Could not connect to ESP32-CAM stream. Check ESP32_STREAM_URL.")
        return
    print("🎥 Recognition loop started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        faces = face_app.get(frame)

        if len(faces) == 0:
            frame_buffer.append("Unknown")
            if len(frame_buffer) > BUFFER_SIZE:
                frame_buffer.pop(0)

        for face in faces:
            live_embedding = face.embedding
            live_norm = live_embedding / np.linalg.norm(live_embedding)

            best_name, best_similarity = "Unknown", -1
            for name, stored_embedding in face_db.items():
                stored_norm = stored_embedding / np.linalg.norm(stored_embedding)
                similarity = float(np.dot(stored_norm, live_norm))
                if similarity > best_similarity:
                    best_similarity = similarity
                    if similarity >= MATCH_THRESHOLD:
                        best_name = name

            frame_buffer.append(best_name)
            if len(frame_buffer) > BUFFER_SIZE:
                frame_buffer.pop(0)

            majority_vote = Counter(frame_buffer).most_common(1)[0]
            matched_name = majority_vote[0] if majority_vote[1] >= 3 else "Unknown"

            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

            if matched_name != "Unknown":
                color = (0, 255, 0)
                label = f"{matched_name} ({best_similarity * 100:.1f}%)"
            else:
                color = (0, 0, 255)
                label = f"Verifying... ({best_similarity * 100:.1f}%)"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, y1 - 30), (x2, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            with _lock:
                latest_status = {
                    "name": matched_name,
                    "similarity": round(best_similarity * 100, 1),
                    "ts": time.time(),
                }

        ok, buf = cv2.imencode('.jpg', frame)
        if ok:
            with _lock:
                latest_frame_jpeg = buf.tobytes()


def mjpeg_generator():
    while True:
        with _lock:
            frame_bytes = latest_frame_jpeg
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # ~30fps cap


@flask_app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


@flask_app.route('/api/status')
def status():
    with _lock:
        return jsonify(latest_status)


@flask_app.route('/')
def index():
    return "Face recognition backend running. See /video_feed and /api/status."


if __name__ == '__main__':
    worker = threading.Thread(target=recognition_loop, daemon=True)
    worker.start()
    print(f"Serving MJPEG stream on http://{HOST}:{PORT}/video_feed")
    flask_app.run(host=HOST, port=PORT, threaded=True)
