"""
Access People
=============
Opens as a separate tab (Streamlit multipage routing) from the main
UWB dashboard's "👥 Access People" fast-action button.

Shows:
  1. A button that opens the ESP32-CAM's own web UI in a new tab.
  2. The live, annotated face-recognition video feed, streamed as MJPEG
     from face_recognition_server.py (a separate Flask process — see
     that file for how to run it).

This page does NOT run face recognition itself. cv2.imshow() from a
notebook/script can't render inside a web page, so recognition runs in
its own backend process that re-serves frames as an MJPEG HTTP stream,
which a plain <img> tag can display natively in any browser.
"""

import streamlit as st

st.set_page_config(page_title="Access People", page_icon="👥", layout="wide")

# Defaults (kept in sync with the sidebar setting on the main page, if set)
if "esp32_cam_url" not in st.session_state:
    st.session_state.esp32_cam_url = "http://192.168.100.103"
if "face_feed_url" not in st.session_state:
    st.session_state.face_feed_url = "https://vigilant-umbrella-q9vgvr77jjhg4p-5001.app.github.dev/video_feed"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Outfit:wght@400;600;700&display=swap');
html, [class*="css"] { font-family: 'Outfit', sans-serif !important; }
.stApp { background: #f0f4f8; color: #1a2333; }
h1,h2,h3 { font-family:'Outfit',sans-serif !important; color:#0f1f35 !important; font-weight:700 !important; }
.stButton>button, a[data-testid="stLinkButton"]>button, .stLinkButton>a {
  background:#1a56db !important; color:#fff !important; border:none !important;
  border-radius:7px !important; font-family:'DM Mono',monospace !important; font-size:12px !important;
}
.video-frame { border-radius:12px; border:1px solid #dde3ec; overflow:hidden; background:#0f1f35; }
.status-pill { display:inline-block; padding:3px 10px; border-radius:999px; font-family:'DM Mono',monospace; font-size:10px; letter-spacing:1px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;gap:12px;padding-bottom:4px">
  <div style="background:#1a56db;border-radius:10px;width:40px;height:40px;display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0">👥</div>
  <div>
    <div style="font-size:20px;font-weight:700;color:#0f1f35;line-height:1.1">Access People</div>
    <div style="font-family:'DM Mono',monospace;font-size:9px;color:#8fa8c8;letter-spacing:2px;margin-top:1px">ESP32-CAM · INSIGHTFACE · LIVE RECOGNITION</div>
  </div>
</div>
<div style="height:2px;background:linear-gradient(90deg,#1a56db,#38bdf8,transparent);border-radius:2px;margin:10px 0 16px"></div>
""", unsafe_allow_html=True)

with st.expander("⚙️ Stream sources", expanded=False):
    st.session_state.esp32_cam_url = st.text_input(
        "ESP32-CAM web UI URL", value=st.session_state.esp32_cam_url,
        help="The camera's own root web page (usually port 80), NOT the raw /stream URL."
    )
    st.session_state.face_feed_url = st.text_input(
        "Face-recognition backend URL", value=st.session_state.face_feed_url,
        help="The /video_feed route served by face_recognition_server.py"
    )

col1, col2 = st.columns([1, 3])
with col1:
    st.link_button("📷 Open ESP32-CAM", st.session_state.esp32_cam_url, use_container_width=True)
with col2:
    st.caption("Opens the camera's own web interface (snapshot controls, resolution, etc.) in a new tab.")

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
st.markdown("### 🎯 Live Face Recognition")
st.caption("🟢 Green box = authorized match · 🔴 Red box = unknown / verifying — labels are burned into the video by the backend.")

st.markdown(
    f"""<div class="video-frame">
        <img src="{st.session_state.face_feed_url}" style="width:100%;display:block" />
    </div>""",
    unsafe_allow_html=True
)

st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
if st.button("🔄 Refresh feed"):
    st.rerun()

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown("""
- Face recognition itself runs in **`face_recognition_server.py`**, a separate process (not inside Streamlit).
  Run it with: `python face_recognition_server.py`
- It pulls frames from the ESP32-CAM stream, runs InsightFace recognition + the 5-frame majority-vote
  logic from your notebook, draws the name/confidence box on each frame, and re-serves the result as
  an MJPEG stream at `/video_feed` — the `<img>` tag above just points at that URL.
- Running it as its own process means the GPU model loads **once** and keeps running, instead of
  reloading every time Streamlit reruns the script (which happens on every widget interaction).
""")
