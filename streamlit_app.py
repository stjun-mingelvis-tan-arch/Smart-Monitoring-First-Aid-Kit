"""
ESP32 UWB Indoor Position Tracker
===================================
• Direct MQTT WebSocket in browser (Leaflet + MQTT.js)
• 4 UWB anchors (NW, NE, SW, SE) with persistent calibration via localStorage
• Light, clean blueprint aesthetic
• Kalman filter smoothing in JS
• Dual tag support
• Fast-action buttons: Access People (face recognition tab) · Stock Tracking (placeholder)

FIXES v2:
  - Axis swap fixed: UWB real_x/real_y swapped in affine solver so NW→NW, SE→SE
  - Anchor real coords updated to match physical layout after axis correction
  - Trail removed — only live dot + pulse circle shown

v3:python3 -c "
import base64
  - Added fast-action row (Access People / Stock Tracking)
  - Access People opens pages/1_Access_People.py in a new browser tab
"""

import streamlit as st
import base64
import json

st.set_page_config(
    page_title="UWB Tracker",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_defaults = {
    "anchors": {
        # real_x = UWB X axis, real_y = UWB Y axis
        # After axis-swap fix, NW corner = low Y, high X → real_x=1.81, real_y=0.0
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
    # New: Access People / face-recognition backend settings
    "esp32_cam_url":   "http://192.168.100.103",
    "face_feed_url":   "http://192.168.100.103:5001/video_feed",
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

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
.stButton>button:disabled { background: #cbd5e1 !important; color: #94a3b8 !important; transform:none !important; box-shadow:none !important; cursor:not-allowed !important; }
a[data-testid="stLinkButton"] > button, .stLinkButton>a { background: #1a56db !important; color:#ffffff !important; border-radius:7px !important; font-family:'DM Mono',monospace !important; }
.stNumberInput input, .stTextInput input { background: #f8fafc !important; color: #1a2333 !important; border-color: #dde3ec !important; border-radius: 7px !important; font-family: 'DM Mono', monospace !important; }
.stSelectbox > div { background: #f8fafc !important; }
hr { border-color: #dde3ec !important; }
.block-container { padding-top: 1.2rem; }
.fast-action-label { font-family:'DM Mono',monospace; font-size:9px; color:#8fa8c8; letter-spacing:2px; margin-bottom:6px; }
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

# ── Fast actions ─────────────────────────────────────────────
st.markdown('<div class="fast-action-label">FAST ACTIONS</div>', unsafe_allow_html=True)
fa1, fa2, fa_spacer = st.columns([1, 1, 4])
with fa1:
    # Opens pages/1_Access_People.py in a NEW browser tab.
    # Using raw HTML with target="_blank" instead of st.link_button,
    # since st.link_button was opening in the same tab.
    st.markdown(
        """
        <a href="/Access_People" target="_blank" rel="noopener noreferrer"
           style="display:inline-block;background:#1a56db;color:#fff;padding:6px 14px;
                  border-radius:7px;font-family:'DM Mono',monospace;font-size:12px;
                  text-decoration:none;text-align:center;width:100%;box-sizing:border-box">
          👥 Access People
        </a>
        """,
        unsafe_allow_html=True
    )
  
with fa2:
    st.button("📦 Stock Tracking", disabled=True, use_container_width=True,
               help="Coming soon — let me know what this should track and I'll wire it up")

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
        st.caption("Set image position (0–1 fraction) and real-world position in metres. Saved in your browser.")
        for aid, a in st.session_state.anchors.items():
            with st.expander(f"{aid} · {a['addr']} · {a['label']}", expanded=False):
                a['label'] = st.text_input("Label", a['label'], key=f"lbl_{aid}")
                a['addr']  = st.text_input("Short addr", a['addr'], key=f"addr_{aid}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**On image (0–1)**")
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

    with st.expander("👥 Access People — Stream Sources", expanded=False):
        st.caption("Used by the Access People tab (opens in a new browser tab).")
        st.session_state.esp32_cam_url = st.text_input("ESP32-CAM web UI URL", value=st.session_state.esp32_cam_url)
        st.session_state.face_feed_url = st.text_input("Face recognition backend URL", value=st.session_state.face_feed_url,
                                                          help="Points at the /video_feed route of face_recognition_server.py")


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

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/mqtt@5.3.4/dist/mqtt.min.js"></script>
<style>
* {{ margin:0;padding:0;box-sizing:border-box; }}
body {{ background:#f0f4f8;font-family:'DM Mono',monospace,sans-serif;overflow:hidden; }}
#map {{ width:100%;height:580px;background:#f5f8ff; }}
.leaflet-container {{ background:#f5f8ff !important; }}
.leaflet-control-zoom a {{ background:#fff !important;color:#1a56db !important;border-color:#dde3ec !important;font-weight:700; }}
.leaflet-popup-content-wrapper {{ background:#fff;border:1px solid #dde3ec;border-radius:10px;color:#1a2333;font-family:'Outfit',sans-serif;font-size:12px;box-shadow:0 4px 16px rgba(0,0,0,.1); }}
.leaflet-popup-tip {{ background:#fff; }}
#hud {{
  position:absolute;top:12px;right:12px;z-index:900;
  background:rgba(255,255,255,.96);border:1px solid #dde3ec;border-radius:12px;
  padding:14px 18px;min-width:210px;box-shadow:0 4px 20px rgba(0,0,0,.1);
  pointer-events:none;backdrop-filter:blur(10px);
}}
.hud-title {{ font-family:'Outfit',sans-serif;font-size:11px;font-weight:700;color:#0f1f35;letter-spacing:1px;text-transform:uppercase;padding-bottom:8px;margin-bottom:8px;border-bottom:2px solid #eff6ff;display:flex;align-items:center;gap:7px; }}
.hud-section {{ border-bottom:1px solid #f0f4f8;margin-bottom:8px;padding-bottom:8px; }}
.hud-section:last-child {{ border:none;margin:0;padding:0; }}
.hud-tag {{ font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;margin-bottom:5px; }}
.hud-row {{ display:flex;justify-content:space-between;align-items:center;margin:3px 0; }}
.hud-lbl {{ color:#8fa8c8;font-size:9px;letter-spacing:1px;text-transform:uppercase; }}
.hud-val {{ color:#1a56db;font-size:11px;font-weight:700;font-family:'DM Mono',monospace; }}
.dot {{ display:inline-block;width:8px;height:8px;border-radius:50%;flex-shrink:0; }}
.dot-g {{ background:#22c55e;box-shadow:0 0 8px #22c55e88;animation:blink 1.4s ease-in-out infinite; }}
.dot-r {{ background:#ef4444; }}
.dot-y {{ background:#f59e0b;animation:blink 1.4s ease-in-out infinite; }}
@keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:.25}} }}
#statusbar {{
  position:absolute;bottom:0;left:0;right:0;z-index:900;
  background:rgba(255,255,255,.94);border-top:1px solid #dde3ec;
  padding:7px 16px;display:flex;gap:24px;align-items:center;
  backdrop-filter:blur(8px);pointer-events:none;
}}
.sb-item {{ display:flex;flex-direction:column; }}
.sb-lbl {{ color:#aab8cc;font-size:8px;letter-spacing:1.5px;text-transform:uppercase; }}
.sb-val {{ color:#1a2333;font-size:11px;font-family:'DM Mono',monospace;font-weight:500;margin-top:1px; }}
#legend {{
  position:absolute;bottom:48px;left:12px;z-index:900;
  background:rgba(255,255,255,.94);border:1px solid #dde3ec;border-radius:10px;
  padding:10px 14px;box-shadow:0 2px 10px rgba(0,0,0,.08);pointer-events:none;
}}
.leg-row {{ display:flex;align-items:center;gap:8px;margin:3px 0; }}
.leg-txt {{ color:#6b7a99;font-size:10px;font-family:'Outfit',sans-serif; }}
#cal-toast {{
  position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:1000;
  background:#dcfce7;border:1px solid #86efac;border-radius:8px;padding:7px 18px;
  color:#166534;font-size:10px;letter-spacing:1px;display:none;box-shadow:0 2px 10px rgba(0,0,0,.1);
}}
#cal-loaded {{
  position:absolute;top:12px;left:50%;transform:translateX(-50%);z-index:1000;
  background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;padding:7px 18px;
  color:#1a56db;font-size:10px;letter-spacing:1px;display:none;box-shadow:0 2px 10px rgba(0,0,0,.1);
}}
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
const CFG = {cfg_json};
const IMG_SRC = "{img_src}";
const LS_ANCHORS = "uwb_calibration_v2";
const LS_BOARD   = "uwb_board_v1";

function loadCal() {{
  let loaded = false;
  try {{
    const raw = localStorage.getItem(LS_ANCHORS);
    if (raw) {{
      const saved = JSON.parse(raw);
      Object.keys(saved).forEach(id => {{
        if (!CFG.anchors[id]) return;
        const s = saved[id];
        ['img_x','img_y','real_x','real_y','label','addr'].forEach(k => {{
          if (s[k] !== undefined) CFG.anchors[id][k] = s[k];
        }});
      }});
      loaded = true;
    }}
    const braw = localStorage.getItem(LS_BOARD);
    if (braw) {{
      const bd = JSON.parse(braw);
      if (bd.boardX) CFG.boardX = bd.boardX;
      if (bd.boardY) CFG.boardY = bd.boardY;
    }}
  }} catch(e) {{ console.warn("Cal load failed:", e); }}
  if (loaded) {{
    document.getElementById('sb-cal-src').textContent = '📂 localStorage';
    const b = document.getElementById('cal-loaded');
    b.style.display = 'block';
    setTimeout(() => b.style.display = 'none', 2800);
  }} else {{
    document.getElementById('sb-cal-src').textContent = '📋 defaults';
  }}
}}

function saveCal() {{
  try {{
    const d = {{}};
    Object.entries(CFG.anchors).forEach(([id, a]) => {{
      d[id] = {{ img_x:a.img_x, img_y:a.img_y, real_x:a.real_x, real_y:a.real_y, label:a.label, addr:a.addr }};
    }});
    localStorage.setItem(LS_ANCHORS, JSON.stringify(d));
    localStorage.setItem(LS_BOARD, JSON.stringify({{ boardX:CFG.boardX, boardY:CFG.boardY }}));
    document.getElementById('sb-cal-src').textContent = '📂 localStorage';
    const t = document.getElementById('cal-toast');
    t.style.display = 'block';
    setTimeout(() => t.style.display = 'none', 2500);
  }} catch(e) {{ console.warn("Cal save failed:", e); }}
}}

function clearCal() {{
  localStorage.removeItem(LS_ANCHORS);
  localStorage.removeItem(LS_BOARD);
  document.getElementById('sb-cal-src').textContent = '📋 defaults';
  alert("Calibration cleared. Reload to restore defaults.");
}}python face_recognition_server.py 

loadCal();

// ── Affine transform ───────────────────────────────────────
// KEY FIX: swap real_x and real_y when building the solver rows.
// UWB firmware reports x=distance-along-one-wall, y=distance-along-other-wall,
// but the physical room layout has these axes transposed relative to the image.
// Swapping here means the affine fit maps (UWB_y, UWB_x) → image fractions,
// correcting the NW↔SE mirror.
function computeTransform(anchors) {{
  const keys = Object.keys(anchors);
  if (keys.length < 2) return null;

  function ne(rows, vals) {{
    let A=[[0,0,0],[0,0,0],[0,0,0]], b=[0,0,0];
    for (let i=0;i<rows.length;i++) {{
      for (let r=0;r<3;r++) for (let c=0;c<3;c++) A[r][c]+=rows[i][r]*rows[i][c];
      for (let r=0;r<3;r++) b[r]+=rows[i][r]*vals[i];
    }}
    for (let col=0;col<3;col++) {{
      let mx=col;
      for (let r=col+1;r<3;r++) if (Math.abs(A[r][col])>Math.abs(A[mx][col])) mx=r;
      [A[col],A[mx]]=[A[mx],A[col]]; [b[col],b[mx]]=[b[mx],b[col]];
      if (Math.abs(A[col][col])<1e-12) return null;
      for (let r=col+1;r<3;r++) {{
        const f=A[r][col]/A[col][col];
        for (let c=col;c<3;c++) A[r][c]-=f*A[col][c];
        b[r]-=f*b[col];
      }}
    }}
    const res=[0,0,0];
    for (let r=2;r>=0;r--) {{
      res[r]=b[r];
      for (let c=r+1;c<3;c++) res[r]-=A[r][c]*res[c];
      res[r]/=A[r][r];
    }}
    return res;
  }}

  // Swap: use real_y as first coordinate, real_x as second
  const rows = keys.map(k => [anchors[k].real_y, anchors[k].real_x, 1]);
  const xv   = keys.map(k => anchors[k].img_x);
  const yv   = keys.map(k => anchors[k].img_y);
  const xc = ne(rows, xv), yc = ne(rows, yv);
  if (!xc || !yc) return null;
  return {{ a:xc[0], b:xc[1], tx:xc[2], c:yc[0], d:yc[1], ty:yc[2] }};
}}

// applyT: pass raw UWB (rx=X, ry=Y); swap internally to match solver
function applyT(T, rx, ry) {{
  return {{ fx: T.a*ry + T.b*rx + T.tx, fy: T.c*ry + T.d*rx + T.ty }};
}}

// ── Kalman ─────────────────────────────────────────────────
function makeKalman(R, Q) {{
  let x=[0,0,0,0], P=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], init=false;
  return {{
    reset() {{ init=false; }},
    update(mx, my, dt) {{
      if (!init) {{ x=[mx,my,0,0]; init=true; return {{x:mx,y:my,vx:0,vy:0}}; }}
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
      return {{x:x[0],y:x[1],vx:x[2],vy:x[3]}};
    }}
  }};
}}

function makeMovAvg(n=5) {{
  let buf=[];
  return {{
    reset() {{ buf=[]; }},
    update(x,y) {{
      buf.push({{x,y}}); if(buf.length>n) buf.shift();
      return {{ x:buf.reduce((s,p)=>s+p.x,0)/buf.length, y:buf.reduce((s,p)=>s+p.y,0)/buf.length }};
    }}
  }};
}}

// ── Map ────────────────────────────────────────────────────
const leafMap = L.map('map', {{
  crs:L.CRS.Simple, minZoom:-3, maxZoom:4,
  zoomSnap:0.25, attributionControl:false
}});
let mapW=900, mapH=720, T=null;
function frac2ll(fx,fy) {{ return [-(fy*mapH), fx*mapW]; }}

// Tag state — dot + pulse only, no trail
const state = [0,1].map(() => ({{
  kalman: makeKalman(CFG.kalmanR, CFG.kalmanQ),
  movAvg: makeMovAvg(5),
  marker:null, pulse:null, lastT:null
}}));

let totalMsg=0, msgWindow=0;
setInterval(()=>{{
  document.getElementById('sb-rate').textContent = msgWindow.toFixed(1);
  msgWindow=0;
}}, 1000);

function addPoint(tagIdx, rx, ry) {{
  if (!T) return;
  const now = Date.now();
  const s = state[tagIdx];
  const dt = s.lastT ? (now - s.lastT) / 1000 : 0.1;
  s.lastT = now;

  let sx=rx, sy=ry, vx=0, vy=0;
  if (CFG.smoothing === 'kalman') {{
    const k = s.kalman.update(rx, ry, dt);
    sx=k.x; sy=k.y; vx=k.vx||0; vy=k.vy||0;
  }} else if (CFG.smoothing === 'moving_avg') {{
    const m = s.movAvg.update(rx, ry);
    sx=m.x; sy=m.y;
  }}

  redraw(tagIdx, rx, ry, sx, sy, vx, vy, now);
  totalMsg++; msgWindow++;
  document.getElementById('h-msgs').textContent = totalMsg;
  document.getElementById('sb-last').textContent = new Date(now).toLocaleTimeString();
}}

function redraw(tagIdx, rx, ry, sx, sy, vx, vy, now) {{
  if (!T) return;
  const s = state[tagIdx];
  const isT1 = tagIdx === 0;
  const cDot   = isT1 ? '#dc2626' : '#0369a1';
  const cGlow  = isT1 ? 'rgba(220,38,38,.2)' : 'rgba(3,105,161,.2)';
  const label  = isT1 ? CFG.tag1Label : CFG.tag2Label;
  const hpfx   = isT1 ? '1' : '2';

  const {{fx,fy}} = applyT(T, sx, sy);
  const ll = frac2ll(fx, fy);
  const speed = Math.hypot(vx, vy);

  const icon = L.divIcon({{className:'',iconSize:[26,26],iconAnchor:[13,13],
    html:`<div style="width:26px;height:26px;border-radius:50%;
      background:${{cDot}};border:3px solid #fff;
      box-shadow:0 0 0 4px ${{cGlow}},0 2px 12px ${{cGlow}}"></div>`
  }});

  if (s.marker) leafMap.removeLayer(s.marker);
  if (s.pulse)  leafMap.removeLayer(s.pulse);

  s.marker = L.marker(ll, {{icon, zIndexOffset:1000-tagIdx}})
    .bindPopup(`<b style="color:${{cDot}};font-family:'Outfit',sans-serif">📍 ${{label}}</b><br>
      <span style="color:#6b7a99;font-family:'DM Mono',monospace;font-size:10px;line-height:1.9">
      RAW &nbsp; (${{rx.toFixed(3)}} m, ${{ry.toFixed(3)}} m)<br>
      SMOOTH (${{sx.toFixed(3)}} m, ${{sy.toFixed(3)}} m)<br>
      SPEED &nbsp;${{(speed*100).toFixed(1)}} cm/s<br>
      ⏱ ${{new Date(now).toLocaleTimeString()}}</span>`)
    .addTo(leafMap);

  s.pulse = L.circle(ll, {{
    radius: Math.min(mapW,mapH)*.025,
    color:cDot, fillColor:cDot, fillOpacity:.06, weight:1.5
  }}).addTo(leafMap);

  document.getElementById(`h${{hpfx}}-rx`).textContent  = rx.toFixed(3)+' m';
  document.getElementById(`h${{hpfx}}-ry`).textContent  = ry.toFixed(3)+' m';
  document.getElementById(`h${{hpfx}}-sx`).textContent  = sx.toFixed(3)+' m';
  document.getElementById(`h${{hpfx}}-sy`).textContent  = sy.toFixed(3)+' m';
  document.getElementById(`h${{hpfx}}-spd`).textContent = (speed*100).toFixed(1)+' cm/s';
}}

// ── Init map ───────────────────────────────────────────────
const imgEl = new Image();
imgEl.onload = function() {{
  mapW = imgEl.naturalWidth  || 900;
  mapH = imgEl.naturalHeight || 720;
  const bounds = [[-mapH,0],[0,mapW]];
  L.imageOverlay(IMG_SRC, bounds, {{opacity:1}}).addTo(leafMap);
  leafMap.invalidateSize({{animate:false}});
  setTimeout(()=>leafMap.fitBounds(bounds,{{padding:[16,16]}}), 60);

  T = computeTransform(CFG.anchors);

  Object.entries(CFG.anchors).forEach(([id, a]) => {{
    const ll = frac2ll(a.img_x, a.img_y);
    const icon = L.divIcon({{className:'',iconSize:[30,30],iconAnchor:[15,15],
      html:`<div style="width:30px;height:30px;border-radius:6px;background:#eff6ff;border:2.5px solid #1a56db;
        box-shadow:0 0 0 4px rgba(26,86,219,.12),0 2px 8px rgba(26,86,219,.2);
        display:flex;align-items:center;justify-content:center;
        color:#1a56db;font-size:9px;font-weight:700;font-family:'DM Mono',monospace">
        ${{id}}</div>`
    }});
    L.marker(ll, {{icon}})
      .bindPopup(`<b style="color:#1a56db">${{a.label}}</b><br>
        <span style="color:#6b7a99;font-family:'DM Mono',monospace;font-size:10px;line-height:1.9">
        Addr: ${{a.addr}}<br>
        Image: (${{a.img_x.toFixed(2)}}, ${{a.img_y.toFixed(2)}})<br>
        Real: (${{a.real_x.toFixed(2)}} m, ${{a.real_y.toFixed(2)}} m)</span>`)
      .addTo(leafMap);
    L.circle(ll,{{radius:Math.min(mapW,mapH)*.02,color:'#1a56db',fillColor:'#1a56db',fillOpacity:.04,weight:1,dashArray:'4,6'}}).addTo(leafMap);
  }});

  const akeys = Object.keys(CFG.anchors);
  for (let i=0;i<akeys.length;i++) for (let j=i+1;j<akeys.length;j++) {{
    const ai=CFG.anchors[akeys[i]], aj=CFG.anchors[akeys[j]];
    L.polyline([frac2ll(ai.img_x,ai.img_y),frac2ll(aj.img_x,aj.img_y)],
      {{color:'#bfdbfe',weight:1,dashArray:'4,8',opacity:.8}}).addTo(leafMap);
  }}

  document.getElementById('h-t1-name').textContent = CFG.tag1Label.toUpperCase();
  document.getElementById('h-t2-name').textContent = CFG.tag2Label.toUpperCase();
  document.getElementById('leg-t1').textContent    = CFG.tag1Label+' (live)';
  document.getElementById('leg-t2').textContent    = CFG.tag2Label+' (live)';
  document.getElementById('sb-broker').textContent = CFG.broker+':'+CFG.port;
  document.getElementById('sb-t1').textContent     = CFG.topic;

  if (CFG.tag2Enabled && CFG.topic2) {{
    document.getElementById('hud-t2').style.display      = 'block';
    document.getElementById('leg-t2-wrap').style.display = 'block';
    document.getElementById('sb-t2').textContent         = CFG.topic2;
  }} else {{
    document.getElementById('sb-t2-wrap').style.display  = 'none';
  }}

  startMQTT();
}};
imgEl.onerror = ()=>console.error('Floor plan failed');
imgEl.src = IMG_SRC;

// ── MQTT ───────────────────────────────────────────────────
function startMQTT() {{
  const proto = (location.protocol==='https:'||CFG.port===8084) ? 'wss' : 'ws';
  const url = `${{proto}}://${{CFG.broker}}:${{CFG.port}}/mqtt`;
  const client = mqtt.connect(url, {{
    clientId:'uwb_'+Math.random().toString(16).slice(2,10),
    clean:true, reconnectPeriod:4000, connectTimeout:10000, keepalive:30
  }});

  const setDot = cls =>
    document.getElementById('h-status').innerHTML = `<span class="dot ${{cls}}"></span>`;

  client.on('connect',    ()=>{{ setDot('dot-g'); client.subscribe(CFG.topic); if(CFG.tag2Enabled&&CFG.topic2) client.subscribe(CFG.topic2); }});
  client.on('reconnect',  ()=>setDot('dot-y'));
  client.on('disconnect', ()=>setDot('dot-r'));
  client.on('offline',    ()=>setDot('dot-r'));
  client.on('error',      ()=>setDot('dot-r'));

  client.on('message', (topic, payload) => {{
    try {{
      const d = JSON.parse(payload.toString());
      const rx = parseFloat(d.x), ry = parseFloat(d.y);
      if (isNaN(rx)||isNaN(ry)) return;
      if (topic===CFG.topic)       addPoint(0,rx,ry);
      else if (topic===CFG.topic2) addPoint(1,rx,ry);
    }} catch(e) {{ console.warn('Parse:',e); }}
  }});

  window.clearDots = ()=>{{
    state.forEach(s=>{{
      s.kalman.reset(); s.movAvg.reset(); s.lastT=null;
      [s.marker,s.pulse].forEach(l=>{{ if(l) leafMap.removeLayer(l); }});
      s.marker=null; s.pulse=null;
    }});
  }};
}}
</script>
</body>
</html>"""


from streamlit.components.v1 import html as st_html
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

st.markdown("""
<div style="margin-top:10px;padding:10px 0;border-top:1px solid #dde3ec;
  color:#aab8cc;font-size:9px;letter-spacing:2px;text-align:center;font-family:monospace">
  DW1000 UWB · ESP32 · MQTT WebSocket · Leaflet CRS.Simple · Kalman Filter ·
  4-Anchor Trilateration · Axis-swap corrected · Calibration persisted in localStorage
</div>
""", unsafe_allow_html=True)
