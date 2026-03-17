import streamlit as st
import streamlit.components.v1 as components
import base64
import json
import hashlib
import os
import math
import time
from PIL import Image

# ─── Paths ────────────────────────────────────────────────────────────────────
USERS_FILE   = "users.json"
PINS_FILE    = "pins.json"
ANCHORS_FILE = "anchors.json"
MAPS_DIR     = "user_maps"
RSSI_LOG     = "rssi_log.json"
os.makedirs(MAPS_DIR, exist_ok=True)

# ─── JSON helpers ─────────────────────────────────────────────────────────────
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# ─── Auth ─────────────────────────────────────────────────────────────────────
def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def register_user(u, p):
    users = load_json(USERS_FILE)
    if u in users:
        return False, "Username already exists."
    users[u] = hash_pw(p)
    save_json(USERS_FILE, users)
    return True, "Account created! You can now log in."

def login_user(u, p):
    users = load_json(USERS_FILE)
    if u not in users:
        return False, "Username not found."
    if users[u] != hash_pw(p):
        return False, "Incorrect password."
    return True, "ok"

# ─── Pins ─────────────────────────────────────────────────────────────────────
def get_pins(username):
    data = load_json(PINS_FILE)
    pins = data.get(username, {})
    if not isinstance(pins, dict):
        pins = {}
    return pins

def save_pins(username, pins):
    if not isinstance(pins, dict):
        pins = {}
    all_pins = load_json(PINS_FILE)
    all_pins[username] = pins
    save_json(PINS_FILE, all_pins)

def remove_pin(username, slot):
    pins = get_pins(username)
    pins.pop(str(slot), None)
    save_pins(username, pins)

def set_label(username, slot, label):
    pins = get_pins(username)
    if str(slot) in pins:
        pins[str(slot)]["label"] = label
        save_pins(username, pins)

def set_pin_info(username, slot, field, value):
    pins = get_pins(username)
    if str(slot) in pins:
        pins[str(slot)][field] = value
        save_pins(username, pins)

# ─── Anchors (RSSI) ───────────────────────────────────────────────────────────
def get_anchors(username):
    data = load_json(ANCHORS_FILE)
    return data.get(username, {})

def save_anchors(username, anchors):
    all_data = load_json(ANCHORS_FILE)
    all_data[username] = anchors
    save_json(ANCHORS_FILE, all_data)

# ─── RSSI → Distance (Log-distance path loss model) ──────────────────────────
def rssi_to_distance(rssi, rssi_ref=-40, n=2.0):
    """Convert RSSI (dBm) to distance (meters). rssi_ref = RSSI at 1m, n = path loss exponent."""
    return 10 ** ((rssi_ref - rssi) / (10 * n))

# ─── Weighted Centroid Trilateration ─────────────────────────────────────────
def trilaterate(anchors_with_rssi, scale_px_per_m=50):
    """
    anchors_with_rssi: list of dicts {x, y, rssi, rssi_ref, path_loss_n}
    Returns (est_x, est_y) in image pixel coordinates.
    """
    if len(anchors_with_rssi) < 2:
        return None, None

    total_weight = 0.0
    wx = 0.0
    wy = 0.0

    for a in anchors_with_rssi:
        dist_m = rssi_to_distance(a["rssi"], a.get("rssi_ref", -40), a.get("path_loss_n", 2.0))
        dist_px = dist_m * scale_px_per_m
        weight = 1.0 / max(dist_px, 1.0)  # inverse distance weighting
        wx += a["x"] * weight
        wy += a["y"] * weight
        total_weight += weight

    if total_weight == 0:
        return None, None
    return wx / total_weight, wy / total_weight

# ─── Floor plan ───────────────────────────────────────────────────────────────
def save_floor_plan(username, b64, w, h, mime):
    with open(os.path.join(MAPS_DIR, f"{username}_img.b64"), "w") as f:
        f.write(b64)
    save_json(os.path.join(MAPS_DIR, f"{username}_meta.json"),
              {"w": w, "h": h, "mime": mime})

def load_floor_plan(username):
    mp = os.path.join(MAPS_DIR, f"{username}_meta.json")
    ip = os.path.join(MAPS_DIR, f"{username}_img.b64")
    if os.path.exists(mp) and os.path.exists(ip):
        meta = load_json(mp)
        with open(ip) as f:
            b64 = f.read()
        return b64, meta["w"], meta["h"], meta["mime"]
    return None, None, None, None

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Floor Plan + RSSI Positioning", layout="wide", page_icon="📡")

for k, v in {
    "logged_in": False, "username": "",
    "edit_mode": False, "anchor_mode": False,
    "active_tab": "map"
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:
    st.title("📡 Floor Plan + RSSI Indoor Positioning")
    tab_l, tab_r = st.tabs(["🔑 Login", "📝 Create Account"])
    with tab_l:
        u = st.text_input("Username", key="li_u")
        p = st.text_input("Password", type="password", key="li_p")
        if st.button("Login", use_container_width=True):
            if not u or not p:
                st.warning("Fill in both fields.")
            else:
                ok, msg = login_user(u, p)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.username  = u
                    st.rerun()
                else:
                    st.error(msg)
    with tab_r:
        u2  = st.text_input("Choose username",  key="ru")
        p2  = st.text_input("Choose password",  type="password", key="rp")
        p2c = st.text_input("Confirm password", type="password", key="rpc")
        if st.button("Create Account", use_container_width=True):
            if not u2 or not p2 or not p2c:
                st.warning("Fill in all fields.")
            elif p2 != p2c:
                st.error("Passwords do not match.")
            elif len(p2) < 4:
                st.error("Password must be at least 4 characters.")
            else:
                ok, msg = register_user(u2, p2)
                st.success(msg) if ok else st.error(msg)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-SAVE pins via URL query param
# ══════════════════════════════════════════════════════════════════════════════
_qp = st.query_params
if "pins" in _qp and st.session_state.logged_in:
    try:
        raw = _qp["pins"]
        new_pins = json.loads(raw)
        if isinstance(new_pins, dict) and new_pins:
            _uname = st.session_state.username
            existing = get_pins(_uname)
            for slot, data in new_pins.items():
                if slot in existing:
                    for field in ["label", "description", "status", "contact", "notes"]:
                        saved_val = existing[slot].get(field, "")
                        if saved_val and saved_val not in [f"Pin {slot}", ""]:
                            data[field] = saved_val
            save_pins(_uname, new_pins)
            st.session_state["_last_saved_count"] = len(new_pins)
    except Exception:
        pass
    st.query_params.clear()
    st.rerun()

# AUTO-SAVE anchors via URL query param
if "anchors" in _qp and st.session_state.logged_in:
    try:
        raw = _qp["anchors"]
        new_anchors = json.loads(raw)
        if isinstance(new_anchors, dict):
            _uname = st.session_state.username
            existing = get_anchors(_uname)
            for aid, adata in new_anchors.items():
                if aid in existing:
                    # Preserve config fields
                    for field in ["name", "mac", "rssi_ref", "path_loss_n", "desc"]:
                        if existing[aid].get(field):
                            adata.setdefault(field, existing[aid][field])
            save_anchors(_uname, new_anchors)
            st.session_state["_last_saved_anchors"] = len(new_anchors)
    except Exception:
        pass
    st.query_params.clear()
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
username    = st.session_state.username
edit_mode   = st.session_state.edit_mode
anchor_mode = st.session_state.anchor_mode

# ── Header ────────────────────────────────────────────────────────────────────
h1, h2, h3, h4, h5 = st.columns([4, 1.3, 1.3, 1.1, 0.9])
with h1:
    st.title("📡 Floor Plan + RSSI Positioning")
    st.caption(f"👋 **{username}**")
with h2:
    st.write(""); st.write("")
    if edit_mode:
        if st.button("✅ Done Editing", use_container_width=True, type="primary"):
            st.session_state.edit_mode = False
            st.rerun()
    else:
        if st.button("✏️ Edit Pins", use_container_width=True):
            st.session_state.edit_mode   = True
            st.session_state.anchor_mode = False
            st.rerun()
with h3:
    st.write(""); st.write("")
    if anchor_mode:
        if st.button("✅ Done Anchors", use_container_width=True, type="primary"):
            st.session_state.anchor_mode = False
            st.rerun()
    else:
        if st.button("📡 Edit Anchors", use_container_width=True):
            st.session_state.anchor_mode = True
            st.session_state.edit_mode   = False
            st.rerun()
with h4:
    st.write(""); st.write("")
    if st.button("🔌 ESP32 API", use_container_width=True):
        st.session_state.active_tab = "esp32"
        st.rerun()
with h5:
    st.write(""); st.write("")
    if st.button("🚪 Logout", use_container_width=True):
        for k in ["logged_in","username","edit_mode","anchor_mode"]:
            st.session_state[k] = False if k != "username" else ""
        st.rerun()

if st.session_state.get("_last_saved_count"):
    n = st.session_state.pop("_last_saved_count")
    st.toast(f"✅ {n} pin(s) auto-saved!", icon="📌")
if st.session_state.get("_last_saved_anchors"):
    n = st.session_state.pop("_last_saved_anchors")
    st.toast(f"✅ {n} anchor(s) saved!", icon="📡")

# ── Mode banner ───────────────────────────────────────────────────────────────
if edit_mode:
    st.info("✏️ **Pin Edit Mode** — Press 1–9 to select slot, click map to place.")
elif anchor_mode:
    st.warning("📡 **Anchor Edit Mode** — Press A–F to select anchor, click map to place. Configure anchors in the sidebar.")
else:
    st.success("👁️ **View Mode** — Click any pin or anchor to see details. Live RSSI position shown if ESP32 is sending data.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
#  ESP32 API TAB
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.get("active_tab") == "esp32":
    if st.button("← Back to Map"):
        st.session_state.active_tab = "map"
        st.rerun()

    anchors = get_anchors(username)
    anchor_list = [{"id": k, **v} for k, v in anchors.items()]

    st.header("🔌 ESP32 Integration Guide")
    st.markdown("""
    Your ESP32 scans for BLE/WiFi signals from the anchors and sends RSSI values to this server.
    Below is everything you need to integrate.
    """)

    # ── Anchor MAC table ──────────────────────────────────────────────────────
    st.subheader("📋 Anchor Configuration")
    if anchors:
        cols = st.columns([0.5, 2, 2.5, 1.2, 1.2, 2])
        cols[0].markdown("**ID**"); cols[1].markdown("**Name**")
        cols[2].markdown("**MAC / SSID**"); cols[3].markdown("**RSSI@1m**")
        cols[4].markdown("**Path Loss n**"); cols[5].markdown("**Position (px)**")
        for aid, a in anchors.items():
            c = st.columns([0.5, 2, 2.5, 1.2, 1.2, 2])
            c[0].code(aid)
            c[1].write(a.get("name", f"Anchor {aid}"))
            c[2].code(a.get("mac", "—"))
            c[3].write(f'{a.get("rssi_ref", -40)} dBm')
            c[4].write(a.get("path_loss_n", 2.0))
            c[5].write(f'x:{a.get("x","?")} y:{a.get("y","?")}')
    else:
        st.info("No anchors placed yet. Go to 📡 Edit Anchors mode to place them on the map.")

    st.divider()

    # ── Arduino code ──────────────────────────────────────────────────────────
    st.subheader("📟 ESP32 Arduino Code")

    server_url = "http://YOUR_SERVER_IP:8501"  # placeholder

    # Build anchor MAC list for code
    mac_lines = ""
    for aid, a in anchors.items():
        mac = a.get("mac", f"AA:BB:CC:DD:EE:0{aid}")
        name = a.get("name", f"Anchor_{aid}")
        mac_lines += f'  {{"{aid}", "{mac}", "{name}"}},\n'
    if not mac_lines:
        mac_lines = '  {"A", "AA:BB:CC:DD:EE:01", "Anchor_A"},\n  {"B", "AA:BB:CC:DD:EE:02", "Anchor_B"},\n'

    arduino_code = f'''#include <Arduino.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <BLEDevice.h>
#include <BLEScan.h>

// ── Config ─────────────────────────────────────────────────────────────────
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL    = "{server_url}";
const char* USERNAME      = "{username}";
const int   SCAN_INTERVAL = 2000;  // ms between scans

// ── Anchor table ───────────────────────────────────────────────────────────
struct Anchor {{
  const char* id;
  const char* mac;   // BLE MAC address (lowercase)
  const char* name;
}};

Anchor anchors[] = {{
{mac_lines}}};
const int NUM_ANCHORS = sizeof(anchors) / sizeof(anchors[0]);

// ── BLE scan ───────────────────────────────────────────────────────────────
BLEScan* pBLEScan;
int rssiValues[10];  // adjust size if >10 anchors

class MyAdvertisedDeviceCallbacks : public BLEAdvertisedDeviceCallbacks {{
  void onResult(BLEAdvertisedDevice dev) {{
    String mac = dev.getAddress().toString().c_str();
    mac.toLowerCase();
    for (int i = 0; i < NUM_ANCHORS; i++) {{
      if (mac == String(anchors[i].mac)) {{
        rssiValues[i] = dev.getRSSI();
        Serial.printf("[BLE] %s (%s): %d dBm\\n", anchors[i].name, anchors[i].mac, rssiValues[i]);
      }}
    }}
  }}
}};

void setup() {{
  Serial.begin(115200);
  
  // ── WiFi ──────────────────────────────────────────────────────────────────
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {{
    delay(500); Serial.print(".");
  }}
  Serial.println("\\n✅ WiFi connected: " + WiFi.localIP().toString());

  // ── BLE ───────────────────────────────────────────────────────────────────
  BLEDevice::init("");
  pBLEScan = BLEDevice::getScan();
  pBLEScan->setAdvertisedDeviceCallbacks(new MyAdvertisedDeviceCallbacks());
  pBLEScan->setActiveScan(true);
  pBLEScan->setInterval(100);
  pBLEScan->setWindow(99);

  // Init RSSI array to 0 (no signal)
  for (int i = 0; i < NUM_ANCHORS; i++) rssiValues[i] = 0;
}}

void sendRSSI() {{
  if (WiFi.status() != WL_CONNECTED) return;

  // Build JSON payload
  StaticJsonDocument<512> doc;
  doc["username"] = USERNAME;
  JsonObject readings = doc.createNestedObject("readings");
  for (int i = 0; i < NUM_ANCHORS; i++) {{
    if (rssiValues[i] != 0) {{
      readings[anchors[i].id] = rssiValues[i];
    }}
  }}

  String payload;
  serializeJson(doc, payload);
  Serial.println("Sending: " + payload);

  HTTPClient http;
  http.begin(SERVER_URL + String("/?rssi_data=") + payload);
  // Alternative: POST endpoint
  // http.begin(SERVER_URL + "/rssi");
  // http.addHeader("Content-Type", "application/json");
  // int code = http.POST(payload);

  int code = http.GET();
  Serial.printf("HTTP %d\\n", code);
  http.end();
}}

void loop() {{
  // BLE scan for 1 second
  BLEScanResults results = pBLEScan->start(1, false);
  pBLEScan->clearResults();
  
  sendRSSI();
  delay(SCAN_INTERVAL);
}}'''

    st.code(arduino_code, language="cpp")

    st.subheader("📦 Required Arduino Libraries")
    st.markdown("""
    Install via Arduino Library Manager:
    - `ArduinoJson` by Benoit Blanchon (v6.x)
    - ESP32 BLE Arduino (included with ESP32 board package)
    - HTTPClient (included with ESP32 board package)
    
    Board: **ESP32 Dev Module** · Board Manager URL:  
    `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
    """)

    st.subheader("🔄 Data Flow")
    st.markdown("""
    ```
    ESP32 (BLE scan)
        │  RSSI values per anchor MAC
        ▼
    HTTP GET /?rssi_data={...}
        │
        ▼
    Streamlit server (this app)
        │  Trilateration (weighted centroid)
        ▼
    Estimated position (x, y) on floor plan
    ```
    """)

    st.divider()
    st.subheader("⚙️ Calibration Tips")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **RSSI @ 1m (rssi_ref)**  
        Place your ESP32 exactly 1 meter from each anchor.  
        Note the average RSSI — use that as `rssi_ref` for that anchor.
        
        Typical values: `-40` to `-55` dBm for BLE at 1m.
        """)
    with col2:
        st.markdown("""
        **Path Loss Exponent (n)**  
        - `n = 2.0` — free space (open area)  
        - `n = 2.5–3.0` — indoor with few walls  
        - `n = 3.0–4.0` — many walls / obstacles  
        
        Start with `2.0` and tune based on accuracy.
        """)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  FLOOR PLAN UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
if edit_mode or anchor_mode:
    uploaded = st.file_uploader("Upload / replace floor plan (PNG or JPG)", type=["png","jpg","jpeg"])
    if uploaded:
        raw  = uploaded.read()
        img  = Image.open(uploaded)
        w, h_img = img.size
        b64  = base64.b64encode(raw).decode()
        ext  = uploaded.name.split(".")[-1].lower()
        mime = "image/jpeg" if ext in ["jpg","jpeg"] else "image/png"
        save_floor_plan(username, b64, w, h_img, mime)
        st.success("Floor plan saved!")
        st.rerun()

b64, img_w, img_h, mime = load_floor_plan(username)
if not b64:
    st.info("Switch to ✏️ Edit Pins or 📡 Edit Anchors mode and upload a floor plan image to get started.")
    st.stop()

img_src = f"data:{mime};base64,{b64}"

# ══════════════════════════════════════════════════════════════════════════════
#  LIVE RSSI from ESP32 (URL query param)
# ══════════════════════════════════════════════════════════════════════════════
rssi_position = None
rssi_data_raw = None

if "rssi_data" in _qp:
    try:
        rssi_data_raw = json.loads(_qp["rssi_data"])
        anchors_all = get_anchors(username)
        anchors_with_rssi = []
        for aid, rssi_val in rssi_data_raw.items():
            if aid in anchors_all:
                a = anchors_all[aid]
                if "x" in a and "y" in a:
                    anchors_with_rssi.append({
                        "x": a["x"], "y": a["y"],
                        "rssi": rssi_val,
                        "rssi_ref": a.get("rssi_ref", -40),
                        "path_loss_n": a.get("path_loss_n", 2.0)
                    })
        if len(anchors_with_rssi) >= 2:
            ex, ey = trilaterate(anchors_with_rssi)
            if ex is not None:
                rssi_position = {"x": round(ex), "y": round(ey)}
                # Log it
                log = load_json(RSSI_LOG) if os.path.exists(RSSI_LOG) else []
                if isinstance(log, list):
                    log.append({"ts": time.time(), "username": username,
                                "rssi": rssi_data_raw, "pos": rssi_position})
                    log = log[-500:]  # keep last 500
                    save_json(RSSI_LOG, log)
    except Exception as e:
        st.warning(f"RSSI parse error: {e}")
    st.query_params.clear()
    st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  BUILD MAP
# ══════════════════════════════════════════════════════════════════════════════
pins         = get_pins(username)
anchors      = get_anchors(username)
pins_json    = json.dumps(pins)
anchors_json = json.dumps(anchors)
pos_json     = json.dumps(rssi_position) if rssi_position else "null"

edit_js      = "true" if edit_mode   else "false"
anchor_js    = "true" if anchor_mode else "false"

COLORS_HEX = {
    "1":"#e74c3c","2":"#e67e22","3":"#f1c40f",
    "4":"#2ecc71","5":"#1abc9c","6":"#3498db",
    "7":"#9b59b6","8":"#e91e8c","9":"#795548"
}
ANCHOR_COLORS = {
    "A":"#00e5ff","B":"#00bcd4","C":"#26c6da",
    "D":"#4dd0e1","E":"#80deea","F":"#b2ebf2"
}

map_col, side_col = st.columns([3, 1])

with map_col:
    html_code = f"""<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:sans-serif;background:#0d0d1a;}}
    #toolbar{{display:flex;gap:6px;padding:8px 10px;background:#0d1117;align-items:center;flex-wrap:wrap;border-bottom:2px solid #1e2d3d;}}
    #toolbar .tb-label{{color:#4a7fa5;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.08em;margin-right:2px;}}
    .slot-btn{{width:34px;height:34px;border-radius:8px;border:1.5px solid #2a3a4a;background:#141c26;color:#7a9bb5;font-size:14px;font-weight:bold;cursor:pointer;position:relative;transition:all .15s;display:flex;align-items:center;justify-content:center;}}
    .slot-btn:hover{{border-color:#4a9eff;color:#fff;}}
    .slot-btn.active{{border-color:#f0a030;background:#1e2800;color:#f0c060;box-shadow:0 0 10px #f0a03066;}}
    .slot-btn.placed:not(.active)::after{{content:'';position:absolute;bottom:2px;right:2px;width:7px;height:7px;border-radius:50%;background:#2ecc71;border:1px solid #0d1117;}}
    .anchor-btn{{border-radius:50%;border:1.5px solid #1a3040;background:#0a1a24;color:#00bcd4;}}
    .anchor-btn.active{{border-color:#00e5ff;background:#001820;color:#00ffff;box-shadow:0 0 10px #00e5ff55;}}
    .anchor-btn.placed:not(.active)::after{{background:#00e5ff;}}
    .tb-sep{{width:1px;height:28px;background:#1e2d3d;margin:0 4px;flex-shrink:0;}}
    #map-wrap{{position:relative;}}
    #map{{width:100%;height:500px;background:#111;}}
    #status{{background:#080e18;color:#5a7a9a;padding:6px 12px;font-size:11px;border-top:1px solid #1a2a3a;min-height:28px;display:flex;align-items:center;gap:8px;font-family:monospace;}}
    #spinner{{display:none;width:10px;height:10px;border:1.5px solid #1e3a5a;border-top-color:#00bcd4;border-radius:50%;animation:spin .6s linear infinite;flex-shrink:0;}}
    @keyframes spin{{to{{transform:rotate(360deg)}}}}

    /* ── RSSI Live Badge ── */
    #rssi-badge{{
      position:absolute;top:10px;right:10px;
      background:rgba(0,20,30,.9);border:1px solid #00bcd4;
      border-radius:8px;padding:6px 10px;z-index:1000;
      color:#00e5ff;font-size:11px;font-family:monospace;
      display:none;
    }}
    #rssi-badge.visible{{display:block;}}

    /* ── Info Panel ── */
    #info-panel{{
      position:absolute;bottom:36px;left:10px;width:270px;
      background:linear-gradient(135deg,#0d1117 0%,#0a1624 100%);
      border:1px solid #1e3050;border-radius:12px;
      box-shadow:0 8px 32px rgba(0,0,0,.7),0 0 0 1px rgba(0,180,255,.06);
      z-index:1000;overflow:hidden;
      transform:translateY(16px) scale(0.96);opacity:0;
      transition:all .22s cubic-bezier(.34,1.56,.64,1);pointer-events:none;
    }}
    #info-panel.visible{{transform:translateY(0) scale(1);opacity:1;pointer-events:all;}}
    #info-header{{display:flex;align-items:center;gap:10px;padding:12px 14px 9px;border-bottom:1px solid rgba(255,255,255,.05);}}
    #info-dot{{width:34px;height:34px;min-width:34px;border-radius:50% 50% 50% 0;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:14px;color:#fff;border:2px solid rgba(255,255,255,.2);box-shadow:0 2px 8px rgba(0,0,0,.5);transform:rotate(-45deg);}}
    #info-dot span{{transform:rotate(45deg);display:block;}}
    #info-title{{font-size:14px;font-weight:700;color:#e0eaf5;line-height:1.2;}}
    #info-slot-label{{font-size:10px;color:#3a5a7a;margin-top:2px;}}
    #info-close{{margin-left:auto;background:none;border:none;color:#2a4a6a;font-size:17px;cursor:pointer;width:26px;height:26px;display:flex;align-items:center;justify-content:center;border-radius:50%;transition:background .15s,color .15s;flex-shrink:0;}}
    #info-close:hover{{background:rgba(255,255,255,.06);color:#88aacc;}}
    #info-body{{padding:10px 14px 14px;display:flex;flex-direction:column;gap:8px;}}
    .info-row{{display:flex;flex-direction:column;gap:2px;}}
    .info-row-label{{font-size:9.5px;font-weight:600;color:#2a4a6a;text-transform:uppercase;letter-spacing:.08em;}}
    .info-row-value{{font-size:12px;color:#a0b8d0;line-height:1.4;}}
    .info-row-value.empty{{color:#2a4a6a;font-style:italic;}}
    .info-badge{{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:600;border:1px solid rgba(255,255,255,.08);}}
    .info-coords{{font-family:monospace;font-size:10px;color:#2a4a6a;padding-top:5px;border-top:1px solid rgba(255,255,255,.04);}}
    .status-active{{background:rgba(46,204,113,.12);color:#2ecc71;border-color:rgba(46,204,113,.25);}}
    .status-inactive{{background:rgba(231,76,60,.1);color:#e74c3c;border-color:rgba(231,76,60,.2);}}
    .status-maintenance{{background:rgba(241,196,15,.1);color:#f1c40f;border-color:rgba(241,196,15,.2);}}
    .status-default{{background:rgba(60,90,120,.15);color:#6699bb;border-color:rgba(60,90,120,.25);}}
  </style>
</head>
<body>
<div id="toolbar">
  <span class="tb-label" id="tb-pin-label" style="display:{'' if not anchor_js else 'none'}">Pin:</span>
  <span id="pin-btns"></span>
  <div class="tb-sep"></div>
  <span class="tb-label">Anchor:</span>
  <span id="anchor-btns"></span>
</div>
<div id="map-wrap">
  <div id="map"></div>
  <div id="rssi-badge" class="{'visible' if rssi_position else ''}">
    📡 RSSI Position: x={rssi_position['x'] if rssi_position else '–'} y={rssi_position['y'] if rssi_position else '–'}
  </div>
  <div id="info-panel">
    <div id="info-header">
      <div id="info-dot"><span id="info-dot-num">1</span></div>
      <div><div id="info-title">Pin</div><div id="info-slot-label">Slot #1</div></div>
      <button id="info-close">✕</button>
    </div>
    <div id="info-body">
      <div class="info-row" id="row-description" style="display:none">
        <div class="info-row-label">📝 Description</div>
        <div class="info-row-value" id="info-description"></div>
      </div>
      <div class="info-row" id="row-status" style="display:none">
        <div class="info-row-label">Status</div>
        <div class="info-row-value" id="info-status"></div>
      </div>
      <div class="info-row" id="row-contact" style="display:none">
        <div class="info-row-label">👤 Contact</div>
        <div class="info-row-value" id="info-contact"></div>
      </div>
      <div class="info-row" id="row-notes" style="display:none">
        <div class="info-row-label">🗒️ Notes</div>
        <div class="info-row-value" id="info-notes"></div>
      </div>
      <div class="info-row" id="row-mac" style="display:none">
        <div class="info-row-label">📶 MAC / SSID</div>
        <div class="info-row-value" id="info-mac"></div>
      </div>
      <div class="info-row" id="row-rssi" style="display:none">
        <div class="info-row-label">📡 Live RSSI</div>
        <div class="info-row-value" id="info-rssi"></div>
      </div>
      <div id="row-empty" class="info-row" style="display:none">
        <div class="info-row-value empty">No details added yet.</div>
      </div>
      <div class="info-coords" id="info-coords"></div>
    </div>
  </div>
</div>
<div id="status"><div id="spinner"></div><span id="status-text">Loading…</span></div>
<script>
  var imgW={img_w}, imgH={img_h};
  var editMode={edit_js}, anchorMode={anchor_js};
  var pins={pins_json};
  var anchors={anchors_json};
  var rssiPos={pos_json};
  var activeSlot=null, activeAnchor=null;
  var markers={{}}, anchorMarkers={{}};

  var COLORS={{1:'#e74c3c',2:'#e67e22',3:'#f1c40f',4:'#2ecc71',5:'#1abc9c',6:'#3498db',7:'#9b59b6',8:'#e91e8c',9:'#795548'}};
  var ANCHOR_COLORS={{A:'#00e5ff',B:'#00bcd4',C:'#26c6da',D:'#4dd0e1',E:'#80deea',F:'#b2ebf2'}};
  var STATUS_CLASSES={{active:'status-active',inactive:'status-inactive',maintenance:'status-maintenance'}};
  var ANCHOR_IDS=['A','B','C','D','E','F'];

  var map=L.map('map',{{crs:L.CRS.Simple,minZoom:-2,maxZoom:4,zoomSnap:0.25}});
  L.imageOverlay("{img_src}",[[0,0],[imgH,imgW]]).addTo(map);
  map.fitBounds([[0,0],[imgH,imgW]]);

  // ── Build toolbar ──────────────────────────────────────────────────────────
  var pinBtnsEl=document.getElementById('pin-btns');
  for(var s=1;s<=9;s++){{
    (function(slot){{
      var btn=document.createElement('button');
      btn.className='slot-btn'; btn.id='btn-'+slot; btn.textContent=slot;
      if(!editMode) btn.disabled=true;
      btn.addEventListener('click',function(){{if(editMode) selectSlot(slot);}});
      pinBtnsEl.appendChild(btn);
    }})(s);
  }}
  var anchorBtnsEl=document.getElementById('anchor-btns');
  ANCHOR_IDS.forEach(function(id){{
    var btn=document.createElement('button');
    btn.className='slot-btn anchor-btn'; btn.id='abtn-'+id; btn.textContent=id;
    if(!anchorMode) btn.disabled=true;
    btn.addEventListener('click',function(){{if(anchorMode) selectAnchor(id);}});
    anchorBtnsEl.appendChild(btn);
  }});

  // ── Info panel ────────────────────────────────────────────────────────────
  var infoPanel=document.getElementById('info-panel');
  document.getElementById('info-close').addEventListener('click',closeInfoPanel);

  function openInfoPanel(type, id){{
    var p = type==='pin' ? pins[id] : anchors[id];
    if(!p) return;
    var c = type==='pin' ? (COLORS[id]||'#888') : (ANCHOR_COLORS[id]||'#00bcd4');

    document.getElementById('info-dot').style.background=c;
    document.getElementById('info-dot-num').textContent=id;
    document.getElementById('info-title').textContent = type==='anchor'
      ? (p.name||('Anchor '+id)) : (p.label||('Pin '+id));
    document.getElementById('info-slot-label').textContent = type==='anchor'
      ? ('📡 Anchor '+id) : ('Slot #'+id);

    var hasAny=false;
    function showRow(rowId,valId,val,renderer){{
      var row=document.getElementById(rowId);
      var el=document.getElementById(valId);
      if(val&&String(val).trim()){{
        row.style.display='flex'; el.innerHTML=renderer?renderer(val):escHtml(val); hasAny=true;
      }} else {{ row.style.display='none'; }}
    }}

    if(type==='pin'){{
      showRow('row-description','info-description',p.description,null);
      showRow('row-status','info-status',p.status,function(v){{
        var lv=v.toLowerCase(); var cls=STATUS_CLASSES[lv]||'status-default';
        var dot=lv==='active'?'🟢':lv==='inactive'?'🔴':lv==='maintenance'?'🟡':'⚪';
        return '<span class="info-badge '+cls+'">'+dot+' '+escHtml(v)+'</span>';
      }});
      showRow('row-contact','info-contact',p.contact,null);
      showRow('row-notes','info-notes',p.notes,function(v){{return escHtml(v).replace(/\\n/g,'<br>');}});
      document.getElementById('row-mac').style.display='none';
      document.getElementById('row-rssi').style.display='none';
    }} else {{
      document.getElementById('row-description').style.display='none';
      document.getElementById('row-status').style.display='none';
      document.getElementById('row-contact').style.display='none';
      document.getElementById('row-notes').style.display='none';
      showRow('row-mac','info-mac',p.mac,function(v){{return '<code style="color:#00e5ff">'+escHtml(v)+'</code>';}});
      if(p.last_rssi){{
        document.getElementById('row-rssi').style.display='flex';
        document.getElementById('info-rssi').innerHTML='<code style="color:#4caf50">'+p.last_rssi+' dBm</code>';
        hasAny=true;
      }} else {{ document.getElementById('row-rssi').style.display='none'; }}
    }}

    document.getElementById('row-empty').style.display=hasAny?'none':'flex';
    document.getElementById('info-coords').textContent='x: '+p.x+'  |  y: '+p.y;
    infoPanel.classList.add('visible');
  }}

  function closeInfoPanel(){{ infoPanel.classList.remove('visible'); }}
  function escHtml(s){{ return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }}

  // ── Map click ─────────────────────────────────────────────────────────────
  map.on('click',function(e){{
    if(!editMode && !anchorMode){{ closeInfoPanel(); return; }}
    var lat=e.latlng.lat, lng=e.latlng.lng;
    if(lat<0||lat>imgH||lng<0||lng>imgW) return;
    var x=Math.round(lng), y=Math.round(lat);

    if(editMode && activeSlot!==null){{
      var existing=pins[activeSlot]||{{}};
      pins[activeSlot]=Object.assign({{}},existing,{{x:x,y:y,label:existing.label||('Pin '+activeSlot)}});
      placePin(activeSlot,x,y,pins[activeSlot],true);
      document.getElementById('btn-'+activeSlot).classList.add('placed');
      setStatus('📌 Pin '+activeSlot+' placed — saving…',true);
      autosavePins();
    }}

    if(anchorMode && activeAnchor!==null){{
      var ea=anchors[activeAnchor]||{{}};
      anchors[activeAnchor]=Object.assign({{}},ea,{{x:x,y:y,name:ea.name||('Anchor '+activeAnchor)}});
      placeAnchor(activeAnchor,x,y,anchors[activeAnchor],true);
      document.getElementById('abtn-'+activeAnchor).classList.add('placed');
      setStatus('📡 Anchor '+activeAnchor+' placed — saving…',true);
      autosaveAnchors();
    }}
  }});

  // ── Autosave ──────────────────────────────────────────────────────────────
  function autosavePins(){{
    var encoded=encodeURIComponent(JSON.stringify(pins));
    try{{
      window.parent.history.replaceState(null,'','?pins='+encoded);
      window.parent.dispatchEvent(new PopStateEvent('popstate',{{state:null}}));
    }} catch(e){{
      window.parent.postMessage({{type:'FP_PINS',data:JSON.stringify(pins)}},'*');
      setStatus('📌 Placed · Use manual save if needed',false);
    }}
  }}
  function autosaveAnchors(){{
    var encoded=encodeURIComponent(JSON.stringify(anchors));
    try{{
      window.parent.history.replaceState(null,'','?anchors='+encoded);
      window.parent.dispatchEvent(new PopStateEvent('popstate',{{state:null}}));
    }} catch(e){{
      window.parent.postMessage({{type:'FP_ANCHORS',data:JSON.stringify(anchors)}},'*');
      setStatus('📡 Placed · Use manual save if needed',false);
    }}
  }}

  // ── Marker makers ─────────────────────────────────────────────────────────
  function makePinIcon(slot,pulse){{
    var c=COLORS[slot]||'#888';
    var glow=pulse?'box-shadow:0 0 0 4px '+c+'44,0 0 14px '+c+'66;':'box-shadow:0 2px 6px rgba(0,0,0,.6);';
    return L.divIcon({{className:'',html:'<div style="background:'+c+';color:#fff;border-radius:50% 50% 50% 0;width:30px;height:30px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:12px;border:2px solid #fff;'+glow+'transform:rotate(-45deg);cursor:pointer;"><span style="transform:rotate(45deg)">'+slot+'</span></div>',iconSize:[30,30],iconAnchor:[15,30]}});
  }}

  function makeAnchorIcon(id,pulse){{
    var c=ANCHOR_COLORS[id]||'#00bcd4';
    var glow=pulse?'box-shadow:0 0 0 5px '+c+'33,0 0 18px '+c+'55;':'box-shadow:0 2px 8px rgba(0,0,0,.6);';
    return L.divIcon({{className:'',html:'<div style="background:#0a1a24;border:2px solid '+c+';border-radius:6px;width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:12px;color:'+c+';'+glow+'cursor:pointer;position:relative"><span>'+id+'</span><div style="position:absolute;bottom:-5px;left:50%;transform:translateX(-50%);width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-top:6px solid '+c+'"></div></div>',iconSize:[32,32],iconAnchor:[16,38]}});
  }}

  function makeRSSIIcon(){{
    return L.divIcon({{className:'',html:'<div style="background:rgba(0,230,255,.15);border:2px solid #00e5ff;border-radius:50%;width:20px;height:20px;box-shadow:0 0 12px #00e5ff99;animation:spin .8s linear infinite"></div>',iconSize:[20,20],iconAnchor:[10,10]}});
  }}

  // ── Place markers ─────────────────────────────────────────────────────────
  function placePin(slot,x,y,pinData,pulse){{
    if(markers[slot]) map.removeLayer(markers[slot]);
    var m=L.marker([y,x],{{icon:makePinIcon(slot,pulse)}}).addTo(map);
    if(!editMode && !anchorMode){{
      m.on('click',function(e){{L.DomEvent.stopPropagation(e);openInfoPanel('pin',slot);}});
    }}
    markers[slot]=m;
  }}

  function placeAnchor(id,x,y,aData,pulse){{
    if(anchorMarkers[id]) map.removeLayer(anchorMarkers[id]);
    var m=L.marker([y,x],{{icon:makeAnchorIcon(id,pulse)}}).addTo(map);
    m.on('click',function(e){{L.DomEvent.stopPropagation(e);openInfoPanel('anchor',id);}});
    anchorMarkers[id]=m;
  }}

  // ── Load existing ─────────────────────────────────────────────────────────
  Object.keys(pins).forEach(function(slot){{
    var p=pins[slot]; placePin(parseInt(slot),p.x,p.y,p,false);
    var b=document.getElementById('btn-'+slot); if(b) b.classList.add('placed');
  }});
  Object.keys(anchors).forEach(function(id){{
    var a=anchors[id]; placeAnchor(id,a.x,a.y,a,false);
    var b=document.getElementById('abtn-'+id); if(b) b.classList.add('placed');
  }});

  // ── RSSI position marker ──────────────────────────────────────────────────
  var rssiMarker=null;
  if(rssiPos){{
    rssiMarker=L.marker([rssiPos.y,rssiPos.x],{{icon:makeRSSIIcon(),zIndexOffset:1000}}).addTo(map);
    rssiMarker.bindTooltip('📡 Estimated Position',{{permanent:false,direction:'top'}});
  }}

  // ── Slot selection ─────────────────────────────────────────────────────────
  function selectSlot(slot){{
    if(activeSlot!==null){{document.getElementById('btn-'+activeSlot).classList.remove('active');if(markers[activeSlot])markers[activeSlot].setIcon(makePinIcon(activeSlot,false));}}
    if(activeSlot===slot){{activeSlot=null;setStatus('No slot selected · Press 1–9');return;}}
    activeSlot=slot;
    document.getElementById('btn-'+slot).classList.add('active');
    if(markers[slot]){{markers[slot].setIcon(makePinIcon(slot,true));map.panTo(markers[slot].getLatLng());}}
    setStatus('Pin '+slot+' · '+(pins[slot]?'click to move':'click map to place'));
  }}

  function selectAnchor(id){{
    if(activeAnchor!==null){{document.getElementById('abtn-'+activeAnchor).classList.remove('active');if(anchorMarkers[activeAnchor])anchorMarkers[activeAnchor].setIcon(makeAnchorIcon(activeAnchor,false));}}
    if(activeAnchor===id){{activeAnchor=null;setStatus('No anchor selected · Press A–F');return;}}
    activeAnchor=id;
    document.getElementById('abtn-'+id).classList.add('active');
    if(anchorMarkers[id]){{anchorMarkers[id].setIcon(makeAnchorIcon(id,true));map.panTo(anchorMarkers[id].getLatLng());}}
    setStatus('Anchor '+id+' · '+(anchors[id]?'click to move':'click map to place'));
  }}

  document.addEventListener('keydown',function(e){{
    var k=e.key.toUpperCase();
    if(editMode){{var n=parseInt(e.key);if(n>=1&&n<=9)selectSlot(n);}}
    if(anchorMode){{if(['A','B','C','D','E','F'].includes(k))selectAnchor(k);}}
  }});

  // ── Cursor + status ───────────────────────────────────────────────────────
  if(editMode||anchorMode){{
    map.getContainer().style.cursor='crosshair';
    setStatus(editMode
      ?'Press 1–9 to select a pin slot, then click the map'
      :'Press A–F to select an anchor, then click the map to place it');
  }} else {{
    map.getContainer().style.cursor='default';
    if(editMode||anchorMode) document.getElementById('toolbar').style.display='none';
    var pinCount=Object.keys(pins).length, anchorCount=Object.keys(anchors).length;
    setStatus('📍 '+pinCount+' pins  📡 '+anchorCount+' anchors'+(rssiPos?' · 🟢 Live RSSI position active':''));
  }}

  function setStatus(msg,saving){{
    document.getElementById('status-text').textContent=msg;
    document.getElementById('spinner').style.display=saving?'block':'none';
  }}
</script>
</body>
</html>"""

    components.html(html_code, height=580, scrolling=False)

    # ── Manual save fallback ────────────────────────────────────────────────
    if edit_mode or anchor_mode:
        with st.expander("💾 Manual Save (fallback)", expanded=False):
            if edit_mode:
                pin_input = st.text_area("Pin JSON", value=pins_json, height=70, key="fp_pin_manual",
                                          placeholder='{"1":{"x":100,"y":200,"label":"Pin 1"}}')
                if st.button("💾 Save Pins", use_container_width=True, type="primary", key="manual_pin_save"):
                    try:
                        np = json.loads(pin_input.strip())
                        if isinstance(np, dict):
                            save_pins(username, np)
                            st.success(f"✅ {len(np)} pin(s) saved!")
                            st.rerun()
                    except Exception as ex:
                        st.error(f"❌ {ex}")
            if anchor_mode:
                anch_input = st.text_area("Anchor JSON", value=anchors_json, height=70, key="fp_anch_manual")
                if st.button("💾 Save Anchors", use_container_width=True, type="primary", key="manual_anch_save"):
                    try:
                        na = json.loads(anch_input.strip())
                        if isinstance(na, dict):
                            save_anchors(username, na)
                            st.success(f"✅ {len(na)} anchor(s) saved!")
                            st.rerun()
                    except Exception as ex:
                        st.error(f"❌ {ex}")

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with side_col:
    pins    = get_pins(username)
    anchors = get_anchors(username)

    sidebar_tab = st.radio("Sidebar", ["📍 Pins", "📡 Anchors"], horizontal=True, label_visibility="collapsed")

    if sidebar_tab == "📍 Pins":
        st.subheader("📍 Pins")
        st.divider()
        if pins:
            for slot in sorted(pins.keys(), key=lambda x: int(x)):
                pin   = pins[slot]
                color = COLORS_HEX.get(slot, "#888")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px">'
                    f'<div style="background:{color};color:#fff;border-radius:50%;width:24px;height:24px;min-width:24px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:12px">{slot}</div>'
                    f'<b style="font-size:13px">{pin["label"]}</b></div>'
                    f'<div style="color:gray;font-size:10px;margin-left:32px;margin-bottom:4px">x:{pin["x"]} y:{pin["y"]}</div>',
                    unsafe_allow_html=True)
                if edit_mode:
                    new_label = st.text_input("Label", value=pin.get("label", f"Pin {slot}"), key=f"lbl_{slot}", placeholder="Pin name…")
                    if new_label.strip() and new_label.strip() != pin.get("label"):
                        set_label(username, slot, new_label.strip()); st.rerun()
                    new_desc = st.text_area("Description", value=pin.get("description",""), key=f"desc_{slot}", placeholder="What is this?", height=60)
                    if new_desc != pin.get("description",""):
                        set_pin_info(username, slot, "description", new_desc)
                    status_options = ["","Active","Inactive","Maintenance"]
                    cur_status = pin.get("status","")
                    idx = status_options.index(cur_status) if cur_status in status_options else 0
                    new_status = st.selectbox("Status", status_options, index=idx, key=f"stat_{slot}")
                    if new_status != cur_status:
                        set_pin_info(username, slot, "status", new_status)
                    new_contact = st.text_input("Contact", value=pin.get("contact",""), key=f"contact_{slot}")
                    if new_contact != pin.get("contact",""):
                        set_pin_info(username, slot, "contact", new_contact)
                    new_notes = st.text_area("Notes", value=pin.get("notes",""), key=f"notes_{slot}", height=60)
                    if new_notes != pin.get("notes",""):
                        set_pin_info(username, slot, "notes", new_notes)
                    if st.button(f"🗑 Remove #{slot}", key=f"del_{slot}", use_container_width=True):
                        remove_pin(username, slot); st.rerun()
                else:
                    if pin.get("description"):
                        st.caption(f'📝 {pin["description"][:55]}…' if len(pin.get("description",""))>55 else f'📝 {pin["description"]}')
                    if pin.get("status"):
                        st.caption(f'● {pin["status"]}')
                st.divider()
            if edit_mode:
                if st.button("🗑️ Clear all pins", use_container_width=True):
                    save_pins(username, {}); st.rerun()
        else:
            st.info("No pins yet. Press ✏️ Edit Pins.")

    else:  # Anchors tab
        st.subheader("📡 Anchors")
        st.caption("RSSI-based positioning anchors (BLE/WiFi)")
        st.divider()

        # Global scale setting
        scale = st.number_input("Scale (px per meter)", min_value=1.0, max_value=1000.0,
                                 value=50.0, step=5.0, key="global_scale",
                                 help="How many image pixels = 1 real-world meter. Measure a known distance on your floor plan.")

        if anchors:
            for aid in sorted(anchors.keys()):
                a = anchors[aid]
                color = ANCHOR_COLORS.get(aid, "#00bcd4")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:2px">'
                    f'<div style="background:#0a1a24;border:2px solid {color};color:{color};border-radius:5px;width:24px;height:24px;min-width:24px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:12px">{aid}</div>'
                    f'<b style="font-size:13px">{a.get("name","Anchor "+aid)}</b></div>'
                    f'<div style="color:gray;font-size:10px;margin-left:32px;margin-bottom:4px">x:{a.get("x","?")} y:{a.get("y","?")}</div>',
                    unsafe_allow_html=True)

                if anchor_mode:
                    # Name
                    new_name = st.text_input("Name", value=a.get("name", f"Anchor {aid}"), key=f"aname_{aid}")
                    if new_name != a.get("name"):
                        anchors[aid]["name"] = new_name
                        save_anchors(username, anchors)

                    # MAC address
                    new_mac = st.text_input("MAC / SSID", value=a.get("mac",""), key=f"amac_{aid}",
                                             placeholder="aa:bb:cc:dd:ee:ff")
                    if new_mac != a.get("mac",""):
                        anchors[aid]["mac"] = new_mac
                        save_anchors(username, anchors)

                    # RSSI calibration
                    new_rssi_ref = st.number_input("RSSI @ 1m (dBm)", min_value=-100, max_value=0,
                                                    value=int(a.get("rssi_ref", -40)), key=f"arssi_{aid}")
                    if new_rssi_ref != a.get("rssi_ref", -40):
                        anchors[aid]["rssi_ref"] = new_rssi_ref
                        save_anchors(username, anchors)

                    new_n = st.number_input("Path loss exponent", min_value=1.0, max_value=6.0,
                                             value=float(a.get("path_loss_n", 2.0)), step=0.1, key=f"an_{aid}")
                    if new_n != a.get("path_loss_n", 2.0):
                        anchors[aid]["path_loss_n"] = new_n
                        save_anchors(username, anchors)

                    new_desc = st.text_input("Description", value=a.get("desc",""), key=f"adesc_{aid}",
                                              placeholder="e.g. Mounted on NW ceiling")
                    if new_desc != a.get("desc",""):
                        anchors[aid]["desc"] = new_desc
                        save_anchors(username, anchors)

                    if st.button(f"🗑 Remove Anchor {aid}", key=f"adel_{aid}", use_container_width=True):
                        anchors.pop(aid)
                        save_anchors(username, anchors)
                        st.rerun()
                else:
                    if a.get("mac"):
                        st.caption(f'📶 {a["mac"]}')
                    if a.get("last_rssi"):
                        st.caption(f'📡 Last RSSI: {a["last_rssi"]} dBm')

                st.divider()

            if anchor_mode:
                if st.button("🗑️ Clear all anchors", use_container_width=True):
                    save_anchors(username, {}); st.rerun()
        else:
            if anchor_mode:
                st.info("Press **A–F** to select an anchor slot, then click the map to place it.")
            else:
                st.info("No anchors yet. Click 📡 Edit Anchors to start.")

        # ── Live RSSI status ──────────────────────────────────────────────────
        if rssi_position:
            st.divider()
            st.success(f"🟢 Live Position\nx: **{rssi_position['x']}** y: **{rssi_position['y']}**")

        # ── RSSI Test tool ────────────────────────────────────────────────────
        if anchors and not anchor_mode:
            with st.expander("🧪 Test Trilateration"):
                st.caption("Simulate RSSI values to test positioning")
                test_vals = {}
                for aid in anchors:
                    test_vals[aid] = st.slider(f"Anchor {aid} RSSI", -100, -20, -65, key=f"trssi_{aid}")
                if st.button("▶ Compute Position", use_container_width=True):
                    anchors_all = get_anchors(username)
                    awrssi = []
                    for aid, rv in test_vals.items():
                        if aid in anchors_all and "x" in anchors_all[aid]:
                            a = anchors_all[aid]
                            awrssi.append({"x":a["x"],"y":a["y"],"rssi":rv,
                                           "rssi_ref":a.get("rssi_ref",-40),
                                           "path_loss_n":a.get("path_loss_n",2.0)})
                    ex, ey = trilaterate(awrssi, scale_px_per_m=scale)
                    if ex:
                        st.success(f"📍 Estimated: x={round(ex)} y={round(ey)}")
                    else:
                        st.warning("Need at least 2 placed anchors.")
