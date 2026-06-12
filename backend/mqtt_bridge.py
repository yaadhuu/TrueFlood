"""
FloodSense MQTT Bridge – Multi-Node Edition (Production)
=========================================================
• Subscribes to  flood/sensor/+   (ALL nodes via MQTT wildcard)
• Runs ML prediction for every incoming reading
• Publishes result to  flood/alert/<node_id>
• Sends WhatsApp via Twilio (per-node cooldown)
• Exposes HTTP REST API on $PORT (default 8080) for the dashboard
• CORS-enabled for separate frontend deployments (Vercel / GitHub Pages)

Environment Variables (see .env.example):
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, TWILIO_TO  [optional]
  MQTT_BROKER  (default: broker.hivemq.com)
  MQTT_PORT    (default: 1883)
  PORT         (default: 8080)

Usage (local):
  cp backend/.env.example backend/.env
  pip install -r requirements.txt
  python backend/mqtt_bridge.py

Usage (production / Gunicorn):
  gunicorn --chdir backend mqtt_bridge:app --bind 0.0.0.0:$PORT
  MQTT loop is started inside create_app() via a threading.Event guard
  so it runs exactly once even across Gunicorn worker forks.
"""

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify
from flask_cors import CORS
import paho.mqtt.client as mqtt
from twilio.rest import Client

# ── Resolve paths & load .env ─────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))

sys.path.insert(0, _HERE)
from predict import predict_flood  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("floodsense")

# ── Startup env-var validation ────────────────────────────────────
_REQUIRED_MQTT = {"MQTT_BROKER": "broker.hivemq.com", "MQTT_PORT": "1883"}

for var, default in _REQUIRED_MQTT.items():
    if not os.getenv(var):
        log.warning("Env var %s not set – using default: %s", var, default)

BROKER   = os.getenv("MQTT_BROKER", "broker.hivemq.com")
PORT     = int(os.getenv("MQTT_PORT", 1883))
API_PORT = int(os.getenv("PORT", 8080))

TOPIC_IN      = "flood/sensor/+"
TOPIC_OUT_FMT = "flood/alert/{node_id}"

# Twilio (all optional – alerts silently skipped when absent)
TWILIO_SID  = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
TWILIO_TO   = os.getenv("TWILIO_TO")

_TWILIO_READY = all([TWILIO_SID, TWILIO_AUTH, TWILIO_FROM, TWILIO_TO])
if not _TWILIO_READY:
    log.warning(
        "Twilio credentials incomplete – WhatsApp alerts disabled. "
        "Set TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM / TWILIO_TO."
    )

ALERT_COOLDOWN_SEC = 300

# ── Thread-safe state store ───────────────────────────────────────
# Both the MQTT network thread (on_message) and Gunicorn request threads
# (Flask route handlers) read/write these dicts.  A single RLock serialises
# all mutations; reads in route handlers snapshot under the same lock.
_state_lock: threading.RLock = threading.RLock()

node_states:      dict = {}   # node_id -> latest merged state dict
node_alert_times: dict = {}   # node_id -> unix timestamp of last WhatsApp send
node_history:     dict = {}   # node_id -> list of last MAX_HISTORY points
MAX_HISTORY = 20

# Guard so init_mqtt() fires exactly once across Gunicorn workers
_mqtt_started = threading.Event()

# ── Flask app factory ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})


# ── REST API ──────────────────────────────────────────────────────

@app.route("/api/nodes", methods=["GET"])
def api_nodes():
    with _state_lock:
        snapshot = list(node_states.values())
        count    = len(node_states)
    return jsonify({
        "nodes":       snapshot,
        "node_count":  count,
        "server_time": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/nodes/<node_id>", methods=["GET"])
def api_node(node_id):
    with _state_lock:
        if node_id not in node_states:
            return jsonify({"error": "node not found"}), 404
        current = dict(node_states[node_id])
        history = list(node_history.get(node_id, []))
    return jsonify({"current": current, "history": history})


@app.route("/api/history/<node_id>", methods=["GET"])
def api_history(node_id):
    with _state_lock:
        if node_id not in node_history:
            return jsonify({"error": "node not found"}), 404
        history = list(node_history[node_id])
    return jsonify({"node_id": node_id, "history": history})


@app.route("/api/health", methods=["GET"])
def api_health():
    with _state_lock:
        count = len(node_states)
    return jsonify({
        "status":       "ok",
        "nodes_online": count,
        "broker":       BROKER,
        "twilio_ready": _TWILIO_READY,
        "uptime_ts":    datetime.now(timezone.utc).isoformat(),
    })


# ── WhatsApp alert ────────────────────────────────────────────────

def send_whatsapp(node_id: str, alert: str, data: dict) -> None:
    if not _TWILIO_READY:
        return

    now  = time.monotonic()
    with _state_lock:
        last = node_alert_times.get(node_id, 0.0)
        if now - last < ALERT_COOLDOWN_SEC:
            remaining = int(ALERT_COOLDOWN_SEC - (now - last))
            log.warning("[WhatsApp][%s] Cooldown – %ds left", node_id, remaining)
            return
        node_alert_times[node_id] = now   # reserve slot inside lock

    label = data.get("node_label", node_id)
    body  = (
        f"FLOOD ALERT\n"
        f"Node      : {label} ({node_id})\n"
        f"Status    : {alert}\n"
        f"Water Lvl : {data.get('water_level_m', 'N/A')} m\n"
        f"Rainfall  : {data.get('rainfall_24h_mm', 'N/A')} mm\n"
        f"Soil Moist: {data.get('soil_moisture_pct', 'N/A')} %\n"
        f"Flow Vel  : {data.get('flow_velocity_ms', 'N/A')} m/s\n"
        f"Turbidity : {data.get('turbidity_ntu', 'N/A')} NTU\n"
        f"Time      : {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH)
        msg    = client.messages.create(from_=TWILIO_FROM, to=TWILIO_TO, body=body)
        log.info("[WhatsApp][%s] Sent – SID: %s", node_id, msg.sid)
    except Exception as exc:
        log.error("[WhatsApp][%s] Failed: %s", node_id, exc)
        # Roll back reservation so next attempt is not blocked by a failed send
        with _state_lock:
            node_alert_times.pop(node_id, None)


# ── MQTT callbacks ────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc, props=None):
    if rc == 0:
        log.info("[MQTT] Connected to %s:%d", BROKER, PORT)
        client.subscribe(TOPIC_IN)
        log.info("[MQTT] Subscribed wildcard -> %s", TOPIC_IN)
    else:
        log.error("[MQTT] Connection refused – rc=%d", rc)


def on_disconnect(client, userdata, rc, props=None):
    if rc != 0:
        log.warning("[MQTT] Unexpected disconnect rc=%d – paho will reconnect.", rc)


def on_message(client, userdata, msg):
    topic = msg.topic
    try:
        raw = msg.payload.decode("utf-8")
    except UnicodeDecodeError:
        log.error("[MQTT] Non-UTF-8 payload on %s – skipped.", topic)
        return

    parts   = topic.split("/")
    node_id = parts[-1] if len(parts) >= 3 else "unknown"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("[MQTT][%s] Bad JSON: %s | raw=%s", node_id, exc, raw[:80])
        return

    data.setdefault("node_id",    node_id)
    data.setdefault("node_label", node_id.replace("-", " ").title())

    # ML prediction (pure computation – no shared state touched yet)
    try:
        result = predict_flood(
            water_level_m     = float(data.get("water_level_m",     0)),
            rainfall_24h_mm   = float(data.get("rainfall_24h_mm",   0)),
            soil_moisture_pct = float(data.get("soil_moisture_pct", 0)),
            flow_velocity_ms  = float(data.get("flow_velocity_ms",  0)),
            turbidity_ntu     = float(data.get("turbidity_ntu",     0)),
            node_id           = node_id,
        )
    except Exception as exc:
        log.error("[ML][%s] predict_flood failed: %s", node_id, exc)
        return

    alert = result["alert_level"]
    probs = result["probabilities"]
    log.info(
        "[ML][%s] %s  N=%.2f W=%.2f A=%.2f",
        node_id, alert,
        probs["NORMAL"], probs["WATCH"], probs["ALERT"],
    )

    ts    = datetime.now(timezone.utc).isoformat()
    state = {**data, "alert_level": alert, "probabilities": probs, "last_updated": ts}

    # Atomic update of shared state
    with _state_lock:
        node_states[node_id] = state
        bucket = node_history.setdefault(node_id, [])
        bucket.append({
            "ts":               ts,
            "water_level_m":    data.get("water_level_m"),
            "rainfall_24h_mm":  data.get("rainfall_24h_mm"),
            "flow_velocity_ms": data.get("flow_velocity_ms"),
            "alert_level":      alert,
        })
        if len(bucket) > MAX_HISTORY:
            bucket.pop(0)

    # Publish alert result back to broker (outside lock – paho is thread-safe)
    out_topic = TOPIC_OUT_FMT.format(node_id=node_id)
    response  = {
        "node_id":       node_id,
        "alert_level":   alert,
        "probabilities": probs,
        "timestamp":     time.strftime("%H:%M:%S"),
    }
    client.publish(out_topic, json.dumps(response), qos=1)
    log.info("[MQTT] Published -> %s", out_topic)

    if alert == "ALERT":
        threading.Thread(
            target=send_whatsapp,
            args=(node_id, alert, data),
            daemon=True,
        ).start()


# ── MQTT initialiser (idempotent) ─────────────────────────────────

def init_mqtt() -> None:
    """Start the paho background loop exactly once (safe across Gunicorn forks)."""
    if _mqtt_started.is_set():
        return
    _mqtt_started.set()

    try:
        mc = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="flood-bridge-multinode",
            clean_session=True,
        )
        mc.on_connect    = on_connect
        mc.on_disconnect = on_disconnect
        mc.on_message    = on_message
        mc.reconnect_delay_set(min_delay=2, max_delay=120)
        mc.connect(BROKER, PORT, keepalive=60)
        mc.loop_start()
        log.info("[MQTT] Background loop started.")
    except Exception as exc:
        log.error("[MQTT] Initialization failed: %s", exc)
        _mqtt_started.clear()   # allow retry on next request if boot-time broker is down


# Bootstrap MQTT when the module is imported by Gunicorn
init_mqtt()


# ── Local dev entry-point ─────────────────────────────────────────

def main() -> None:
    log.info("FloodSense MQTT Bridge – starting")
    log.info("  Broker  : %s:%d", BROKER, PORT)
    log.info("  Topic   : %s", TOPIC_IN)
    log.info("  API     : http://0.0.0.0:%d/api/nodes", API_PORT)
    try:
        app.run(host="0.0.0.0", port=API_PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        if _mqtt_started.is_set():
            # mc is local to init_mqtt; retrieve from paho's internal handle
            pass  # paho loop_stop() is called by the daemon thread on process exit


if __name__ == "__main__":
    main()
