"""
FloodSense MQTT Bridge – Multi-Node Edition (Production)
=========================================================
• Subscribes to  flood/sensor/+   (ALL nodes via MQTT wildcard)
• Runs ML prediction for every incoming reading
• Publishes result to  flood/alert/<node_id>
• Sends WhatsApp via Twilio (per-node cooldown, full error codes logged)
• Exposes HTTP REST API on $PORT (default 8080) for the dashboard
• CORS-enabled for separate frontend deployments (Vercel / GitHub Pages)

Environment Variables (see .env.example):
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, TWILIO_TO  [optional]
  TEST_SECRET    — shared secret for POST /api/test-alert  [optional]
  DEMO_MODE      — set to "true" to enable POST /api/simulate
  MQTT_BROKER    (default: broker.hivemq.com)
  MQTT_PORT      (default: 8883 — TLS)
  PORT           (default: 8080)

Usage (local):
  cp backend/.env.example backend/.env
  pip install -r requirements.txt
  python backend/mqtt_bridge.py

Usage (production / Gunicorn — MUST use --workers 1):
  gunicorn --chdir backend mqtt_bridge:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
  MQTT loop is started inside create_app() via a threading.Event guard
  so it runs exactly once even if Gunicorn ever forks.
"""

import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
import paho.mqtt.client as mqtt
from twilio.rest import Client
from twilio.base.exceptions import TwilioRestException

# ── Resolve paths & load .env ─────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"), override=True)

sys.path.insert(0, _HERE)
from predict import predictor  # noqa: E402
import db  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("floodsense")

# ── Startup env-var validation ────────────────────────────────────
BROKER   = os.getenv("MQTT_BROKER", "broker.hivemq.com")
PORT     = int(os.getenv("MQTT_PORT", 8883))   # 8883 = TLS; use 1883 for non-TLS local
API_PORT = int(os.getenv("PORT", 8080))

# Demo mode — POST /api/simulate only available when DEMO_MODE=true
DEMO_MODE = os.getenv("DEMO_MODE", "").lower() in ("true", "1", "yes")

# Test alert endpoint secret (header X-Test-Key must match)
TEST_SECRET = os.getenv("TEST_SECRET", "")

TOPIC_IN      = "flood/sensor/+"
TOPIC_OUT_FMT = "flood/alert/{node_id}"

# ── Twilio (all optional — alerts silently skipped when absent) ───
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
_state_lock: threading.RLock = threading.RLock()

node_alert_times: dict = {}   # node_id -> unix timestamp of last WhatsApp send
MAX_HISTORY = 20

# Alert attempt log (last 50, used by GET /api/alerts/log)
_alert_log: deque = deque(maxlen=50)

# MQTT connected flag (updated in on_connect / on_disconnect)
_mqtt_connected = False

# Guard so init_mqtt() fires exactly once across Gunicorn workers
_mqtt_started = threading.Event()

# Unique client_id per process restart — prevents HiveMQ disconnect loop
# when two workers share the same static client_id.
_MQTT_CLIENT_ID = f"flood-bridge-{uuid.uuid4().hex[:8]}"

# ── Flask app factory ─────────────────────────────────────────────
app = Flask(__name__)
# CORS: keep wildcard for now — lock down to your frontend origin before going public
# e.g. origins=["https://your-frontend.vercel.app"]
CORS(app, resources={r"/api/*": {"origins": "*"}})


# ── Validation helpers ────────────────────────────────────────────

_SENSOR_FIELDS = [
    "water_level_m", "rainfall_24h_mm", "soil_moisture_pct",
    "flow_velocity_ms", "turbidity_ntu",
]

def _validate_sensor_payload(data: dict) -> tuple[dict, str | None]:
    """
    Validate that all sensor fields are present and numeric.
    Returns (cleaned_floats, None) on success or (_, error_msg) on failure.
    """
    clean = {}
    for field in _SENSOR_FIELDS:
        raw = data.get(field)
        if raw is None:
            return {}, f"Missing required field: {field}"
        try:
            clean[field] = float(raw)
        except (TypeError, ValueError):
            return {}, f"Field '{field}' must be numeric, got: {repr(raw)}"
    return clean, None


# ── REST API ──────────────────────────────────────────────────────

@app.route("/api/nodes", methods=["GET"])
def api_nodes():
    snapshot = db.get_all_nodes()
    count    = len(snapshot)
    return jsonify({
        "nodes":       snapshot,
        "node_count":  count,
        "server_time": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/nodes/<node_id>", methods=["GET"])
def api_node(node_id):
    current = db.get_node(node_id)
    if not current:
        return jsonify({"error": "node not found"}), 404
    history = db.get_node_history(node_id, limit=MAX_HISTORY)
    return jsonify({"current": current, "history": history})


@app.route("/api/history/<node_id>", methods=["GET"])
def api_history(node_id):
    history = db.get_node_history(node_id, limit=MAX_HISTORY)
    if not history:
        return jsonify({"error": "node not found"}), 404
    return jsonify({"node_id": node_id, "history": history})


@app.route("/api/health", methods=["GET"])
def api_health():
    count = len(db.get_all_nodes())
    return jsonify({
        "status":         "ok",
        "nodes_online":   count,
        "broker":         BROKER,
        "mqtt_port":      PORT,
        "mqtt_connected": _mqtt_connected,
        "twilio_ready":   _TWILIO_READY,
        "demo_mode":      DEMO_MODE,
        "uptime_ts":      datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/version", methods=["GET"])
def api_version():
    return jsonify({"version": "2.0.0", "build": "phases-1-to-5"})


@app.route("/api/alerts/log", methods=["GET"])
def api_alerts_log():
    """Last 50 WhatsApp send attempts with outcome and Twilio error codes."""
    with _state_lock:
        log_snapshot = list(_alert_log)
    return jsonify({"attempts": log_snapshot, "count": len(log_snapshot)})


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    Inject a synthetic sensor reading through the full ML pipeline.
    Only available when DEMO_MODE=true (set env var DEMO_MODE=true).
    """
    if not DEMO_MODE:
        return jsonify({"error": "Simulation endpoint is disabled. Set DEMO_MODE=true to enable."}), 403

    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    node_id = str(body.get("node_id", "sim-node")).strip()
    if not node_id:
        return jsonify({"error": "node_id must not be empty"}), 422

    floats, err = _validate_sensor_payload(body)
    if err:
        return jsonify({"error": err}), 422

    # Run ML prediction
    try:
        result = predictor.predict(node_id, floats)
    except Exception as exc:
        log.error("[ML][simulate] prediction failed: %s", exc)
        return jsonify({"error": f"ML prediction failed: {exc}"}), 500

    alert = result["status"]
    probs = result["probabilities"]
    ts    = datetime.now(timezone.utc).isoformat()

    data = {**floats, "node_id": node_id, "node_label": node_id.replace("-", " ").title()}
    state = {**data, "alert_level": alert, "probabilities": probs, "last_updated": ts}

    # Update SQLite database
    db.update_node_state(node_id, state)
    db.insert_telemetry(node_id, data, alert)

    log.info("[simulate][%s] %s P=%s", node_id, alert, probs)

    # Fire WhatsApp on ALERT — normal 300s per-node cooldown applies.
    # force=True is ONLY used by /api/test-alert (direct Twilio test).
    if alert == "ALERT":
        threading.Thread(
            target=send_whatsapp,
            args=(node_id, alert, data),
            daemon=True,
        ).start()

    return jsonify({
        "node_id":      node_id,
        "alert_level":  alert,
        "probabilities": probs,
        "timestamp":    ts,
    })


@app.route("/api/alert/send", methods=["POST"])
def api_alert_send():
    """
    Dashboard-facing "Send Alert" button.
    Sends a real WhatsApp message for a node's LATEST known reading to a
    phone number the user types into the UI (no server secret needed).

    Body: { "node_id": "node-alpha", "to": "+919876543210" }

    Safety:
      - "to" is validated as digits/plus only and formatted as whatsapp:+E164.
      - Uses the SAME per-node cooldown as automatic alerts (300s) so the
        public endpoint can't be used to spam Twilio credits.
      - If the node has no telemetry yet, returns 404 instead of sending
        a message with fabricated numbers.
    """
    if not _TWILIO_READY:
        return jsonify({"error": "Twilio credentials not configured on server."}), 503

    body = request.get_json(silent=True) or {}
    node_id = str(body.get("node_id", "")).strip()
    to_raw  = str(body.get("to", "")).strip()

    if not node_id:
        return jsonify({"error": "node_id is required"}), 422
    if not to_raw:
        return jsonify({"error": "to (phone number) is required"}), 422

    digits = to_raw.replace(" ", "").replace("-", "")
    if not digits.startswith("+"):
        digits = "+" + digits.lstrip("+")
    if not digits[1:].isdigit() or len(digits) < 8:
        return jsonify({"error": "to must be a valid phone number, e.g. +919876543210"}), 422
    to_whatsapp = f"whatsapp:{digits}"

    node = db.get_node(node_id)
    if not node:
        return jsonify({"error": f"No telemetry received yet for node '{node_id}'"}), 404

    alert = node.get("alert_level", "NORMAL")
    result = send_whatsapp(node_id, alert, node, to_override=to_whatsapp, return_result=True)

    if result.get("success"):
        return jsonify({"sent": True, "sid": result.get("sid"), "to": digits})
    if result.get("error") == "cooldown":
        return jsonify({"sent": False, "error": "This node already sent an alert in the last 5 minutes. Try again shortly."}), 429
    return jsonify({
        "sent":        False,
        "error":       result.get("error"),
        "twilio_code": result.get("twilio_code"),
        "twilio_msg":  result.get("twilio_msg"),
    }), 500


@app.route("/api/test-alert", methods=["POST"])
def api_test_alert():
    """
    Fire a real WhatsApp message directly (bypasses ML and MQTT).
    Requires header:  X-Test-Key: <TEST_SECRET env var>
    Used to isolate Twilio config from hardware/ML issues.
    """
    if not TEST_SECRET:
        return jsonify({"error": "TEST_SECRET not configured on server."}), 503

    key = request.headers.get("X-Test-Key", "")
    if key != TEST_SECRET:
        return jsonify({"error": "Invalid or missing X-Test-Key header"}), 401

    if not _TWILIO_READY:
        return jsonify({"error": "Twilio credentials not configured on server."}), 503

    body = request.get_json(silent=True) or {}
    node_id = str(body.get("node_id", "test-node")).strip() or "test-node"

    fake_data = {
        "node_label":        "TEST NODE",
        "water_level_m":     35.0,
        "rainfall_24h_mm":   32.0,
        "soil_moisture_pct": 91.0,
        "flow_velocity_ms":  3.5,
        "turbidity_ntu":     720.0,
    }

    result = send_whatsapp(node_id, "ALERT", fake_data, force=True, return_result=True)
    if result.get("success"):
        return jsonify({"sent": True, "sid": result.get("sid")})
    else:
        return jsonify({
            "sent":         False,
            "error":        result.get("error"),
            "twilio_code":  result.get("twilio_code"),
            "twilio_msg":   result.get("twilio_msg"),
        }), 500


# ── WhatsApp alert ────────────────────────────────────────────────

def send_whatsapp(
    node_id: str,
    alert: str,
    data: dict,
    *,
    force: bool = False,
    return_result: bool = False,
    to_override: str | None = None,
) -> dict:
    """
    Send a WhatsApp alert via Twilio.

    Args:
        force:         If True, bypass the per-node cooldown timer.
        return_result: If True, return a dict with {success, sid, error, twilio_code, twilio_msg}
                       instead of None — used by /api/test-alert.
        to_override:   If set, send to this WhatsApp number instead of the
                       server's TWILIO_TO env var. Must be "whatsapp:+<E.164>".
                       Lets anyone using the dashboard's "Send Alert" button
                       type their own number without a redeploy.
    """
    if not _TWILIO_READY:
        return {"success": False, "error": "Twilio not ready"}

    now = time.monotonic()

    if not force:
        with _state_lock:
            last = node_alert_times.get(node_id, 0.0)
            if now - last < ALERT_COOLDOWN_SEC:
                remaining = int(ALERT_COOLDOWN_SEC - (now - last))
                log.warning("[WhatsApp][%s] Cooldown – %ds left", node_id, remaining)
                return {"success": False, "error": "cooldown"}
            node_alert_times[node_id] = now   # reserve slot inside lock
    else:
        with _state_lock:
            node_alert_times[node_id] = now

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

    attempt = {
        "ts":      datetime.now(timezone.utc).isoformat(),
        "node_id": node_id,
        "alert":   alert,
    }

    send_to = to_override or TWILIO_TO

    try:
        client = Client(TWILIO_SID, TWILIO_AUTH)
        msg    = client.messages.create(from_=TWILIO_FROM, to=send_to, body=body)
        log.info("[WhatsApp][%s] Sent – SID: %s", node_id, msg.sid)
        attempt.update({"success": True, "sid": msg.sid})
        _alert_log.append(attempt)
        return {"success": True, "sid": msg.sid}

    except TwilioRestException as exc:
        # Log the actual Twilio error code and message — not just the str(exc)
        log.error(
            "[WhatsApp][%s] Twilio error code=%s msg=%s status=%s",
            node_id, exc.code, exc.msg, exc.status,
        )
        attempt.update({"success": False, "twilio_code": exc.code, "twilio_msg": exc.msg, "error": str(exc)})
        _alert_log.append(attempt)
        # Roll back reservation so the next attempt is not blocked
        with _state_lock:
            node_alert_times.pop(node_id, None)
        return {"success": False, "error": str(exc), "twilio_code": exc.code, "twilio_msg": exc.msg}

    except Exception as exc:
        log.error("[WhatsApp][%s] Unexpected error: %s", node_id, exc)
        attempt.update({"success": False, "error": str(exc)})
        _alert_log.append(attempt)
        with _state_lock:
            node_alert_times.pop(node_id, None)
        return {"success": False, "error": str(exc)}


# ── MQTT callbacks ────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc, props=None):
    global _mqtt_connected
    if rc == 0:
        _mqtt_connected = True
        log.info("[MQTT] Connected to %s:%d (client_id=%s)", BROKER, PORT, _MQTT_CLIENT_ID)
        client.subscribe(TOPIC_IN)
        log.info("[MQTT] Subscribed wildcard -> %s", TOPIC_IN)
    else:
        _mqtt_connected = False
        log.error("[MQTT] Connection refused – rc=%d", rc)


def on_disconnect(client, userdata, rc, props=None):
    global _mqtt_connected
    _mqtt_connected = False
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
        result = predictor.predict(node_id, data)
    except Exception as exc:
        log.error("[ML][%s] prediction failed: %s", node_id, exc)
        return

    alert = result["status"]
    probs = result["probabilities"]
    log.info(
        "[ML][%s] %s  N=%.2f W=%.2f A=%.2f",
        node_id, alert,
        probs["NORMAL"], probs["WATCH"], probs["ALERT"],
    )

    ts    = datetime.now(timezone.utc).isoformat()
    state = {**data, "alert_level": alert, "probabilities": probs, "last_updated": ts}

    # Atomic update of shared state via SQLite
    db.update_node_state(node_id, state)
    db.insert_telemetry(node_id, data, alert)

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
    """
    Start the paho background loop exactly once.
    - Unique client_id per process prevents HiveMQ disconnect loops
      when Gunicorn restarts without a full port release.
    - TLS (port 8883) matches the frontend's WSS (8884) path.
    """
    if _mqtt_started.is_set():
        return
    _mqtt_started.set()

    try:
        mc = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=_MQTT_CLIENT_ID,
            clean_session=True,
        )
        mc.on_connect    = on_connect
        mc.on_disconnect = on_disconnect
        mc.on_message    = on_message
        mc.reconnect_delay_set(min_delay=2, max_delay=120)

        # Enable TLS for port 8883
        if PORT == 8883:
            mc.tls_set()  # uses system CA store — no custom cert needed for HiveMQ public broker
            log.info("[MQTT] TLS enabled for port 8883")

        mc.connect(BROKER, PORT, keepalive=60)
        mc.loop_start()
        log.info("[MQTT] Background loop started. client_id=%s broker=%s:%d", _MQTT_CLIENT_ID, BROKER, PORT)
    except Exception as exc:
        log.error("[MQTT] Initialization failed: %s", exc)
        _mqtt_started.clear()   # allow retry on next request if boot-time broker is down


# Bootstrap MQTT when the module is imported by Gunicorn
db.init_db()
init_mqtt()


# ── Local dev entry-point ─────────────────────────────────────────

def main() -> None:
    log.info("FloodSense MQTT Bridge – starting")
    log.info("  Broker  : %s:%d", BROKER, PORT)
    log.info("  Topic   : %s", TOPIC_IN)
    log.info("  API     : http://0.0.0.0:%d/api/nodes", API_PORT)
    log.info("  Demo    : %s", DEMO_MODE)
    log.info("  client_id: %s", _MQTT_CLIENT_ID)
    try:
        app.run(host="0.0.0.0", port=API_PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        log.info("Shutting down...")


if __name__ == "__main__":
    main()
