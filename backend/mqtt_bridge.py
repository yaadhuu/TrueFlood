"""
FloodSense MQTT Bridge – Multi-Node Edition
===========================================
• Subscribes to  flood/sensor/+   (ALL nodes via MQTT wildcard)
• Subscribes to  flood/status/+   (retained LWT online/offline per node)
• Runs the three-layer predictor for every incoming reading
• Publishes the result to  flood/alert/<node_id>
• Sends WhatsApp via Twilio (authenticated, allowlisted, globally rate-budgeted)
  with SMS fallback
• Exposes a REST API + Prometheus metrics on $PORT (default 8080)

Environment variables (see .env.example):
  MQTT_BROKER, MQTT_PORT, MQTT_USER, MQTT_PASS
  PORT, ALLOWED_ORIGINS, DB_PATH, MODEL_PATH
  DASHBOARD_KEY        — required header X-Dashboard-Key for /api/alert/send
  ALERT_RECIPIENTS     — comma-separated E.164 allowlist for outbound alerts
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, TWILIO_TO
  SMS_FALLBACK_FROM    — Twilio SMS-capable number for the fallback path
  TEST_SECRET          — shared secret for POST /api/test-alert
  DEMO_MODE            — "true" enables POST /api/simulate
  NODE_STALE_SEC       — seconds without telemetry before a node reads NO DATA
  NODE_COORDS          — "node-1:lat,lon;node-2:lat,lon" for Layer 3

Production (Gunicorn — MUST be --workers 1; the MQTT client is process-local):
  gunicorn --chdir backend mqtt_bridge:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
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
import weather  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("floodsense")

# ── Config ────────────────────────────────────────────────────────
BROKER   = os.getenv("MQTT_BROKER", "broker.hivemq.com")
PORT     = int(os.getenv("MQTT_PORT", 8883))   # 8883 = TLS; 1883 = plaintext
API_PORT = int(os.getenv("PORT", 8080))
MQTT_USER = os.getenv("MQTT_USER") or None
MQTT_PASS = os.getenv("MQTT_PASS") or None

DEMO_MODE = os.getenv("DEMO_MODE", "").lower() in ("true", "1", "yes")
TEST_SECRET = os.getenv("TEST_SECRET", "")

# BLOCKER fix: /api/alert/send used to be entirely unauthenticated, so anyone
# could make this server send WhatsApp messages to any number on the owner's
# Twilio account. Two independent gates now apply, plus a global budget.
DASHBOARD_KEY = os.getenv("DASHBOARD_KEY", "")

NODE_STALE_SEC = int(os.getenv("NODE_STALE_SEC", 60))

TOPIC_IN       = "flood/sensor/+"
TOPIC_STATUS   = "flood/status/+"
TOPIC_OUT_FMT  = "flood/alert/{node_id}"

# ── Twilio ────────────────────────────────────────────────────────
TWILIO_SID  = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
TWILIO_TO   = os.getenv("TWILIO_TO")
SMS_FALLBACK_FROM = os.getenv("SMS_FALLBACK_FROM")

_TWILIO_READY = all([TWILIO_SID, TWILIO_AUTH, TWILIO_FROM, TWILIO_TO])
if not _TWILIO_READY:
    log.warning(
        "Twilio credentials incomplete – WhatsApp alerts disabled (auto AND "
        "manual). Set TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_FROM / "
        "TWILIO_TO in backend/.env, then restart. A Twilio sandbox account is "
        "free at twilio.com/console; the recipient number must WhatsApp "
        "'join <sandbox-code>' to +1 415 523 8886 before every test session "
        "after 3 days idle."
    )
if not DASHBOARD_KEY:
    log.warning(
        "DASHBOARD_KEY is unset – /api/alert/send will reject every request. "
        "Set it to enable the dashboard's Send Alert button."
    )

ALERT_RECIPIENTS = {
    n.strip() for n in os.getenv("ALERT_RECIPIENTS", "").split(",") if n.strip()
}
if not ALERT_RECIPIENTS and TWILIO_TO:
    # An empty allowlist makes /api/alert/send 403 for every number, including
    # the operator's own configured recipient — a trap for anyone who filled
    # in Twilio creds but didn't know ALERT_RECIPIENTS was a separate setting.
    # The number already trusted enough to receive automatic alerts is a safe
    # default; set ALERT_RECIPIENTS explicitly to allow additional numbers.
    _default_recipient = TWILIO_TO.replace("whatsapp:", "").strip()
    if _default_recipient:
        ALERT_RECIPIENTS = {_default_recipient}
        log.info(
            "ALERT_RECIPIENTS unset – defaulting the dashboard allowlist to "
            "TWILIO_TO (%s). Set ALERT_RECIPIENTS explicitly to change this.",
            _default_recipient,
        )

# One client for the process, not one per message: each Client() builds a new
# HTTP session and TLS context.
_twilio: Client | None = (
    Client(TWILIO_SID, TWILIO_AUTH) if (TWILIO_SID and TWILIO_AUTH) else None
)

ALERT_COOLDOWN_SEC = 300
GLOBAL_MAX_PER_HOUR = int(os.getenv("GLOBAL_MAX_PER_HOUR", 20))

# ── Thread-safe state ─────────────────────────────────────────────
_state_lock: threading.RLock = threading.RLock()

node_alert_times: dict = {}   # node_id -> monotonic timestamp of last send
node_link_state: dict = {}    # node_id -> "online" | "offline" (from MQTT LWT)
MAX_HISTORY = 20

_alert_log: deque = deque(maxlen=50)

# A per-node cooldown alone does not bound spend: the ceiling grows with every
# node added. This is an absolute cap across the whole deployment.
_global_sends: deque = deque(maxlen=200)

_mqtt_connected = False
_mqtt_started = threading.Event()
_MQTT_CLIENT_ID = f"flood-bridge-{uuid.uuid4().hex[:8]}"

# ── Flask ─────────────────────────────────────────────────────────
app = Flask(__name__)
_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
CORS(app, resources={r"/api/*": {"origins": _origins or "*"}})

# --workers 1 is load-bearing: two workers means two MQTT subscriptions, two
# predictors with separate hysteresis state, and duplicate alerts. Fail loudly.
_web_concurrency = int(os.getenv("WEB_CONCURRENCY", "1"))
if _web_concurrency != 1:
    raise RuntimeError(
        f"WEB_CONCURRENCY={_web_concurrency}; this app must run with exactly one "
        "worker (the MQTT client and predictor state are process-local)."
    )

# Required in every packet. flow_velocity_ms / turbidity_ntu were removed
# with their sensors; pressure_hpa and humidity_pct are optional because a
# node whose BMP180/DHT22 failed to init still has a usable water level and
# must not be rejected outright.
_SENSOR_FIELDS = [
    "water_level_m", "rainfall_24h_mm", "soil_moisture_pct",
]
_OPTIONAL_SENSOR_FIELDS = [
    "pressure_hpa", "pressure_trend_hpa_per_hr", "humidity_pct", "temperature_c",
]


def _validate_sensor_payload(data: dict) -> tuple[dict, str | None]:
    clean = {}
    for field in _SENSOR_FIELDS:
        raw = data.get(field)
        if raw is None:
            return {}, f"Missing required field: {field}"
        try:
            clean[field] = float(raw)
        except (TypeError, ValueError):
            return {}, f"Field '{field}' must be numeric, got: {repr(raw)}"
    for field in _OPTIONAL_SENSOR_FIELDS:
        raw = data.get(field)
        if raw is None:
            continue
        try:
            clean[field] = float(raw)
        except (TypeError, ValueError):
            return {}, f"Field '{field}' must be numeric, got: {repr(raw)}"
    return clean, None


# ── Node liveness ─────────────────────────────────────────────────

def _seconds_since_update(node: dict) -> float | None:
    raw = node.get("last_updated")
    if not raw:
        return None
    try:
        last = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last).total_seconds()


def _is_stale(node: dict) -> bool:
    """
    A dead sensor and a healthy one look identical unless something checks.
    Stale means: no telemetry within NODE_STALE_SEC, or the broker delivered
    this node's Last Will ("offline").
    """
    if node_link_state.get(node.get("node_id")) == "offline":
        return True
    age = _seconds_since_update(node)
    return True if age is None else age > NODE_STALE_SEC


def _annotate(node: dict) -> dict:
    age = _seconds_since_update(node)
    return {
        **node,
        "stale": _is_stale(node),
        "seconds_since_update": None if age is None else round(age, 1),
        "link_state": node_link_state.get(node.get("node_id"), "unknown"),
    }


def regional_check(all_nodes: list[dict]) -> str | None:
    """
    Cross-node corroboration. One sensor spiking can be a glitch; two live
    nodes elevated at once is a regional event. Context only - it never
    overrides an individual node's Layer-1 decision.
    """
    elevated = [n for n in all_nodes
                if n.get("alert_level") in ("WATCH", "ALERT") and not n.get("stale")]
    if len(elevated) >= 2:
        names = ", ".join(sorted(n.get("node_id", "?") for n in elevated))
        return (f"REGIONAL WATCH: {len(elevated)} nodes elevated simultaneously "
                f"({names})")
    return None


# ── REST API ──────────────────────────────────────────────────────

@app.route("/api/nodes", methods=["GET"])
def api_nodes():
    snapshot = [_annotate(n) for n in db.get_all_nodes()]
    return jsonify({
        "nodes":         snapshot,
        "node_count":    len(snapshot),
        "stale_count":   sum(1 for n in snapshot if n["stale"]),
        "regional_note": regional_check(snapshot),
        "server_time":   datetime.now(timezone.utc).isoformat(),
    })


@app.route("/api/nodes/<node_id>", methods=["GET"])
def api_node(node_id):
    current = db.get_node(node_id)
    if not current:
        return jsonify({"error": "node not found"}), 404
    history = db.get_node_history(node_id, limit=MAX_HISTORY)
    return jsonify({"current": _annotate(current), "history": history})


@app.route("/api/history/<node_id>", methods=["GET"])
def api_history(node_id):
    history = db.get_node_history(node_id, limit=MAX_HISTORY)
    if not history:
        return jsonify({"error": "node not found"}), 404
    return jsonify({"node_id": node_id, "history": history})


@app.route("/api/health", methods=["GET"])
def api_health():
    nodes = [_annotate(n) for n in db.get_all_nodes()]
    degraded = not _mqtt_connected
    body = {
        "status":         "degraded" if degraded else "ok",
        "nodes_total":    len(nodes),
        "nodes_online":   sum(1 for n in nodes if not n["stale"]),
        "nodes_stale":    sum(1 for n in nodes if n["stale"]),
        "broker":         BROKER,
        "mqtt_port":      PORT,
        "mqtt_connected": _mqtt_connected,
        "twilio_ready":   _TWILIO_READY,
        "sms_fallback":   bool(SMS_FALLBACK_FROM),
        "model_loaded":   predictor.model is not None,
        "demo_mode":      DEMO_MODE,
        "uptime_ts":      datetime.now(timezone.utc).isoformat(),
    }
    # 503 while degraded: a 200 here is why "healthy" was reported for a bridge
    # that was deaf to the broker.
    return jsonify(body), (503 if degraded else 200)


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Prometheus text exposition, hand-rolled (no extra dependency)."""
    nodes = [_annotate(n) for n in db.get_all_nodes()]
    lines = [
        "# HELP floodsense_nodes_total Registered nodes",
        "# TYPE floodsense_nodes_total gauge",
        f"floodsense_nodes_total {len(nodes)}",
        "# HELP floodsense_nodes_stale Nodes with no fresh telemetry",
        "# TYPE floodsense_nodes_stale gauge",
        f"floodsense_nodes_stale {sum(1 for n in nodes if n['stale'])}",
        "# HELP floodsense_mqtt_connected MQTT broker connection state",
        "# TYPE floodsense_mqtt_connected gauge",
        f"floodsense_mqtt_connected {1 if _mqtt_connected else 0}",
        "# HELP floodsense_model_loaded Layer 2 model availability",
        "# TYPE floodsense_model_loaded gauge",
        f"floodsense_model_loaded {1 if predictor.model is not None else 0}",
        "# HELP floodsense_alert_sends_last_hour Outbound alerts in the last hour",
        "# TYPE floodsense_alert_sends_last_hour gauge",
        f"floodsense_alert_sends_last_hour {_global_send_count()}",
        "# HELP floodsense_node_alert_level 0=NORMAL 1=WATCH 2=ALERT -1=unknown",
        "# TYPE floodsense_node_alert_level gauge",
    ]
    for n in nodes:
        lvl = {"NORMAL": 0, "WATCH": 1, "ALERT": 2}.get(n.get("alert_level"), -1)
        node_id = str(n.get("node_id", "unknown")).replace('"', "")
        lines.append(f'floodsense_node_alert_level{{node_id="{node_id}"}} {lvl}')
        lines.append(
            f'floodsense_node_seconds_since_update{{node_id="{node_id}"}} '
            f'{n.get("seconds_since_update") if n.get("seconds_since_update") is not None else -1}'
        )
    return "\n".join(lines) + "\n", 200, {"Content-Type": "text/plain; version=0.0.4"}


@app.route("/api/version", methods=["GET"])
def api_version():
    return jsonify({"version": "3.0.0", "build": "three-layer"})


@app.route("/api/alerts/log", methods=["GET"])
def api_alerts_log():
    with _state_lock:
        log_snapshot = list(_alert_log)
    return jsonify({"attempts": log_snapshot, "count": len(log_snapshot)})


@app.route("/api/weather/<node_id>", methods=["GET"])
def api_weather(node_id):
    fc = weather.get_forecast(node_id)
    if fc is None:
        return jsonify({
            "error": "no forecast for this node",
            "hint": "set NODE_COORDS=node-1:lat,lon;... to enable Layer 3",
            "configured_nodes": weather.configured_nodes(),
        }), 404
    return jsonify({"node_id": node_id, **fc})


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Inject a synthetic reading through the full pipeline. DEMO_MODE only."""
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

    result, state, data = _process_reading(node_id, {**body, **floats})
    log.info("[simulate][%s] %s reasons=%s", node_id, result["status"], result["reasons"])

    if result["status"] == "ALERT":
        threading.Thread(target=send_alert, args=(node_id, result["status"], data),
                         daemon=True).start()

    return jsonify({
        "node_id":           node_id,
        "alert_level":       result["status"],
        "reasons":           result["reasons"],
        "forecast_advisory": result["forecast_advisory"],
        "ood_features":      result["ood_features"],
        "probabilities":     result["probabilities"],
        "timestamp":         state["last_updated"],
    })


@app.route("/api/alert/send", methods=["POST"])
def api_alert_send():
    """
    Dashboard "Send Alert" button.

    Body: { "node_id": "node-1", "to": "+919876543210" }
    Header: X-Dashboard-Key: <DASHBOARD_KEY>

    Three gates, because this endpoint spends real money:
      1. shared secret (401 without it)
      2. recipient allowlist (403 for anything not in ALERT_RECIPIENTS)
      3. per-node cooldown + global hourly budget (429)
    """
    if not DASHBOARD_KEY or request.headers.get("X-Dashboard-Key", "") != DASHBOARD_KEY:
        return jsonify({"error": "unauthorized"}), 401

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

    # Authorisation is decided before configuration: whether a recipient is
    # allowed must not depend on whether Twilio happens to be wired up, or the
    # same request would 503 on a misconfigured box and 403 on a working one.
    if digits not in ALERT_RECIPIENTS:
        return jsonify({"error": "recipient not in allowlist"}), 403

    if not _TWILIO_READY:
        return jsonify({"error": "Twilio credentials not configured on server."}), 503

    node = db.get_node(node_id)
    if not node:
        return jsonify({"error": f"No telemetry received yet for node '{node_id}'"}), 404

    alert = node.get("alert_level", "NORMAL")
    result = send_alert(node_id, alert, node, to_override=digits)

    if result.get("success"):
        return jsonify({"sent": True, "sid": result.get("sid"),
                        "channel": result.get("channel"), "to": digits})
    if result.get("error") in ("cooldown", "global_budget"):
        return jsonify({"sent": False, "error": result["error"]}), 429
    return jsonify({
        "sent":        False,
        "error":       result.get("error"),
        "twilio_code": result.get("twilio_code"),
        "twilio_msg":  result.get("twilio_msg"),
    }), 500


@app.route("/api/test-alert", methods=["POST"])
def api_test_alert():
    """Fire a real message directly (bypasses ML and MQTT). X-Test-Key required."""
    if not TEST_SECRET:
        return jsonify({"error": "TEST_SECRET not configured on server."}), 503
    if request.headers.get("X-Test-Key", "") != TEST_SECRET:
        return jsonify({"error": "Invalid or missing X-Test-Key header"}), 401
    if not _TWILIO_READY:
        return jsonify({"error": "Twilio credentials not configured on server."}), 503

    body = request.get_json(silent=True) or {}
    node_id = str(body.get("node_id", "test-node")).strip() or "test-node"

    fake_data = {
        "node_label":        "TEST NODE",
        "water_level_m":     3.4,
        "rainfall_24h_mm":   180.0,
        "soil_moisture_pct": 95.0,
        "pressure_hpa":      996.0,
        "pressure_trend_hpa_per_hr": -2.8,
        "humidity_pct":      93.0,
        "reasons":           ["manual test of the Twilio path"],
    }
    result = send_alert(node_id, "ALERT", fake_data, force=True)
    if result.get("success"):
        return jsonify({"sent": True, "sid": result.get("sid"),
                        "channel": result.get("channel")})
    if result.get("error") == "global_budget":
        return jsonify({"sent": False, "error": "global_budget"}), 429
    return jsonify({
        "sent":        False,
        "error":       result.get("error"),
        "twilio_code": result.get("twilio_code"),
        "twilio_msg":  result.get("twilio_msg"),
    }), 500


# ── Outbound alerting ─────────────────────────────────────────────

def _global_send_count() -> int:
    now = time.monotonic()
    with _state_lock:
        while _global_sends and now - _global_sends[0] > 3600:
            _global_sends.popleft()
        return len(_global_sends)


def _global_budget_ok() -> bool:
    """Absolute cap on outbound messages per hour, across all nodes."""
    now = time.monotonic()
    with _state_lock:
        while _global_sends and now - _global_sends[0] > 3600:
            _global_sends.popleft()
        if len(_global_sends) >= GLOBAL_MAX_PER_HOUR:
            return False
        _global_sends.append(now)
        return True


def _refund_global_budget() -> None:
    with _state_lock:
        if _global_sends:
            _global_sends.pop()


def _trend_suffix(data: dict) -> str:
    """' (falling 2.8 hPa/hr)' when a trend is known, else ''."""
    t = data.get("pressure_trend_hpa_per_hr")
    if t is None:
        return ""
    return f" ({'falling' if t < 0 else 'rising'} {abs(float(t)):.1f} hPa/hr)"


def _alert_body(node_id: str, alert: str, data: dict) -> str:
    reasons = data.get("reasons") or []
    why = "\n".join(f"  - {r}" for r in reasons[:5])
    return (
        f"FLOOD ALERT\n"
        f"Node      : {data.get('node_label', node_id)} ({node_id})\n"
        f"Status    : {alert}\n"
        f"Water Lvl : {data.get('water_level_m', 'N/A')} m\n"
        f"Rainfall  : {data.get('rainfall_24h_mm', 'N/A')} mm\n"
        f"Soil Moist: {data.get('soil_moisture_pct', 'N/A')} %\n"
        f"Humidity  : {data.get('humidity_pct', 'N/A')} %\n"
        f"Pressure  : {data.get('pressure_hpa', 'N/A')} hPa"
        f"{_trend_suffix(data)}\n"
        + (f"Why       :\n{why}\n" if why else "")
        + f"Time      : {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )


def _twilio_send(*, from_: str, to: str, body: str) -> dict:
    if _twilio is None:
        return {"success": False, "error": "Twilio not ready"}
    if not _global_budget_ok():
        log.error("[alert] global budget exhausted (%d/h) – refusing to send",
                  GLOBAL_MAX_PER_HOUR)
        return {"success": False, "error": "global_budget"}
    try:
        msg = _twilio.messages.create(from_=from_, to=to, body=body)
        return {"success": True, "sid": msg.sid}
    except TwilioRestException as exc:
        _refund_global_budget()
        return {"success": False, "error": str(exc),
                "twilio_code": exc.code, "twilio_msg": exc.msg}
    except Exception as exc:  # noqa: BLE001
        _refund_global_budget()
        return {"success": False, "error": str(exc)}


def send_whatsapp(node_id: str, alert: str, data: dict, *,
                  force: bool = False, to_override: str | None = None) -> dict:
    if not _TWILIO_READY:
        return {"success": False, "error": "Twilio not ready"}

    now = time.monotonic()
    if not force:
        with _state_lock:
            last = node_alert_times.get(node_id, 0.0)
            if now - last < ALERT_COOLDOWN_SEC:
                log.warning("[WhatsApp][%s] Cooldown – %ds left", node_id,
                            int(ALERT_COOLDOWN_SEC - (now - last)))
                return {"success": False, "error": "cooldown"}
            node_alert_times[node_id] = now
    else:
        with _state_lock:
            node_alert_times[node_id] = now

    to_digits = to_override or (TWILIO_TO or "").replace("whatsapp:", "")
    result = _twilio_send(from_=TWILIO_FROM, to=f"whatsapp:{to_digits}",
                          body=_alert_body(node_id, alert, data))
    result["channel"] = "whatsapp"
    if not result.get("success") and result.get("error") != "global_budget":
        # Release the reservation so a genuine retry is not blocked.
        with _state_lock:
            node_alert_times.pop(node_id, None)
        log.error("[WhatsApp][%s] failed code=%s msg=%s", node_id,
                  result.get("twilio_code"), result.get("twilio_msg"))
    elif result.get("success"):
        log.info("[WhatsApp][%s] Sent – SID: %s", node_id, result["sid"])
    return result


def send_sms(node_id: str, alert: str, data: dict, *,
             to_override: str | None = None) -> dict:
    """Plain-SMS fallback on the same Twilio account (no 'whatsapp:' prefix)."""
    if not SMS_FALLBACK_FROM:
        return {"success": False, "error": "SMS fallback not configured"}
    to_digits = to_override or (TWILIO_TO or "").replace("whatsapp:", "")
    result = _twilio_send(from_=SMS_FALLBACK_FROM, to=to_digits,
                          body=_alert_body(node_id, alert, data))
    result["channel"] = "sms"
    if result.get("success"):
        log.info("[SMS][%s] Sent – SID: %s", node_id, result["sid"])
    else:
        log.error("[SMS][%s] failed: %s", node_id, result.get("error"))
    return result


def send_alert(node_id: str, alert: str, data: dict, *,
               force: bool = False, to_override: str | None = None) -> dict:
    """
    WhatsApp first, SMS second. WhatsApp delivery fails for boring reasons
    (sandbox not joined, 24h session window closed) that have nothing to do
    with whether the flood is real.
    """
    result = send_whatsapp(node_id, alert, data, force=force,
                           to_override=to_override)
    if (not result.get("success")
            and result.get("error") not in ("cooldown", "global_budget")
            and SMS_FALLBACK_FROM):
        log.info("[alert][%s] WhatsApp failed – falling back to SMS", node_id)
        result = send_sms(node_id, alert, data, to_override=to_override)

    attempt = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "node_id": node_id,
        "alert": alert,
        "channel": result.get("channel"),
        "success": bool(result.get("success")),
        "sid": result.get("sid"),
        "error": result.get("error"),
        "twilio_code": result.get("twilio_code"),
        "twilio_msg": result.get("twilio_msg"),
    }
    with _state_lock:
        _alert_log.append(attempt)
    return result


# ── Reading pipeline (shared by MQTT and /api/simulate) ───────────

def _process_reading(node_id: str, data: dict) -> tuple[dict, dict, dict]:
    data.setdefault("node_id", node_id)
    data.setdefault("node_label", node_id.replace("-", " ").title())

    result = predictor.predict(
        node_id, data, forecast_rain_24h_mm=weather.rain_24h(node_id))

    ts = datetime.now(timezone.utc).isoformat()
    state = {
        **data,
        "alert_level": result["status"],
        "raw_alert_level": result["raw_status"],
        "reasons": result["reasons"],
        "forecast_advisory": result["forecast_advisory"],
        "ood_features": result["ood_features"],
        "rise_m_per_min": result["rise_m_per_min"],
        "pressure_trend_hpa_per_hr": result["pressure_trend_hpa_per_hr"],
        "probabilities": result["probabilities"],
        "last_updated": ts,
    }
    db.update_node_state(node_id, state)
    db.insert_telemetry(node_id, data, result["status"])
    return result, state, {**data, "reasons": result["reasons"]}


# ── MQTT callbacks ────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc, props=None):
    global _mqtt_connected
    if rc == 0:
        _mqtt_connected = True
        log.info("[MQTT] Connected to %s:%d (client_id=%s)", BROKER, PORT, _MQTT_CLIENT_ID)
        client.subscribe(TOPIC_IN)
        client.subscribe(TOPIC_STATUS)
        log.info("[MQTT] Subscribed -> %s and %s", TOPIC_IN, TOPIC_STATUS)
    else:
        _mqtt_connected = False
        log.error("[MQTT] Connection refused – rc=%s", rc)


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    """
    paho's VERSION2 callback takes five arguments. The old four-argument
    signature raised TypeError inside paho on every disconnect, so this line
    never ran, `_mqtt_connected` never went false, and /api/health cheerfully
    reported healthy while the bridge was deaf.
    """
    global _mqtt_connected
    _mqtt_connected = False
    log.warning("[MQTT] Disconnected rc=%s – paho will reconnect.", reason_code)


def on_message(client, userdata, msg):
    topic = msg.topic
    try:
        raw = msg.payload.decode("utf-8")
    except UnicodeDecodeError:
        log.error("[MQTT] Non-UTF-8 payload on %s – skipped.", topic)
        return

    parts = topic.split("/")
    node_id = parts[-1] if len(parts) >= 3 else "unknown"

    # Node liveness: retained LWT on flood/status/<node_id>
    if topic.startswith("flood/status/"):
        state = raw.strip().lower()
        with _state_lock:
            node_link_state[node_id] = "offline" if state == "offline" else "online"
        log.info("[MQTT][%s] link state -> %s", node_id, node_link_state[node_id])
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("[MQTT][%s] Bad JSON: %s | raw=%s", node_id, exc, raw[:80])
        return

    try:
        result, state, alert_data = _process_reading(node_id, data)
    except Exception as exc:  # noqa: BLE001
        log.error("[pipeline][%s] failed: %s", node_id, exc)
        return

    alert = result["status"]
    log.info("[%s] %s  raw=%s reasons=%s", node_id, alert, result["raw_status"],
             result["reasons"])
    if result["ood_features"]:
        log.warning("[%s] out-of-distribution: %s", node_id, result["ood_features"])

    client.publish(
        TOPIC_OUT_FMT.format(node_id=node_id),
        json.dumps({
            "node_id":           node_id,
            "alert_level":       alert,
            "reasons":           result["reasons"],
            "forecast_advisory": result["forecast_advisory"],
            "probabilities":     result["probabilities"],
            "timestamp":         time.strftime("%H:%M:%S"),
        }),
        qos=1,
    )

    if alert == "ALERT":
        threading.Thread(target=send_alert, args=(node_id, alert, alert_data),
                         daemon=True).start()


# ── MQTT initialiser (idempotent) ─────────────────────────────────

def init_mqtt() -> None:
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

        if MQTT_USER:
            mc.username_pw_set(MQTT_USER, MQTT_PASS)
        if PORT == 8883:
            mc.tls_set()
            log.info("[MQTT] TLS enabled for port 8883")

        mc.connect(BROKER, PORT, keepalive=60)
        mc.loop_start()
        log.info("[MQTT] Background loop started. client_id=%s broker=%s:%d",
                 _MQTT_CLIENT_ID, BROKER, PORT)
    except Exception as exc:  # noqa: BLE001
        log.error("[MQTT] Initialization failed: %s", exc)
        _mqtt_started.clear()   # allow a retry if the broker was down at boot


db.init_db()
if os.getenv("DISABLE_MQTT", "").lower() not in ("1", "true", "yes"):
    init_mqtt()


def main() -> None:
    log.info("FloodSense MQTT Bridge – starting")
    log.info("  Broker   : %s:%d", BROKER, PORT)
    log.info("  Topics   : %s , %s", TOPIC_IN, TOPIC_STATUS)
    log.info("  API      : http://0.0.0.0:%d/api/nodes", API_PORT)
    log.info("  Demo     : %s", DEMO_MODE)
    log.info("  Allowlist: %d recipient(s)", len(ALERT_RECIPIENTS))
    log.info("  Layer 3  : %s", weather.configured_nodes() or "no NODE_COORDS set")
    try:
        app.run(host="0.0.0.0", port=API_PORT, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        log.info("Shutting down...")


if __name__ == "__main__":
    main()
