"""
TrueFlood backend - entry point and REST API.

    MQTT readings ──► mqtt_client ──► nodes.process_reading ──► predict (3 layers)
                                                │                    │
                                                ▼                    ▼
                                             SQLite (db)        alerts (Twilio)
                                                │
    dashboard ◄── this file's REST API ◄────────┘

Run locally:   python backend/app.py
Production:    gunicorn --chdir backend app:app --workers 1
(--workers 1 is required: the MQTT connection and the predictor's per-node
memory live inside one process. Two workers would mean two subscriptions and
duplicate alerts, so startup refuses anything else.)
"""

from datetime import datetime, timezone

from flask import Flask, jsonify, request
from flask_cors import CORS

import alerts
import config
import db
import mqtt_client
import nodes
import weather
from config import log
from predict import predictor

if config.WEB_CONCURRENCY != 1:
    raise RuntimeError(f"WEB_CONCURRENCY={config.WEB_CONCURRENCY}; "
                       "this app must run with exactly one worker.")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": config.ALLOWED_ORIGINS}})


def _annotated_nodes() -> list[dict]:
    return [nodes.annotate(n) for n in db.get_all_nodes()]


def _twilio_error(result: dict, status: int = 500):
    return jsonify({"sent": False, "error": result.get("error"),
                    "twilio_code": result.get("twilio_code"),
                    "twilio_msg": result.get("twilio_msg")}), status


# ── Read endpoints (dashboard) ────────────────────────────────────

@app.get("/api/nodes")
def api_nodes():
    snapshot = _annotated_nodes()
    return jsonify({
        "nodes": snapshot,
        "node_count": len(snapshot),
        "stale_count": sum(1 for n in snapshot if n["stale"]),
        "regional_note": nodes.regional_check(snapshot),
        "server_time": datetime.now(timezone.utc).isoformat(),
    })


@app.get("/api/nodes/<node_id>")
def api_node(node_id):
    current = db.get_node(node_id)
    if not current:
        return jsonify({"error": "node not found"}), 404
    return jsonify({"current": nodes.annotate(current),
                    "history": db.get_node_history(node_id, limit=config.MAX_HISTORY)})


@app.get("/api/history/<node_id>")
def api_history(node_id):
    history = db.get_node_history(node_id, limit=config.MAX_HISTORY)
    if not history:
        return jsonify({"error": "node not found"}), 404
    return jsonify({"node_id": node_id, "history": history})


@app.get("/api/weather/<node_id>")
def api_weather(node_id):
    forecast = weather.get_forecast(node_id)
    if forecast is None:
        return jsonify({"error": "no forecast for this node",
                        "hint": "set NODE_COORDS=node-1:lat,lon;... to enable Layer 3",
                        "configured_nodes": weather.configured_nodes()}), 404
    return jsonify({"node_id": node_id, **forecast})


@app.get("/api/alerts/log")
def api_alerts_log():
    attempts = list(alerts.alert_log)
    return jsonify({"attempts": attempts, "count": len(attempts)})


@app.get("/api/version")
def api_version():
    return jsonify({"version": "3.0.0", "build": "three-layer"})


# ── Operations ────────────────────────────────────────────────────

@app.get("/api/health")
def api_health():
    snapshot = _annotated_nodes()
    degraded = not mqtt_client.connected
    body = {
        "status": "degraded" if degraded else "ok",
        "nodes_total": len(snapshot),
        "nodes_online": sum(1 for n in snapshot if not n["stale"]),
        "nodes_stale": sum(1 for n in snapshot if n["stale"]),
        "broker": config.BROKER,
        "mqtt_port": config.MQTT_PORT,
        "mqtt_connected": mqtt_client.connected,
        "twilio_ready": config.TWILIO_READY,
        "sms_fallback": bool(config.SMS_FALLBACK_FROM),
        "model_loaded": predictor.model is not None,
        "demo_mode": config.DEMO_MODE,
        "uptime_ts": datetime.now(timezone.utc).isoformat(),
    }
    # 503 when the broker is unreachable, so a hosting platform's health check
    # restarts a bridge that can no longer hear its sensors.
    return jsonify(body), (503 if degraded else 200)


@app.get("/api/metrics")
def api_metrics():
    """Prometheus text format, written by hand (no extra dependency)."""
    snapshot = _annotated_nodes()

    def gauge(name, help_text, value):
        return [f"# HELP {name} {help_text}", f"# TYPE {name} gauge", f"{name} {value}"]

    lines = (
        gauge("floodsense_nodes_total", "Registered nodes", len(snapshot))
        + gauge("floodsense_nodes_stale", "Nodes with no fresh telemetry",
                sum(1 for n in snapshot if n["stale"]))
        + gauge("floodsense_mqtt_connected", "MQTT broker connection state",
                int(mqtt_client.connected))
        + gauge("floodsense_model_loaded", "Layer 2 model availability",
                int(predictor.model is not None))
        + gauge("floodsense_alert_sends_last_hour", "Outbound alerts in the last hour",
                alerts.sends_last_hour())
        + ["# HELP floodsense_node_alert_level 0=NORMAL 1=WATCH 2=ALERT -1=unknown",
           "# TYPE floodsense_node_alert_level gauge"]
    )
    for n in snapshot:
        level = {"NORMAL": 0, "WATCH": 1, "ALERT": 2}.get(n.get("alert_level"), -1)
        node_id = str(n.get("node_id", "unknown")).replace('"', "")
        age = n.get("seconds_since_update")
        lines.append(f'floodsense_node_alert_level{{node_id="{node_id}"}} {level}')
        lines.append(f'floodsense_node_seconds_since_update{{node_id="{node_id}"}} '
                     f'{age if age is not None else -1}')
    return "\n".join(lines) + "\n", 200, {"Content-Type": "text/plain; version=0.0.4"}


# ── Write endpoints ───────────────────────────────────────────────

@app.post("/api/simulate")
def api_simulate():
    """Push a fake reading through the full pipeline (DEMO_MODE only)."""
    if not config.DEMO_MODE:
        return jsonify({"error": "Simulation endpoint is disabled. "
                                 "Set DEMO_MODE=true to enable."}), 403
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400
    node_id = str(body.get("node_id", "sim-node")).strip()
    if not node_id:
        return jsonify({"error": "node_id must not be empty"}), 422
    numbers, err = nodes.validate_reading(body)
    if err:
        return jsonify({"error": err}), 422

    result, state, alert_data = nodes.process_reading(node_id, {**body, **numbers})
    if result["status"] == "ALERT":
        alerts.send_alert_in_background(node_id, result["status"], alert_data)
    return jsonify({
        "node_id": node_id,
        "alert_level": result["status"],
        "reasons": result["reasons"],
        "forecast_advisory": result["forecast_advisory"],
        "ood_features": result["ood_features"],
        "probabilities": result["probabilities"],
        "timestamp": state["last_updated"],
    })


@app.post("/api/alert/send")
def api_alert_send():
    """Dashboard "Send Alert" button. This spends real money, so it has three gates:
    a shared key (401), a recipient allowlist (403), and cooldown + hourly budget (429)."""
    if (not config.DASHBOARD_KEY
            or request.headers.get("X-Dashboard-Key", "") != config.DASHBOARD_KEY):
        return jsonify({"error": "unauthorized"}), 401

    body = request.get_json(silent=True) or {}
    node_id = str(body.get("node_id", "")).strip()
    number = str(body.get("to", "")).strip().replace(" ", "").replace("-", "")
    if not node_id:
        return jsonify({"error": "node_id is required"}), 422
    if not number:
        return jsonify({"error": "to (phone number) is required"}), 422
    number = "+" + number.lstrip("+")
    if not number[1:].isdigit() or len(number) < 8:
        return jsonify({"error": "to must be a valid phone number, e.g. +919876543210"}), 422

    # The allowlist is checked before configuration, so the same request gets
    # the same answer whether or not Twilio is set up on this server.
    if number not in config.ALERT_RECIPIENTS:
        return jsonify({"error": "recipient not in allowlist"}), 403
    if not config.TWILIO_READY:
        return jsonify({"error": "Twilio credentials not configured on server."}), 503
    node = db.get_node(node_id)
    if not node:
        return jsonify({"error": f"No telemetry received yet for node '{node_id}'"}), 404

    result = alerts.send_alert(node_id, node.get("alert_level", "NORMAL"), node,
                               to_override=number)
    if result.get("success"):
        return jsonify({"sent": True, "sid": result.get("sid"),
                        "channel": result.get("channel"), "to": number})
    if result.get("error") in ("cooldown", "global_budget"):
        return jsonify({"sent": False, "error": result["error"]}), 429
    return _twilio_error(result)


@app.post("/api/test-alert")
def api_test_alert():
    """Send a real test message, skipping ML and MQTT. Needs the X-Test-Key header."""
    if not config.TEST_SECRET:
        return jsonify({"error": "TEST_SECRET not configured on server."}), 503
    if request.headers.get("X-Test-Key", "") != config.TEST_SECRET:
        return jsonify({"error": "Invalid or missing X-Test-Key header"}), 401
    if not config.TWILIO_READY:
        return jsonify({"error": "Twilio credentials not configured on server."}), 503

    node_id = str((request.get_json(silent=True) or {}).get("node_id", "")).strip() or "test-node"
    sample = {
        "node_label": "TEST NODE", "water_level_m": 3.4, "rainfall_24h_mm": 180.0,
        "soil_moisture_pct": 95.0, "pressure_hpa": 996.0,
        "pressure_trend_hpa_per_hr": -2.8, "humidity_pct": 93.0,
        "reasons": ["manual test of the Twilio path"],
    }
    result = alerts.send_alert(node_id, "ALERT", sample, force=True)
    if result.get("success"):
        return jsonify({"sent": True, "sid": result.get("sid"),
                        "channel": result.get("channel")})
    if result.get("error") == "global_budget":
        return jsonify({"sent": False, "error": "global_budget"}), 429
    return _twilio_error(result)


# ── Startup ───────────────────────────────────────────────────────

db.init_db()
if not config.DISABLE_MQTT:
    mqtt_client.start()


def main() -> None:
    log.info("TrueFlood backend - broker %s:%d, API on :%d, demo=%s, layer 3: %s",
             config.BROKER, config.MQTT_PORT, config.API_PORT, config.DEMO_MODE,
             weather.configured_nodes() or "no NODE_COORDS set")
    app.run(host="0.0.0.0", port=config.API_PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
