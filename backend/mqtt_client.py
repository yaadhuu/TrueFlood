"""
MQTT ingest: receive sensor readings, run them through the pipeline, publish
the result, and trigger an alert when a node reaches ALERT.

Topics
    flood/sensor/<node_id>   in   JSON reading from an ESP32 node
    flood/status/<node_id>   in   "online" / "offline" (retained Last Will)
    flood/alert/<node_id>    out  the decision for that reading

paho runs its network loop in a background thread (loop_start), so these
callbacks run alongside Flask's request threads.
"""

import json
import threading
import time
import uuid

import paho.mqtt.client as mqtt

import alerts
import config
import nodes
from config import log

connected = False                      # read by /api/health and /api/metrics
_started = threading.Event()
CLIENT_ID = f"flood-bridge-{uuid.uuid4().hex[:8]}"


def on_connect(client, userdata, flags, rc, props=None):
    global connected
    connected = rc == 0
    if connected:
        client.subscribe(config.TOPIC_IN)
        client.subscribe(config.TOPIC_STATUS)
        log.info("[MQTT] connected to %s:%d, subscribed to %s and %s",
                 config.BROKER, config.MQTT_PORT, config.TOPIC_IN, config.TOPIC_STATUS)
    else:
        log.error("[MQTT] connection refused, rc=%s", rc)


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    # paho's VERSION2 API passes five arguments here; a wrong signature makes
    # paho swallow the call, and /api/health would keep reporting "connected".
    global connected
    connected = False
    log.warning("[MQTT] disconnected rc=%s - paho will reconnect", reason_code)


def on_message(client, userdata, msg):
    try:
        raw = msg.payload.decode("utf-8")
    except UnicodeDecodeError:
        log.error("[MQTT] non-UTF-8 payload on %s - skipped", msg.topic)
        return

    parts = msg.topic.split("/")
    node_id = parts[-1] if len(parts) >= 3 else "unknown"

    if msg.topic.startswith("flood/status/"):
        nodes.set_link_state(node_id, raw.strip().lower())
        return

    try:
        data = json.loads(raw)
        result, _state, alert_data = nodes.process_reading(node_id, data)
    except json.JSONDecodeError as exc:
        log.error("[MQTT][%s] bad JSON: %s", node_id, exc)
        return
    except Exception as exc:  # noqa: BLE001 - one bad packet must not kill the loop
        log.error("[pipeline][%s] failed: %s", node_id, exc)
        return

    log.info("[%s] %s reasons=%s", node_id, result["status"], result["reasons"])
    client.publish(
        config.TOPIC_OUT_FMT.format(node_id=node_id),
        json.dumps({
            "node_id": node_id,
            "alert_level": result["status"],
            "reasons": result["reasons"],
            "forecast_advisory": result["forecast_advisory"],
            "probabilities": result["probabilities"],
            "timestamp": time.strftime("%H:%M:%S"),
        }),
        qos=1,
    )
    if result["status"] == "ALERT":
        alerts.send_alert_in_background(node_id, result["status"], alert_data)


def start() -> None:
    """Connect and start the background loop. Safe to call more than once."""
    if _started.is_set():
        return
    _started.set()
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                             client_id=CLIENT_ID, clean_session=True)
        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message
        client.reconnect_delay_set(min_delay=2, max_delay=120)
        if config.MQTT_USER:
            client.username_pw_set(config.MQTT_USER, config.MQTT_PASS)
        if config.MQTT_PORT == 8883:
            client.tls_set()
        client.connect(config.BROKER, config.MQTT_PORT, keepalive=60)
        client.loop_start()
    except Exception as exc:  # noqa: BLE001
        log.error("[MQTT] could not start: %s", exc)
        _started.clear()      # allow a retry if the broker was down at boot
