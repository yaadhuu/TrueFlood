"""
Backend API tests: the security gates, the health contract, node liveness,
metrics, and the regional-consensus note.

MQTT is disabled (DISABLE_MQTT=1) so the module can be imported without a
broker; every test drives the Flask app directly.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

DASHBOARD_KEY = "test-key-123"
ALLOWED_NUMBER = "+919999999999"


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    monkeypatch.setenv("DISABLE_MQTT", "1")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("DASHBOARD_KEY", DASHBOARD_KEY)
    monkeypatch.setenv("ALERT_RECIPIENTS", f"{ALLOWED_NUMBER},+911111111111")
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("NODE_STALE_SEC", "60")
    monkeypatch.setenv("GLOBAL_MAX_PER_HOUR", "3")
    # No Twilio credentials: nothing in these tests may reach the network.
    for var in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM",
                "TWILIO_TO", "SMS_FALLBACK_FROM", "NODE_COORDS"):
        monkeypatch.delenv(var, raising=False)

    # mqtt_bridge.py calls load_dotenv(..., override=True) at import time, so
    # a developer's local backend/.env (created by following .env.example)
    # would silently clobber every env var this fixture just set. Neutralize
    # it so the test's environment is the only source of truth.
    import dotenv  # noqa: PLC0415
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)

    import db  # noqa: PLC0415
    importlib.reload(db)
    import mqtt_bridge  # noqa: PLC0415
    importlib.reload(mqtt_bridge)
    mqtt_bridge.app.config["TESTING"] = True
    return mqtt_bridge


@pytest.fixture
def client(bridge):
    return bridge.app.test_client()


STORM = {
    "water_level_m": 3.4, "rainfall_24h_mm": 180.0, "soil_moisture_pct": 95.0,
    "flow_velocity_ms": 8.0, "turbidity_ntu": 900.0,
}
CALM = {
    "water_level_m": 0.4, "rainfall_24h_mm": 3.0, "soil_moisture_pct": 45.0,
    "flow_velocity_ms": 0.8, "turbidity_ntu": 90.0,
}


# ── security ─────────────────────────────────────────────────────

def test_alert_send_requires_dashboard_key(client):
    r = client.post("/api/alert/send",
                    json={"node_id": "x", "to": ALLOWED_NUMBER})
    assert r.status_code == 401


def test_alert_send_rejects_number_not_on_allowlist(client):
    # No Twilio credentials are set: the allowlist must still decide first,
    # otherwise the same request 503s on a misconfigured box and 403s on a
    # working one.
    r = client.post("/api/alert/send",
                    json={"node_id": "x", "to": "+15550001111"},
                    headers={"X-Dashboard-Key": DASHBOARD_KEY})
    assert r.status_code == 403
    assert "allowlist" in r.get_json()["error"]


def test_alert_send_never_returns_200_without_credentials(client):
    for headers in ({}, {"X-Dashboard-Key": "wrong"}):
        r = client.post("/api/alert/send",
                        json={"node_id": "x", "to": ALLOWED_NUMBER},
                        headers=headers)
        assert r.status_code != 200


def test_test_alert_requires_secret(client):
    r = client.post("/api/test-alert", json={})
    assert r.status_code in (401, 503)


def test_global_send_budget_caps_outbound(bridge):
    """The global budget must bite even when every per-node cooldown allows."""
    assert bridge.GLOBAL_MAX_PER_HOUR == 3
    assert [bridge._global_budget_ok() for _ in range(4)] == [True, True, True, False]


# ── health / metrics ─────────────────────────────────────────────

def test_health_is_503_when_mqtt_disconnected(client, bridge):
    bridge._mqtt_connected = False
    r = client.get("/api/health")
    assert r.status_code == 503
    assert r.get_json()["status"] == "degraded"


def test_health_is_200_when_connected(client, bridge):
    bridge._mqtt_connected = True
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["mqtt_connected"] is True


def test_metrics_is_prometheus_text(client, bridge):
    bridge._mqtt_connected = True
    client.post("/api/simulate", json={"node_id": "node-1", **CALM})
    r = client.get("/api/metrics")
    assert r.status_code == 200
    assert r.headers["Content-Type"].startswith("text/plain")
    body = r.get_data(as_text=True)
    assert "floodsense_mqtt_connected 1" in body
    assert 'floodsense_node_alert_level{node_id="node-1"}' in body
    for line in body.strip().splitlines():
        assert line.startswith("#") or len(line.split(" ")) == 2


# ── pipeline / liveness / regional ───────────────────────────────

def test_simulate_returns_layer1_reasons(client):
    r = client.post("/api/simulate", json={"node_id": "node-1", **STORM})
    assert r.status_code == 200
    body = r.get_json()
    assert body["alert_level"] == "ALERT"
    assert len(body["reasons"]) >= 2


def test_nodes_report_staleness(client, bridge):
    client.post("/api/simulate", json={"node_id": "node-1", **CALM})
    body = client.get("/api/nodes").get_json()
    node = body["nodes"][0]
    assert node["stale"] is False
    assert node["seconds_since_update"] is not None

    # An LWT "offline" marks the node dead immediately, without waiting out
    # NODE_STALE_SEC.
    bridge.node_link_state["node-1"] = "offline"
    body = client.get("/api/nodes").get_json()
    assert body["nodes"][0]["stale"] is True
    assert body["stale_count"] == 1


def test_regional_note_appears_with_two_elevated_nodes(client):
    assert client.get("/api/nodes").get_json()["regional_note"] is None
    client.post("/api/simulate", json={"node_id": "node-1", **STORM})
    client.post("/api/simulate", json={"node_id": "node-2", **STORM})
    note = client.get("/api/nodes").get_json()["regional_note"]
    assert note is not None and "2 nodes elevated" in note


def test_stale_nodes_are_excluded_from_regional_consensus(client, bridge):
    client.post("/api/simulate", json={"node_id": "node-1", **STORM})
    client.post("/api/simulate", json={"node_id": "node-2", **STORM})
    bridge.node_link_state["node-2"] = "offline"
    assert client.get("/api/nodes").get_json()["regional_note"] is None


def test_simulate_validates_payload(client):
    r = client.post("/api/simulate", json={"node_id": "n", "water_level_m": "abc"})
    assert r.status_code == 422
