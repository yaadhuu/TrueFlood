"""
The reading pipeline and node health - shared by MQTT and the REST API.

    process_reading(): validated reading -> predictor (3 layers) -> SQLite
    liveness:          is a node still sending? (timeout or MQTT Last Will)
    regional_check():  are several nodes elevated at once?
"""

import threading
from datetime import datetime, timezone

import config
import db
import weather
from predict import predictor

REQUIRED_FIELDS = ["water_level_m", "rainfall_24h_mm", "soil_moisture_pct"]
# Optional: a node whose pressure or humidity sensor failed still has a usable
# water level, so its readings must not be rejected.
OPTIONAL_FIELDS = ["pressure_hpa", "pressure_trend_hpa_per_hr",
                   "humidity_pct", "temperature_c"]

_lock = threading.RLock()
node_link_state: dict[str, str] = {}   # node_id -> "online" | "offline" (MQTT Last Will)


# ── Validation ────────────────────────────────────────────────────

def validate_reading(data: dict) -> tuple[dict, str | None]:
    """Return (numeric fields, None) or ({}, error message)."""
    clean = {}
    for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
        raw = data.get(field)
        if raw is None:
            if field in REQUIRED_FIELDS:
                return {}, f"Missing required field: {field}"
            continue
        try:
            clean[field] = float(raw)
        except (TypeError, ValueError):
            return {}, f"Field '{field}' must be numeric, got: {raw!r}"
    return clean, None


# ── Pipeline ──────────────────────────────────────────────────────

def process_reading(node_id: str, data: dict) -> tuple[dict, dict, dict]:
    """Run one reading through the predictor and store it.

    Returns (prediction result, stored node state, data for the alert message).
    """
    data.setdefault("node_id", node_id)
    data.setdefault("node_label", node_id.replace("-", " ").title())

    result = predictor.predict(node_id, data,
                               forecast_rain_24h_mm=weather.rain_24h(node_id))

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
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }
    db.update_node_state(node_id, state)
    db.insert_telemetry(node_id, data, result["status"])
    return result, state, {**data, "reasons": result["reasons"]}


# ── Liveness ──────────────────────────────────────────────────────

def set_link_state(node_id: str, state: str) -> None:
    with _lock:
        node_link_state[node_id] = "offline" if state == "offline" else "online"


def seconds_since_update(node: dict) -> float | None:
    try:
        last = datetime.fromisoformat(node.get("last_updated") or "")
    except (TypeError, ValueError):
        return None
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - last).total_seconds()


def is_stale(node: dict) -> bool:
    """A dead sensor looks exactly like a healthy one unless something checks.

    Stale = the broker delivered this node's Last Will ("offline"), or no
    reading arrived within NODE_STALE_SEC.
    """
    if node_link_state.get(node.get("node_id")) == "offline":
        return True
    age = seconds_since_update(node)
    return age is None or age > config.NODE_STALE_SEC


def annotate(node: dict) -> dict:
    """Add liveness fields to a stored node record before returning it."""
    age = seconds_since_update(node)
    return {
        **node,
        "stale": is_stale(node),
        "seconds_since_update": None if age is None else round(age, 1),
        "link_state": node_link_state.get(node.get("node_id"), "unknown"),
    }


def regional_check(nodes: list[dict]) -> str | None:
    """One sensor spiking can be a glitch; two live nodes elevated at once is a
    regional event. Shown as context only - it never changes a node's alert level."""
    elevated = [n for n in nodes
                if n.get("alert_level") in ("WATCH", "ALERT") and not n.get("stale")]
    if len(elevated) < 2:
        return None
    names = ", ".join(sorted(n.get("node_id", "?") for n in elevated))
    return f"REGIONAL WATCH: {len(elevated)} nodes elevated simultaneously ({names})"
