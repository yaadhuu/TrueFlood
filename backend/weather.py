"""
Layer 3: weather fusion.

A sensor-only flood system is purely reactive - it learns about a storm when
the water is already rising.  Free public forecast data makes it partly
anticipatory: if the ground at a node is already near saturation and 80 mm of
rain is forecast for tomorrow, that is actionable *today*, and no sensor on
the pole can tell you.

Source: Open-Meteo forecast API (free, no key, distinct from the *archive* API
that ml_pipeline/fetch_openmeteo.py uses for the real-data track).

    GET https://api.open-meteo.com/v1/forecast
        ?latitude=&longitude=&daily=precipitation_sum&forecast_days=2&timezone=UTC

Polled at most once per POLL_INTERVAL_SEC per location and cached in memory -
never once per MQTT packet.  At 5 s packets from 4 nodes that would be 69k
requests a day for data that changes hourly.

Node coordinates come from the NODE_COORDS env var:

    NODE_COORDS=node-1:43.027,-91.173;node-2:19.076,72.877

Nodes without coordinates simply get no Layer-3 input; the rest of the system
is unaffected.  Every network failure degrades to "no forecast" rather than
raising - a weather API being down must never stop a flood alarm.
"""

from __future__ import annotations

import logging
import os
import threading
import time

import requests

log = logging.getLogger("floodsense.weather")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
POLL_INTERVAL_SEC = int(os.getenv("WEATHER_POLL_SEC", "3600"))
REQUEST_TIMEOUT_SEC = int(os.getenv("WEATHER_TIMEOUT_SEC", "10"))

_lock = threading.RLock()
_cache: dict[str, dict] = {}          # node_id -> {fetched_at, rain_24h, rain_48h}


def parse_node_coords(raw: str | None = None) -> dict[str, tuple[float, float]]:
    """Parse NODE_COORDS. Malformed entries are skipped with a warning."""
    raw = os.getenv("NODE_COORDS", "") if raw is None else raw
    out: dict[str, tuple[float, float]] = {}
    for entry in raw.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        try:
            node_id, coords = entry.split(":", 1)
            lat_s, lon_s = coords.split(",")
            out[node_id.strip()] = (float(lat_s), float(lon_s))
        except (ValueError, TypeError):
            log.warning("[weather] ignoring malformed NODE_COORDS entry: %r", entry)
    return out


NODE_COORDS = parse_node_coords()


def _fetch(lat: float, lon: float) -> dict | None:
    try:
        r = requests.get(
            FORECAST_URL,
            params={"latitude": lat, "longitude": lon,
                    "daily": "precipitation_sum", "forecast_days": 2,
                    "timezone": "UTC"},
            timeout=REQUEST_TIMEOUT_SEC,
        )
        r.raise_for_status()
        series = r.json()["daily"]["precipitation_sum"]
        vals = [float(v) for v in series if v is not None]
    except Exception as exc:  # noqa: BLE001 - never let weather break alarms
        log.warning("[weather] fetch failed for %s,%s: %s", lat, lon, exc)
        return None
    if not vals:
        return None
    return {
        "forecast_rain_24h_mm": vals[0],
        "forecast_rain_48h_mm": float(sum(vals[:2])),
    }


def get_forecast(node_id: str, *, force: bool = False) -> dict | None:
    """
    Cached forecast for a node, or None when unknown/unavailable.

    Returns: {"forecast_rain_24h_mm", "forecast_rain_48h_mm", "age_sec", "stale"}
    """
    coords = NODE_COORDS.get(node_id)
    if coords is None:
        return None

    now = time.monotonic()
    with _lock:
        cached = _cache.get(node_id)
        fresh = cached and (now - cached["fetched_at"]) < POLL_INTERVAL_SEC
        if fresh and not force:
            return {**cached["data"], "age_sec": int(now - cached["fetched_at"]),
                    "stale": False}

    data = _fetch(*coords)          # network call outside the lock
    if data is None:
        with _lock:
            cached = _cache.get(node_id)
        if cached:  # serve the stale value rather than nothing
            return {**cached["data"],
                    "age_sec": int(now - cached["fetched_at"]), "stale": True}
        return None

    with _lock:
        _cache[node_id] = {"fetched_at": now, "data": data}
    return {**data, "age_sec": 0, "stale": False}


def rain_24h(node_id: str) -> float | None:
    """Convenience accessor for SafetyRules. None when no forecast is known."""
    fc = get_forecast(node_id)
    return None if fc is None else fc["forecast_rain_24h_mm"]


def configured_nodes() -> list[str]:
    return sorted(NODE_COORDS)
