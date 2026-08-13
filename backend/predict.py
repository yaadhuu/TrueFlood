"""
FloodSense inference engine - three layers.

    Layer 1  SAFETY RULES      deterministic, auditable, drives every alarm
    Layer 2  ML FORECAST       advisory only, never drives an alarm
    Layer 3  WEATHER FUSION    advisory context folded into Layer 1's reasons

Why the split
-------------
The ML model in this project is trained on synthetic data whose label is a
threshold rule over its own features (see MODEL_CARD.md), and it loses to a
persistence baseline on a proper forecast task.  A model in that state must
not be wired to a siren.  Layer 1 is a small set of named constants that a
domain expert can read, argue with, and audit; it is what actually fires
alerts.  Layer 2 is reported alongside, clearly labelled as advisory.

Thread safety: all mutable per-node state lives behind `_LOCK` (an RLock),
because the MQTT bridge calls `predict()` from paho's network thread while
Flask serves requests from worker threads.

Degradation: if `flood_model.joblib` is missing or unreadable, Layer 2 is
skipped and Layer 1 alone decides.  It fails *loud*, not safe-silent - a
missing model must never turn a real flood into a NORMAL reading.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import pandas as pd

def _env_path(name: str, default: str) -> str:
    """
    os.getenv(name, default) only falls back when the var is absent — a var
    present but set to "" (which is exactly what `.env.example`'s "leave
    blank for default" documents) returns "" verbatim, silently breaking
    model loading. Blank is treated the same as unset.
    """
    val = os.getenv(name)
    return val if val else default


MODEL_PATH = _env_path(
    "MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml_pipeline",
                 "flood_model.joblib"),
)
CARD_PATH = _env_path(
    "MODEL_CARD_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml_pipeline",
                 "model_card.json"),
)

LABEL_NAMES = ["NORMAL", "WATCH", "ALERT"]

# One live packet arrives every ~5 s (firmware PUBLISH_MS).  The model was
# trained on daily rows.  Feeding raw 5-second packets into a feature named
# `rainfall_72h_mm` is the training/serving skew that made the original
# deployment meaningless.  Layer 2 therefore aggregates the packet buffer into
# RESAMPLE_WINDOW_SEC-wide bins and treats one bin as one training step, so
# "72h" means three bins in both places.  The absolute duration differs from
# training (a demo cannot wait three days); what matters is that the *shape* of
# the feature - a sum over three consecutive steps - is the same.
RESAMPLE_WINDOW_SEC = float(os.getenv("RESAMPLE_WINDOW_SEC", "60"))
BUFFER_SECONDS = RESAMPLE_WINDOW_SEC * 6

# A barometric trend (hPa/HOUR) is a slow, minutes-to-hours quantity.
# Extrapolating it from a single ~5s packet-to-packet delta (like
# rise_m_per_min does for water level) amplifies any noise ~700x - a 0.1 hPa
# ADC jitter over 5s alone reads as 72 hPa/h, blowing straight past the
# storm-alert threshold on nothing. The trend is instead computed against a
# rolling baseline reading that only refreshes once per window, damping that
# to ~60x - fine for a Wokwi slider (smooth, no sensor noise) but still worth
# raising (e.g. to 300-600s) on real BMP180 hardware if its own measurement
# noise (~0.03-0.3 hPa depending on oversampling) starts tripping the rule.
PRESSURE_TREND_WINDOW_SEC = float(os.getenv("PRESSURE_TREND_WINDOW_SEC", "60"))

_LOCK = threading.RLock()


class SafetyRules:
    """
    Layer 1: deterministic threshold engine with hysteresis.

    Scale note (Phase 1): these constants were calibrated against the 0-4 m
    Wokwi water-level range (`TANK_DEPTH_M = 4.0f` in firmware/sketch.ino),
    independently of the training dataset.  `ml_pipeline/rescale_dataset.py`
    now maps the training data onto that same 0-4 m range and re-derives its
    labels, and it prints a check confirming that `WL_ALERT_M = 3.0` sits
    inside the rescaled ALERT band (breakpoint 1.668 m).  The two layers
    therefore agree on what a metre means.  They still disagree on where to
    draw the line - deliberately: Layer 2's breakpoint is inherited from a
    synthetic labelling rule, Layer 1's is a physical judgement about a 4 m
    channel, and Layer 1 is the one that fires alarms.

    Hysteresis: a level must persist for ESCALATE_AFTER_SEC before the node
    escalates, and must stay clear for DEESCALATE_AFTER_SEC before it stands
    down.  Corroboration bypass: >= CORROBORATION_MIN independent trigger
    reasons escalate immediately - a real flood should not have to wait out a
    noise filter.
    """

    # Water level (metres, 0 - TANK_DEPTH_M = 4.0)
    WL_WATCH_M = 2.0
    WL_ALERT_M = 3.0

    # Rain accumulation (mm over the reported 24 h window, 0-200)
    RAIN_WATCH_MM = 90.0
    RAIN_ALERT_MM = 180.0

    # Soil moisture (%). At this saturation the ground has effectively zero
    # absorption capacity left, so rain runs straight off into the channel.
    SOIL_SATURATED_PCT = 98.0

    # Rate of rise (m per minute) - the single most predictive flash-flood
    # signal. Retained through the hardware refresh: it is derived from the
    # water-level sensor, not from either of the sensors that were removed.
    RISE_WATCH_M_PER_MIN = 0.10
    RISE_ALERT_M_PER_MIN = 0.25

    # Barometric pressure trend (hPa/hour, BMP180), negative = falling.
    # Meteorological convention: about -1 hPa/hr is a rapid fall signalling an
    # approaching system, and -2.5 hPa/hr is the severe band associated with
    # flash-flood-producing systems. This replaced the flow-velocity and
    # turbidity rules when those sensors left the hardware roster, and is a
    # genuinely better predictor than either: it is the only Layer-1 signal
    # that can fire BEFORE the water level has moved at all.
    PRESSURE_FALL_WARNING_HPA_HR = -1.0
    PRESSURE_FALL_SEVERE_HPA_HR = -2.5

    # Relative humidity (%) and rain floor for the storm-front compound rule.
    HUMIDITY_STORM_PCT = 85.0
    RAIN_STARTED_MM = 5.0

    # Hysteresis / debounce
    ESCALATE_AFTER_SEC = 10.0
    DEESCALATE_AFTER_SEC = 600.0
    CORROBORATION_MIN = 2

    @classmethod
    def evaluate(
        cls,
        data: dict,
        *,
        rise_m_per_min: float = 0.0,
        pressure_trend_hpa_per_hr: float | None = None,
        forecast_rain_24h_mm: float | None = None,
    ) -> tuple[int, list[str]]:
        """Return (level 0/1/2, human-readable reasons). Pure function."""
        wl = float(data.get("water_level_m", 0.0) or 0.0)
        rain = float(data.get("rainfall_24h_mm", 0.0) or 0.0)
        soil = float(data.get("soil_moisture_pct", 0.0) or 0.0)
        humidity = float(data.get("humidity_pct", 0.0) or 0.0)

        level = 0
        reasons: list[str] = []

        if wl >= cls.WL_ALERT_M:
            level = max(level, 2)
            reasons.append(f"water level {wl:.2f} m >= {cls.WL_ALERT_M} m")
        elif wl >= cls.WL_WATCH_M:
            level = max(level, 1)
            reasons.append(f"water level {wl:.2f} m >= {cls.WL_WATCH_M} m")

        if rain >= cls.RAIN_WATCH_MM:
            level = max(level, 1)
            reasons.append(f"rainfall {rain:.0f} mm/24h >= {cls.RAIN_WATCH_MM:.0f} mm")

        if rise_m_per_min >= cls.RISE_ALERT_M_PER_MIN:
            level = max(level, 2)
            reasons.append(f"water rising {rise_m_per_min:.2f} m/min "
                           f">= {cls.RISE_ALERT_M_PER_MIN} m/min")
        elif rise_m_per_min >= cls.RISE_WATCH_M_PER_MIN:
            level = max(level, 1)
            reasons.append(f"water rising {rise_m_per_min:.2f} m/min "
                           f">= {cls.RISE_WATCH_M_PER_MIN} m/min")

        # Falling barometer: an anticipatory signal from the node's own BMP180,
        # not an external forecast. Only a DROP matters - rising or steady
        # pressure is fine weather and never escalates.
        if pressure_trend_hpa_per_hr is not None:
            trend = pressure_trend_hpa_per_hr
            if trend <= cls.PRESSURE_FALL_SEVERE_HPA_HR:
                # Severe fall stands alone: this band is associated with the
                # systems that actually produce flash floods.
                level = max(level, 2)
                reasons.append(f"pressure falling {abs(trend):.1f} hPa/hr "
                               "(severe, flash-flood band)")
            elif (trend <= cls.PRESSURE_FALL_WARNING_HPA_HR
                    and humidity > cls.HUMIDITY_STORM_PCT
                    and rain > cls.RAIN_STARTED_MM):
                # The predictive branch. A moderate fall on its own is just
                # weather; a moderate fall with saturated air AND rain already
                # falling is a storm front arriving, and this fires before the
                # water level has moved at all.
                level = max(level, 1)
                reasons.append(f"pressure falling {abs(trend):.1f} hPa/hr "
                               f"at {humidity:.0f}% RH")

        # Compound: high water, peak rain and ground with no absorption
        # capacity left. Each alone is survivable; together the next hour of
        # rain has nowhere to go but the channel.
        if (wl >= cls.WL_WATCH_M and rain >= cls.RAIN_ALERT_MM
                and soil >= cls.SOIL_SATURATED_PCT):
            level = max(level, 2)
            reasons.append(f"high water {wl:.2f} m with {rain:.0f} mm rain on "
                           f"{soil:.0f}% saturated ground - runoff, not absorption")

        # Layer 3: forecast rain on already-wet ground is an anticipatory signal
        # a sensor-only system cannot see yet.
        if forecast_rain_24h_mm is not None:
            if (soil >= cls.SOIL_SATURATED_PCT - 10
                    and forecast_rain_24h_mm >= cls.RAIN_WATCH_MM):
                level = max(level, 1)
                reasons.append(
                    f"forecast rain {forecast_rain_24h_mm:.0f} mm on already-wet "
                    f"ground ({soil:.0f}%)")

        return level, reasons


class _NodeState:
    """Per-node buffer plus hysteresis bookkeeping."""

    __slots__ = ("packets", "reported_level", "candidate_level", "candidate_since",
                 "clear_since", "last_ts", "last_wl", "pressure_baseline_ts",
                 "pressure_baseline", "pressure_trend_cached")

    def __init__(self) -> None:
        self.packets: deque[dict] = deque(maxlen=512)
        self.reported_level = 0
        self.candidate_level = 0
        self.candidate_since = 0.0
        self.clear_since = 0.0
        self.last_ts: float | None = None
        self.last_wl: float | None = None
        self.pressure_baseline_ts: float | None = None
        self.pressure_baseline: float | None = None
        self.pressure_trend_cached: float | None = None


class FloodPredictor:
    def __init__(self, model_path: str = MODEL_PATH, card_path: str = CARD_PATH):
        self.model = None
        self.feature_cols: list[str] = []
        self.feature_ranges: dict[str, dict] = {}
        self.model_error: str | None = None
        self._nodes: dict[str, _NodeState] = {}

        try:
            import joblib  # imported lazily so Layer 1 works without sklearn
            artifact = joblib.load(model_path)
            self.model = artifact["model"]
            self.feature_cols = list(artifact["features"])
            print(f"[predict] Layer 2 model loaded: {len(self.feature_cols)} features")
        except Exception as exc:  # noqa: BLE001 - degrade to Layer 1
            self.model_error = f"{type(exc).__name__}: {exc}"
            print(f"[predict] WARNING: Layer 2 unavailable ({self.model_error}). "
                  "Layer 1 safety rules still active.")

        try:
            with open(card_path, encoding="utf-8") as fh:
                self.feature_ranges = json.load(fh).get("feature_ranges", {})
        except Exception as exc:  # noqa: BLE001
            print(f"[predict] WARNING: no model card ({exc}); OOD check disabled.")

    # ── helpers ──────────────────────────────────────────────────────

    def _state(self, node_id: str) -> _NodeState:
        with _LOCK:
            if node_id not in self._nodes:
                self._nodes[node_id] = _NodeState()
            return self._nodes[node_id]

    @staticmethod
    def _floats(data: dict) -> dict[str, float]:
        # Flow velocity and turbidity left the hardware roster; humidity is
        # now a Layer-1 input (the storm-front compound rule needs it).
        keys = ["water_level_m", "rainfall_24h_mm", "soil_moisture_pct",
                "humidity_pct"]
        out = {}
        for k in keys:
            try:
                out[k] = float(data.get(k, 0.0) or 0.0)
            except (TypeError, ValueError):
                out[k] = 0.0
        return out

    @staticmethod
    def _reported_trend(data: dict) -> float | None:
        """
        The device's own hPa/hour figure, when it publishes one. Rejects
        physically absurd magnitudes: a genuine barometric trend is single
        digits, so anything past +/-50 is a bug or a corrupted packet and
        must not be allowed to trip the storm rule.
        """
        raw = data.get("pressure_trend_hpa_per_hr")
        if raw is None:
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        return val if -50.0 <= val <= 50.0 else None

    @staticmethod
    def _pressure(data: dict) -> float | None:
        """
        pressure_hpa is optional - nodes without a BMP180 simply omit it, and
        that must not raise or fabricate a fake reading. Distinct from
        `_floats()` because a missing/invalid value here means "no pressure
        signal available" (None), not "treat as zero" (0.0 would look like a
        physically impossible vacuum and misfire the drop rule).
        """
        raw = data.get("pressure_hpa")
        if raw is None:
            return None
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        return val if 300.0 <= val <= 1100.0 else None

    def _resample(self, packets: list[dict], now: float) -> list[dict]:
        """
        Collapse the raw packet buffer into RESAMPLE_WINDOW_SEC bins, newest
        last.  One bin stands in for one training step.  Rainfall is summed
        within a bin (it is an accumulation); everything else is averaged.
        """
        if not packets:
            return []
        bins: dict[int, list[dict]] = {}
        for p in packets:
            idx = int((now - p["ts"]) // RESAMPLE_WINDOW_SEC)
            bins.setdefault(idx, []).append(p)
        out = []
        for idx in sorted(bins.keys(), reverse=True):  # oldest bin first
            grp = bins[idx]
            agg = {k: float(np.mean([g[k] for g in grp]))
                   for k in ("water_level_m", "soil_moisture_pct",
                             "humidity_pct")}
            agg["rainfall_24h_mm"] = float(np.mean([g["rainfall_24h_mm"] for g in grp]))
            out.append(agg)
        return out

    def _ood(self, floats: dict[str, float]) -> list[str]:
        """Features outside the model's training support (from model_card.json)."""
        bad = []
        for feat, rng in self.feature_ranges.items():
            if feat not in floats:
                continue
            v = floats[feat]
            if v < rng["min"] or v > rng["max"]:
                bad.append(f"{feat}={v:.2f} outside training range "
                           f"[{rng['min']:.2f}, {rng['max']:.2f}]")
        return bad

    def _layer2(self, steps: list[dict]) -> dict | None:
        if self.model is None or not steps:
            return None
        cur = steps[-1]
        prev = steps[-2] if len(steps) > 1 else cur

        # Feature set must match ml_pipeline/train.py's FEATURE_COLS exactly.
        # discharge_m3s and turbidity_spike went with the sensors that fed
        # them: discharge was water_level x flow_velocity, which is undefined
        # without a velocity reading (and was mislabelled anyway - depth x
        # velocity is m2/s, not m3/s).
        w, r = cur["water_level_m"], cur["rainfall_24h_mm"]
        s = cur["soil_moisture_pct"]

        feats = {
            "water_level_m": w, "rainfall_24h_mm": r, "soil_moisture_pct": s,
            "rainfall_72h_mm": float(sum(x["rainfall_24h_mm"] for x in steps[-3:])),
            "water_level_change": w - prev["water_level_m"],
            "soil_saturated": 1 if s > 85.0 else 0,
            "water_level_lag1": prev["water_level_m"],
            "rainfall_lag1": prev["rainfall_24h_mm"],
        }
        try:
            X = pd.DataFrame([feats])[self.feature_cols]
            proba = self.model.predict_proba(X)[0]
        except Exception as exc:  # noqa: BLE001
            return {"status": None, "error": f"{type(exc).__name__}: {exc}"}

        probs = {name: 0.0 for name in LABEL_NAMES}
        for i, cls in enumerate(self.model.classes_):
            probs[LABEL_NAMES[int(cls)]] = round(float(proba[i]), 4)
        idx = int(np.argmax(proba))
        return {
            "status": LABEL_NAMES[int(self.model.classes_[idx])],
            "confidence": round(float(proba[idx]), 3),
            "probabilities": probs,
            "note": "advisory only - does not drive alarms",
        }

    # ── main entry point ─────────────────────────────────────────────

    def predict(
        self,
        node_id: str,
        current_data: dict,
        *,
        forecast_rain_24h_mm: float | None = None,
    ) -> dict[str, Any]:
        now = time.monotonic()
        floats = self._floats(current_data)
        pressure = self._pressure(current_data)
        st = self._state(node_id)

        with _LOCK:
            # Rate of rise, in metres per minute, from the previous packet.
            rise = 0.0
            if st.last_ts is not None and st.last_wl is not None:
                dt = now - st.last_ts
                if dt > 0.5:
                    rise = (floats["water_level_m"] - st.last_wl) / dt * 60.0
            st.last_ts, st.last_wl = now, floats["water_level_m"]

            # Pressure trend, hPa/hour. The DEVICE's own figure wins when it
            # sends one: the firmware keeps a ring buffer over its full sample
            # history, which is strictly better information than anything the
            # server can reconstruct from packets that may have been dropped,
            # delayed or reordered in transit. The server-side rolling
            # baseline below is the fallback for nodes that report a raw
            # pressure but no trend (older firmware, or a node whose window
            # has not filled yet).
            reported_trend = self._reported_trend(current_data)
            if reported_trend is not None:
                pressure_trend = reported_trend
                # Keep the baseline moving so a later fallback is not computed
                # against a stale anchor.
                if pressure is not None:
                    st.pressure_baseline_ts, st.pressure_baseline = now, pressure
            else:
                if pressure is not None:
                    if st.pressure_baseline_ts is None:
                        st.pressure_baseline_ts, st.pressure_baseline = now, pressure
                    elif now - st.pressure_baseline_ts >= PRESSURE_TREND_WINDOW_SEC:
                        dt_p = now - st.pressure_baseline_ts
                        st.pressure_trend_cached = (
                            (pressure - st.pressure_baseline) / dt_p * 3600.0)
                        st.pressure_baseline_ts, st.pressure_baseline = now, pressure
                    # else: window not elapsed yet - keep reporting the last
                    # computed trend rather than a freshly noisy one.
                pressure_trend = st.pressure_trend_cached if pressure is not None else None

            st.packets.append({"ts": now, **floats})
            while st.packets and now - st.packets[0]["ts"] > BUFFER_SECONDS:
                st.packets.popleft()
            packets = list(st.packets)

            raw_level, reasons = SafetyRules.evaluate(
                floats, rise_m_per_min=rise,
                pressure_trend_hpa_per_hr=pressure_trend,
                forecast_rain_24h_mm=forecast_rain_24h_mm,
            )

            # ── hysteresis + corroboration bypass ──
            corroborated = len(reasons) >= SafetyRules.CORROBORATION_MIN
            if raw_level > st.reported_level:
                if st.candidate_level != raw_level:
                    st.candidate_level = raw_level
                    st.candidate_since = now
                held = now - st.candidate_since
                if corroborated or held >= SafetyRules.ESCALATE_AFTER_SEC:
                    st.reported_level = raw_level
                    st.clear_since = 0.0
            elif raw_level < st.reported_level:
                st.candidate_level = raw_level
                if st.clear_since == 0.0:
                    st.clear_since = now
                if now - st.clear_since >= SafetyRules.DEESCALATE_AFTER_SEC:
                    st.reported_level = raw_level
                    st.clear_since = 0.0
            else:
                st.candidate_level = raw_level
                st.clear_since = 0.0

            level = st.reported_level
            debounced = level != raw_level

        if level == 0 and not reasons:
            reasons = ["all sensors within normal thresholds"]

        steps = self._resample(packets, now)
        advisory = self._layer2(steps)
        ood = self._ood(floats)

        return {
            "status": LABEL_NAMES[level],
            "level": level,
            "reasons": reasons,
            "rise_m_per_min": round(rise, 3),
            "pressure_hpa": pressure,
            "pressure_trend_hpa_per_hr": (
                round(pressure_trend, 2) if pressure_trend is not None else None),
            "debounced": debounced,
            "raw_status": LABEL_NAMES[raw_level],
            "decided_by": "layer1_safety_rules",
            "forecast_advisory": advisory,
            "ood_features": ood,
            "model_available": self.model is not None,
            "model_error": self.model_error,
            "sensor_source": current_data.get("sensor_source"),
            # Back-compat: older callers and the frontend read `probabilities`.
            "confidence": 1.0,
            "probabilities": (advisory or {}).get(
                "probabilities", {n: 0.0 for n in LABEL_NAMES}),
        }

    def reset(self, node_id: str | None = None) -> None:
        """Clear hysteresis/buffer state. Used by tests."""
        with _LOCK:
            if node_id is None:
                self._nodes.clear()
            else:
                self._nodes.pop(node_id, None)


# Global instance imported by the MQTT bridge.
predictor = FloodPredictor()


def predict_flood(
    water_level_m: float,
    rainfall_24h_mm: float,
    soil_moisture_pct: float,
    node_id: str = "default",
) -> dict:
    """Backward-compatible wrapper kept for older callers."""
    res = predictor.predict(node_id, {
        "water_level_m": water_level_m,
        "rainfall_24h_mm": rainfall_24h_mm,
        "soil_moisture_pct": soil_moisture_pct,
    })
    return {
        "alert_level": res["status"],
        "class_id": res["level"],
        "reasons": res["reasons"],
        "probabilities": res["probabilities"],
    }
