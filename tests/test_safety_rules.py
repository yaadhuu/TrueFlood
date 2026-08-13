"""
Layer 1 safety-rule tests.

These are the tests that matter most in this repo: Layer 1 is the only thing
that fires an alarm, so a regression here is a missed flood, not a worse
metric.  Keep them passing; add alongside, do not replace.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.predict import FloodPredictor, SafetyRules  # noqa: E402

# Fixtures follow the post-hardware-refresh telemetry schema: flow_velocity_ms
# and turbidity_ntu left with their sensors, humidity_pct arrived as a Layer-1
# input for the storm-front compound rule.
CALM = {
    "water_level_m": 0.4,
    "rainfall_24h_mm": 3.0,
    "soil_moisture_pct": 45.0,
    "humidity_pct": 55.0,
}

STORM = {
    "water_level_m": 3.4,
    "rainfall_24h_mm": 190.0,
    "soil_moisture_pct": 99.0,
    "humidity_pct": 95.0,
}


@pytest.fixture
def predictor(tmp_path):
    """A predictor with no model file: Layer 1 only, fresh hysteresis state."""
    return FloodPredictor(model_path=str(tmp_path / "missing.joblib"),
                          card_path=str(tmp_path / "missing.json"))


# ── 1-5: the rule engine itself (pure function, no state) ────────────

def test_calm_conditions_are_normal():
    level, reasons = SafetyRules.evaluate(CALM)
    assert level == 0
    assert reasons == []


def test_storm_is_alert_with_multiple_reasons():
    level, reasons = SafetyRules.evaluate(STORM)
    assert level == 2
    assert len(reasons) >= 2, f"expected corroborating reasons, got {reasons}"


def test_rapid_rise_alone_triggers_alert():
    """Water climbing fast is an alarm even when every level is still low."""
    level, reasons = SafetyRules.evaluate(
        CALM, rise_m_per_min=SafetyRules.RISE_ALERT_M_PER_MIN + 0.05)
    assert level == 2
    assert any("rising" in r for r in reasons)


def test_saturated_soil_plus_rain_escalates():
    """
    Intent unchanged: ground with no absorption capacity left, under peak
    rain, is a CRITICAL. The rule now also requires the channel to already be
    high - saturated ground matters because the runoff has somewhere to go.
    """
    data = {**CALM,
            "water_level_m": SafetyRules.WL_WATCH_M + 0.1,
            "soil_moisture_pct": SafetyRules.SOIL_SATURATED_PCT + 1,
            "rainfall_24h_mm": SafetyRules.RAIN_ALERT_MM + 5}
    level, reasons = SafetyRules.evaluate(data)
    assert level == 2
    assert any("saturated ground" in r for r in reasons)

    # Saturated soil with a dry sky and a low channel is not an emergency.
    dry_sky = {**CALM, "soil_moisture_pct": SafetyRules.SOIL_SATURATED_PCT + 1}
    assert SafetyRules.evaluate(dry_sky)[0] == 0


def test_every_alert_carries_a_human_readable_reason():
    for data in (STORM, {**CALM, "water_level_m": 3.5},
                 {**CALM, "rainfall_24h_mm": SafetyRules.RAIN_WATCH_MM + 5}):
        level, reasons = SafetyRules.evaluate(data)
        assert level > 0
        assert reasons and all(isinstance(r, str) and r.strip() for r in reasons)


# ── 6-7: hysteresis and the corroboration bypass (stateful) ──────────

def test_single_sensor_spike_is_debounced(predictor):
    """One reason is not enough to escalate instantly - it must persist."""
    spike = {**CALM, "water_level_m": SafetyRules.WL_WATCH_M + 0.2}
    res = predictor.predict("node-debounce", spike)
    assert res["status"] == "NORMAL", "a lone spike should not escalate immediately"
    assert res["raw_status"] == "WATCH"
    assert res["debounced"] is True


def test_corroborated_alert_is_immediate(predictor):
    """Two independent reasons skip the debounce - floods do not wait."""
    res = predictor.predict("node-corroborated", STORM)
    assert res["status"] == "ALERT"
    assert res["debounced"] is False
    assert len(res["reasons"]) >= SafetyRules.CORROBORATION_MIN


# ── 8: degradation ───────────────────────────────────────────────────

def test_missing_model_still_fails_safe(predictor):
    """
    No model file must not mean "everything is fine".  Layer 1 alarms, and
    the response says plainly that Layer 2 is unavailable.
    """
    assert predictor.model is None
    res = predictor.predict("node-nomodel", STORM)
    assert res["status"] == "ALERT"
    assert res["model_available"] is False
    assert res["forecast_advisory"] is None
    assert res["decided_by"] == "layer1_safety_rules"

    calm = predictor.predict("node-nomodel-calm", CALM)
    assert calm["status"] == "NORMAL"


# ── additions: Layer 3 weather fusion, de-escalation, ordering ───────

def test_forecast_rain_on_wet_ground_raises_watch():
    """Layer 3: anticipatory signal a sensor-only system cannot see."""
    wet = {**CALM, "soil_moisture_pct": SafetyRules.SOIL_SATURATED_PCT - 5}
    base, _ = SafetyRules.evaluate(wet)
    fused, reasons = SafetyRules.evaluate(
        wet, forecast_rain_24h_mm=SafetyRules.RAIN_WATCH_MM + 10)
    assert base == 0
    assert fused == 1
    assert any("forecast rain" in r for r in reasons)


def test_forecast_alone_on_dry_ground_does_not_alarm():
    fused, _ = SafetyRules.evaluate(CALM, forecast_rain_24h_mm=190.0)
    assert fused == 0


def test_deescalation_is_slow(predictor):
    """After an ALERT, one calm packet must not clear the alarm."""
    predictor.predict("node-hold", STORM)
    res = predictor.predict("node-hold", CALM)
    assert res["status"] == "ALERT", "alarm cleared too fast"
    assert res["raw_status"] == "NORMAL"
    assert res["debounced"] is True


# ── pressure: the BMP180 storm-precursor rule ─────────────────────────

def test_storm_front_precursor_fires_before_water_rises():
    """
    The predictive path, and the reason the BMP180 is on the node: a falling
    barometer + saturated air + rain already falling is a storm front, and it
    escalates while the channel is still low.
    """
    front = {**CALM,
             "humidity_pct": SafetyRules.HUMIDITY_STORM_PCT + 5,
             "rainfall_24h_mm": SafetyRules.RAIN_STARTED_MM + 2}
    level, reasons = SafetyRules.evaluate(
        front, pressure_trend_hpa_per_hr=SafetyRules.PRESSURE_FALL_WARNING_HPA_HR - 0.8)
    assert level == 1
    assert any("pressure falling" in r for r in reasons)
    # ...and the water level genuinely has not moved.
    assert front["water_level_m"] < SafetyRules.WL_WATCH_M


def test_moderate_fall_in_dry_air_does_not_escalate():
    """A moderate fall on its own is just weather, not a flood signal."""
    level, reasons = SafetyRules.evaluate(
        CALM, pressure_trend_hpa_per_hr=SafetyRules.PRESSURE_FALL_WARNING_HPA_HR - 0.5)
    assert level == 0
    assert reasons == []


def test_severe_pressure_fall_escalates_on_its_own():
    """The severe band is the flash-flood-producing one; it stands alone."""
    level, reasons = SafetyRules.evaluate(
        CALM, pressure_trend_hpa_per_hr=SafetyRules.PRESSURE_FALL_SEVERE_HPA_HR - 0.1)
    assert level == 2
    assert any("severe" in r for r in reasons)


def test_rising_or_steady_pressure_never_escalates():
    """Only a fall matters - fine weather must never trip the rule."""
    for trend in (0.0, 0.5, 3.0, None):
        level, reasons = SafetyRules.evaluate(CALM, pressure_trend_hpa_per_hr=trend)
        assert level == 0
        assert reasons == []


def test_pressure_trend_is_computed_from_consecutive_packets(predictor, monkeypatch):
    """
    End-to-end: FloodPredictor tracks pressure per node and derives the
    hPa/hour trend itself from real elapsed time - the firmware only ever
    sends a raw reading (see firmware/sketch.ino's readPressure()).

    The real window is 30s (damping single-packet noise amplification - see
    PRESSURE_TREND_WINDOW_SEC's docstring in predict.py); shrunk here so the
    test doesn't need to sleep 30s to prove the mechanism works.
    """
    import backend.predict as predict_module
    monkeypatch.setattr(predict_module, "PRESSURE_TREND_WINDOW_SEC", 0.3)

    first = predictor.predict("node-baro", {**CALM, "pressure_hpa": 1015.0})
    assert first["pressure_trend_hpa_per_hr"] is None  # baseline just set, no trend yet

    time.sleep(0.4)  # past the shrunk window
    second = predictor.predict("node-baro", {**CALM, "pressure_hpa": 1014.5})
    assert second["pressure_trend_hpa_per_hr"] is not None
    assert second["pressure_trend_hpa_per_hr"] < 0
    assert any("pressure falling" in r for r in second["reasons"])


def test_pressure_trend_does_not_recompute_within_the_window(predictor, monkeypatch):
    """
    A single noisy packet-to-packet delta must not blow up into an absurd
    hourly rate: two packets a fraction of a second apart, inside the window,
    must not report a fresh (noisy) trend at all.
    """
    import backend.predict as predict_module
    monkeypatch.setattr(predict_module, "PRESSURE_TREND_WINDOW_SEC", 30.0)

    first = predictor.predict("node-baro-noisy", {**CALM, "pressure_hpa": 1015.0})
    assert first["pressure_trend_hpa_per_hr"] is None

    second = predictor.predict("node-baro-noisy", {**CALM, "pressure_hpa": 1013.0})
    assert second["pressure_trend_hpa_per_hr"] is None, (
        "a same-instant 2 hPa swing must not be reported as a hair-trigger "
        "hourly rate before the baseline window has actually elapsed"
    )


def test_missing_pressure_sensor_does_not_crash_or_alarm(predictor):
    """Nodes without a BMP180 simply omit pressure_hpa - must degrade cleanly."""
    res = predictor.predict("node-no-baro", CALM)
    assert res["status"] == "NORMAL"
    assert res["pressure_hpa"] is None
    assert res["pressure_trend_hpa_per_hr"] is None


def test_layer2_never_overrides_layer1(tmp_path):
    """
    Even with a model loaded, `status` comes from Layer 1 and the model's
    opinion is confined to `forecast_advisory`.
    """
    p = FloodPredictor()
    res = p.predict("node-layers", STORM)
    assert res["decided_by"] == "layer1_safety_rules"
    if res["forecast_advisory"] is not None:
        assert "note" in res["forecast_advisory"]
    assert res["status"] == "ALERT"
