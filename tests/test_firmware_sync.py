"""
Keep the firmware's local risk matrix in sync with the backend's Layer 1.

firmware/sketch.ino computes its own SAFE/WARNING/CRITICAL verdict so a node
whose uplink dies still alarms locally. That fallback is only trustworthy if
it uses the SAME thresholds as backend/predict.py -- otherwise the LCD and the
dashboard disagree in front of whoever is watching, and nobody can tell which
one is lying.

Nothing in the build enforces that: the constants live in two languages, in
two files, and drift silently the moment somebody tunes one side. These tests
are the enforcement.
"""

from __future__ import annotations

import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKETCH = os.path.join(REPO_ROOT, "firmware", "sketch.ino")

import sys
sys.path.insert(0, REPO_ROOT)
from backend.predict import SafetyRules  # noqa: E402


def sketch_source() -> str:
    with open(SKETCH, encoding="utf-8") as fh:
        return fh.read()


def sketch_defines() -> dict[str, float]:
    """Every numeric #define in the sketch, as {name: value}."""
    out: dict[str, float] = {}
    pattern = re.compile(
        r"^\s*#define\s+(\w+)\s+(-?[\d.]+)[fFuUlL]*\s*(?://.*)?$", re.M)
    for name, raw in pattern.findall(sketch_source()):
        try:
            out[name] = float(raw)
        except ValueError:
            continue
    return out


# (firmware #define, SafetyRules attribute)
SHARED_THRESHOLDS = [
    ("WL_WATCH_M", "WL_WATCH_M"),
    ("WL_ALERT_M", "WL_ALERT_M"),
    ("RAIN_WATCH_MM", "RAIN_WATCH_MM"),
    ("RAIN_ALERT_MM", "RAIN_ALERT_MM"),
    ("SOIL_SATURATED_PCT", "SOIL_SATURATED_PCT"),
    ("HUMIDITY_STORM_PCT", "HUMIDITY_STORM_PCT"),
    ("RAIN_STARTED_MM", "RAIN_STARTED_MM"),
    # Flow-velocity and turbidity thresholds went with their sensors in the
    # hardware refresh; the barometric pair replaced them.
    ("PRESSURE_FALL_WARNING", "PRESSURE_FALL_WARNING_HPA_HR"),
    ("PRESSURE_FALL_SEVERE", "PRESSURE_FALL_SEVERE_HPA_HR"),
]


@pytest.mark.parametrize("define_name,rule_attr", SHARED_THRESHOLDS)
def test_firmware_threshold_matches_backend(define_name, rule_attr):
    defines = sketch_defines()
    assert define_name in defines, f"{define_name} missing from sketch.ino"
    firmware_value = defines[define_name]
    backend_value = float(getattr(SafetyRules, rule_attr))
    assert firmware_value == backend_value, (
        f"{define_name}={firmware_value} in firmware/sketch.ino but "
        f"SafetyRules.{rule_attr}={backend_value} in backend/predict.py. "
        "The node and the server would disagree about the same reading."
    )


def test_pressure_trend_window_matches_backend():
    """
    Both sides derive an hPa/HOUR rate against a rolling baseline. If the
    windows differ the two compute measurably different slopes from identical
    readings, which is the same disagreement problem in slower motion.
    """
    import backend.predict as predict_module
    firmware_ms = sketch_defines()["TREND_WINDOW_MS"]
    backend_sec = float(predict_module.PRESSURE_TREND_WINDOW_SEC)
    assert firmware_ms / 1000.0 == backend_sec, (
        f"firmware window {firmware_ms/1000.0}s vs backend {backend_sec}s"
    )


def test_tank_depth_matches_rescaled_training_range():
    """
    TANK_DEPTH_M is the ceiling of every water-level path. The training data
    is rescaled to that same ceiling by ml_pipeline/rescale_dataset.py; if
    they diverge the model is extrapolating on every deep reading.
    """
    from ml_pipeline.rescale_dataset import SENSOR_RANGES
    firmware_depth = sketch_defines()["TANK_DEPTH_M"]
    _, dataset_max = SENSOR_RANGES["water_level_m"]
    assert firmware_depth == dataset_max


def test_loop_contains_no_blocking_delay():
    """
    The alert patterns (flashing LED, chirping buzzer, scrolling marquee) are
    all millis()-scheduled. A single delay() inside loop() stalls mqtt.loop()
    too, which risks a broker keepalive timeout -- the bug this firmware
    already had once, in the MQTT callback.
    """
    src = sketch_source()
    loop_start = src.index("void loop()")
    loop_body = src[loop_start:]
    offenders = [ln.strip() for ln in loop_body.splitlines()
                 if re.search(r"\bdelay\s*\(", ln) and not ln.strip().startswith("//")]
    assert not offenders, f"blocking delay() reachable from loop(): {offenders}"


def test_analog_pins_are_all_adc1():
    """
    The ESP32's ADC2 is owned by the WiFi radio: analogRead() on an ADC2 pin
    returns garbage once WiFi is up, and this firmware keeps WiFi on
    permanently. Only ADC1 (GPIO32-39) is safe for the analog sensors.
    """
    adc1 = {32, 33, 34, 35, 36, 37, 38, 39}
    defines = sketch_defines()
    for pin_name in ("PIN_RAIN", "PIN_SOIL"):
        gpio = int(defines[pin_name])
        assert gpio in adc1, (
            f"{pin_name}=GPIO{gpio} is not on ADC1; analogRead() there returns "
            "noise while WiFi is active"
        )


def test_lcd_writes_go_through_the_width_limiter():
    """
    A 16x2 LCD mangles anything longer than 16 characters per line, silently.
    Status lines are built with lcdLine(), which pads/truncates to exactly 16;
    this guards against someone reintroducing raw chained lcd.print() of a
    value whose width depends on the reading.
    """
    src = sketch_source()
    assert 'snprintf(dst, 17, "%-16.16s", tmp);' in src, (
        "lcdLine() no longer clamps to 16 chars"
    )
    start = src.index("void drawStatus(")
    end = src.index("void updateLCD")
    body = src[start:end]
    # Inside the status renderer, only the two prepared buffers may be printed.
    prints = re.findall(r"lcd\.print\(([^)]*)\)", body)
    assert set(prints) <= {"l0", "l1"}, (
        f"drawStatusScreen prints unclamped values: {prints}"
    )
