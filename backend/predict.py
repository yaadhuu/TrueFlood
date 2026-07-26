"""
FloodSense Prediction Module
=============================
Loads the trained Random Forest model and exposes predict_flood().
Thread-safe: _rain_history is guarded by a per-module RLock so concurrent
MQTT callbacks across multiple nodes never corrupt rolling rainfall state.

Model search order:
  1. Same directory as this file  (backend/)
  2. ../ml_pipeline/              (monorepo dev layout)
"""

import os
import threading
from collections import deque

import joblib
import pandas as pd

# ── Model discovery ───────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_MODEL_PATHS = [
    os.path.join(_SCRIPT_DIR, "flood_model.joblib"),
    os.path.join(_SCRIPT_DIR, "..", "ml_pipeline", "flood_model.joblib"),
]

_saved: dict | None = None
for _p in _MODEL_PATHS:
    if os.path.exists(_p):
        _saved = joblib.load(_p)
        print(f"[predict] Model loaded from: {_p}")
        break

if _saved is None:
    raise FileNotFoundError(
        "flood_model.joblib not found. "
        "Run ml_pipeline/flood_ml_training2.py first."
    )

_model        = _saved["model"]
_FEATURE_COLS = _saved["features"]
_ALERT_LABELS = {0: "NORMAL", 1: "WATCH", 2: "ALERT"}

# ── Stateful history (thread-safe) ──────────────────────────────
_state_lock: threading.Lock = threading.Lock()
_rain_history: dict[str, deque] = {}
_prev_reading: dict[str, dict] = {}


# ── Public API ────────────────────────────────────────────────────

def predict_flood(
    water_level_m: float,
    rainfall_24h_mm: float,
    soil_moisture_pct: float,
    flow_velocity_ms: float,
    turbidity_ntu: float,
    node_id: str = "default",
) -> dict:
    """
    Run ML inference for a single sensor reading.
    """
    discharge = water_level_m * flow_velocity_ms
    soil_saturated = int(soil_moisture_pct > 85)

    with _state_lock:
        if node_id not in _rain_history:
            _rain_history[node_id] = deque(maxlen=3)
        _rain_history[node_id].append(rainfall_24h_mm)
        rainfall_72h = sum(_rain_history[node_id])

        prev = _prev_reading.get(node_id, {})
        
        water_level_lag1 = prev.get("water_level_m", water_level_m)
        rainfall_lag1 = prev.get("rainfall_24h_mm", rainfall_24h_mm)
        discharge_lag1 = prev.get("discharge", discharge)
        turbidity_lag1 = prev.get("turbidity_ntu", turbidity_ntu)

        water_level_change = water_level_m - water_level_lag1
        turbidity_spike = turbidity_ntu - turbidity_lag1

        # Save current for next time
        _prev_reading[node_id] = {
            "water_level_m": water_level_m,
            "rainfall_24h_mm": rainfall_24h_mm,
            "discharge": discharge,
            "turbidity_ntu": turbidity_ntu
        }

    row = pd.DataFrame([[
        water_level_m, rainfall_24h_mm, soil_moisture_pct,
        flow_velocity_ms, turbidity_ntu, discharge,
        rainfall_72h,
        water_level_change,
        turbidity_spike,
        soil_saturated,
        water_level_lag1,
        rainfall_lag1,
        discharge_lag1,
    ]], columns=_FEATURE_COLS)


    class_id = int(_model.predict(row)[0])
    raw_probs = _model.predict_proba(row)[0]

    prob_dict = {0: 0.0, 1: 0.0, 2: 0.0}
    for i, cls in enumerate(_model.classes_):
        prob_dict[int(cls)] = float(raw_probs[i])

    return {
        "alert_level": _ALERT_LABELS[class_id],
        "class_id":    class_id,
        "probabilities": {
            "NORMAL": round(prob_dict[0], 4),
            "WATCH":  round(prob_dict[1], 4),
            "ALERT":  round(prob_dict[2], 4),
        },
    }


# ── Demo ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    _cases = [
        dict(water_level_m=0.8,  rainfall_24h_mm=1.5,  soil_moisture_pct=42.0,
             flow_velocity_ms=0.6, turbidity_ntu=110),
        dict(water_level_m=6.5,  rainfall_24h_mm=14.0, soil_moisture_pct=74.0,
             flow_velocity_ms=1.8, turbidity_ntu=390),
        dict(water_level_m=35.0, rainfall_24h_mm=32.0, soil_moisture_pct=91.0,
             flow_velocity_ms=3.5, turbidity_ntu=720),
    ]
    for tc in _cases:
        r = predict_flood(**tc)
        print(
            f"WL={tc['water_level_m']:5.1f}m  "
            f"Rain={tc['rainfall_24h_mm']:5.1f}mm  "
            f"Soil={tc['soil_moisture_pct']}%  "
            f"-> {r['alert_level']:6s}  P={r['probabilities']}"
        )
