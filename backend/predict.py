"""
Deployment Version: FloodSense Live Inference Engine
Computes rolling time-series features in-memory for live MQTT packets.
"""

import os
import joblib
import pandas as pd
import numpy as np
from collections import deque

# Resolve the path to where the trainer saved the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_pipeline", "flood_model.joblib")
LABEL_NAMES = ["NORMAL", "WATCH", "ALERT"]

# In-memory streaming buffer (keeps the last 20 readings per ESP32 node)
NODE_BUFFERS = {}

class FloodPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.model = None
        self.feature_cols = []
        try:
            artifact = joblib.load(model_path)
            self.model = artifact["model"]
            self.feature_cols = artifact["features"]
            print(f"[SUCCESS] ML Model loaded successfully with {len(self.feature_cols)} features.")
        except Exception as e:
            print(f"[WARNING] Warning: Could not load ML model. {e}")

    def predict(self, node_id, current_data):
        if self.model is None:
            # Safe fallback if the model file is missing on the server
            return {
                "status": "NORMAL",
                "confidence": 1.0,
                "probabilities": {"NORMAL": 1.0, "WATCH": 0.0, "ALERT": 0.0}
            }

        # Initialize buffer for a new node
        if node_id not in NODE_BUFFERS:
            NODE_BUFFERS[node_id] = deque(maxlen=20)
            
        buffer = NODE_BUFFERS[node_id]
        
        # Parse incoming live data
        w_curr = float(current_data.get("water_level_m", 0.0))
        r_curr = float(current_data.get("rainfall_24h_mm", 0.0))
        s_curr = float(current_data.get("soil_moisture_pct", 0.0))
        v_curr = float(current_data.get("flow_velocity_ms", 0.0))
        t_curr = float(current_data.get("turbidity_ntu", 0.0))

        discharge_curr = w_curr * v_curr

        # Fetch Lag 1 (previous) data from the buffer, or use current if it's the very first packet
        if len(buffer) > 0:
            prev = buffer[-1]
            w_lag1, r_lag1, discharge_lag1, t_lag1 = prev["w"], prev["r"], prev["d"], prev["t"]
        else:
            w_lag1, r_lag1, discharge_lag1, t_lag1 = w_curr, r_curr, discharge_curr, t_curr

        # Compute Trend Features
        recent_rain = [p["r"] for p in buffer] + [r_curr]
        rainfall_72h = float(np.sum(recent_rain[-3:]))
        
        water_level_change = w_curr - w_lag1
        turbidity_spike = t_curr - t_lag1
        soil_saturated = 1 if s_curr > 85.0 else 0

        # Save this packet to the buffer for the NEXT prediction
        buffer.append({"w": w_curr, "r": r_curr, "d": discharge_curr, "t": t_curr})

        # Build the exact feature array the model expects
        features_dict = {
            "water_level_m": w_curr, "rainfall_24h_mm": r_curr, "soil_moisture_pct": s_curr,
            "flow_velocity_ms": v_curr, "turbidity_ntu": t_curr, "discharge_m3s": discharge_curr,
            "rainfall_72h_mm": rainfall_72h, "water_level_change": water_level_change,
            "turbidity_spike": turbidity_spike, "soil_saturated": soil_saturated,
            "water_level_lag1": w_lag1, "rainfall_lag1": r_lag1, "discharge_lag1": discharge_lag1
        }

        # Run inference
        X_live = pd.DataFrame([features_dict])[self.feature_cols]
        probs = self.model.predict_proba(X_live)[0]
        pred_idx = int(np.argmax(probs))
        
        # Build probabilities dict for backward compatibility
        prob_dict = {}
        for i, cls in enumerate(self.model.classes_):
            prob_dict[LABEL_NAMES[int(cls)]] = round(float(probs[i]), 4)
        for name in LABEL_NAMES:
            if name not in prob_dict:
                prob_dict[name] = 0.0

        return {
            "status": LABEL_NAMES[pred_idx],
            "confidence": round(float(probs[pred_idx]), 3),
            "probabilities": prob_dict
        }

# Global instance to be imported by the MQTT bridge
predictor = FloodPredictor()

# Backward compatibility wrapper for mqtt_bridge.py
def predict_flood(
    water_level_m: float,
    rainfall_24h_mm: float,
    soil_moisture_pct: float,
    flow_velocity_ms: float,
    turbidity_ntu: float,
    node_id: str = "default",
) -> dict:
    current_data = {
        "water_level_m": water_level_m,
        "rainfall_24h_mm": rainfall_24h_mm,
        "soil_moisture_pct": soil_moisture_pct,
        "flow_velocity_ms": flow_velocity_ms,
        "turbidity_ntu": turbidity_ntu,
    }
    res = predictor.predict(node_id, current_data)
    return {
        "alert_level": res["status"],
        "class_id": LABEL_NAMES.index(res["status"]),
        "probabilities": res["probabilities"]
    }
