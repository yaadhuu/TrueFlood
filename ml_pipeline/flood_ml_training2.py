"""
Deployment Version: FloodSense ML Trainer
Run this ONCE locally to generate the 'flood_model.joblib' file.
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(SCRIPT_DIR, "flood_dataset_2021_final.csv")
MODEL_OUT  = os.path.join(SCRIPT_DIR, "flood_model.joblib")

FORECAST_HORIZON = 2

FEATURE_COLS = [
    "water_level_m", "rainfall_24h_mm", "soil_moisture_pct",
    "flow_velocity_ms", "turbidity_ntu", "discharge_m3s",
    "rainfall_72h_mm", "water_level_change", "turbidity_spike",
    "soil_saturated", "water_level_lag1", "rainfall_lag1", "discharge_lag1"
]

def build_and_save_model():
    print("[INIT] Starting Production Training Pipeline...")
    
    # 1. Load Data
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    
    # 2. Engineer Features
    df["discharge_m3s"] = df["water_level_m"] * df["flow_velocity_ms"]
    df["rainfall_72h_mm"] = df["rainfall_24h_mm"].rolling(3, min_periods=1).sum()
    df["water_level_change"] = df["water_level_m"].diff().fillna(0)
    df["turbidity_spike"] = df["turbidity_ntu"].diff().fillna(0)
    df["soil_saturated"] = (df["soil_moisture_pct"] > 85).astype(int)
    
    df["water_level_lag1"] = df["water_level_m"].shift(1)
    df["rainfall_lag1"] = df["rainfall_24h_mm"].shift(1)
    df["discharge_lag1"] = df["discharge_m3s"].shift(1)
    
    # Shift target for forecasting
    df["future_flood_event"] = df["flood_event"].shift(-FORECAST_HORIZON)
    df = df.fillna(0)
    
    # 3. Chronological Split
    df = df.dropna(subset=FEATURE_COLS + ["future_flood_event"])
    X = df[FEATURE_COLS]
    y = df["future_flood_event"].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 4. Train
    print("[TRAIN] Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5, 
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 5. Save Artifact
    joblib.dump({"model": model, "features": FEATURE_COLS}, MODEL_OUT)
    print(f"[SUCCESS] Model successfully saved to {MODEL_OUT}")

if __name__ == "__main__":
    build_and_save_model()
