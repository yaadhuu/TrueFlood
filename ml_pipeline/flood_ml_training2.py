"""
Flood Risk ML Training Pipeline  —  3-class model (NORMAL / WATCH / ALERT)
===========================================================================
Run:
  cd ml_pipeline/
  python flood_ml_training2.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import joblib
import warnings

warnings.filterwarnings("ignore")

LABEL_NAMES = ["NORMAL", "WATCH", "ALERT"]

# ── Resolve paths relative to this script ────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE  = os.path.join(SCRIPT_DIR, "flood_dataset_2021_final.csv")
MODEL_OUT  = os.path.join(SCRIPT_DIR, "flood_model.joblib")
CM_OUT     = os.path.join(SCRIPT_DIR, "confusion_matrix.png")


# ══════════════════════════════════════════════
# STEP 1: LOAD DATA
# ══════════════════════════════════════════════
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


# ══════════════════════════════════════════════
# STEP 2: INSPECT
# ══════════════════════════════════════════════
def inspect(df):
    print("=" * 55)
    print("DATASET OVERVIEW")
    print("=" * 55)
    print(f"Shape         : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Date range    : {df['timestamp'].min()} → {df['timestamp'].max()}")
    vc = df["flood_event"].value_counts().sort_index()
    for cls, name in enumerate(LABEL_NAMES):
        count = vc.get(cls, 0)
        print(f"{name:8s} (class {cls}) : {count} ({count / len(df) * 100:.1f}%)")
    return df


# ══════════════════════════════════════════════
# STEP 3: FEATURE ENGINEERING
# ══════════════════════════════════════════════
def engineer_features(df):
    df = df.copy()
    if "discharge_m3s" not in df.columns:
        df["discharge_m3s"] = df["water_level_m"] * df["flow_velocity_ms"]

    df["rainfall_72h_mm"]    = df["rainfall_24h_mm"].rolling(3, min_periods=1).sum()
    df["water_level_change"] = df["water_level_m"].diff().fillna(0)
    df["turbidity_spike"]    = df["turbidity_ntu"].diff().fillna(0)
    df["soil_saturated"]     = (df["soil_moisture_pct"] > 85).astype(int)
    df["water_level_lag1"]   = df["water_level_m"].shift(1)
    df["rainfall_lag1"]      = df["rainfall_24h_mm"].shift(1)
    df["discharge_lag1"]     = df["discharge_m3s"].shift(1)
    df = df.fillna(0)
    return df


# ══════════════════════════════════════════════
# STEP 4: PREPROCESS
# ══════════════════════════════════════════════
FEATURE_COLS = [
    "water_level_m", "rainfall_24h_mm", "soil_moisture_pct",
    "flow_velocity_ms", "turbidity_ntu", "discharge_m3s",
    "rainfall_72h_mm", "water_level_change", "turbidity_spike",
    "soil_saturated", "water_level_lag1", "rainfall_lag1", "discharge_lag1",
]


def preprocess(df):
    df = df.dropna(subset=FEATURE_COLS + ["flood_event"])
    X = df[FEATURE_COLS]
    y = df["flood_event"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════
# STEP 5: TRAIN
# ══════════════════════════════════════════════
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    cv = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")
    print(f"\n✓ CV F1-macro: {cv.mean():.3f} ± {cv.std():.3f}")
    return model


# ══════════════════════════════════════════════
# STEP 6: EVALUATE
# ══════════════════════════════════════════════
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=LABEL_NAMES)
    plt.title("Flood Risk – Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {CM_OUT}")


# ══════════════════════════════════════════════
# STEP 7: SAVE
# ══════════════════════════════════════════════
def save_model(model):
    joblib.dump({"model": model, "features": FEATURE_COLS}, MODEL_OUT)
    print(f"Model saved → {MODEL_OUT}")


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("FLOOD ML TRAINING\n")
    df = load_data()
    df = inspect(df)
    df = engineer_features(df)
    X_train, X_test, y_train, y_test = preprocess(df)
    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)
    save_model(model)
    print("\nDONE — Model ready!")
