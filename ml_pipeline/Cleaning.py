"""
FloodSense – Data Cleaning Pipeline
=====================================
Merges and cleans raw CSVs from the /raw_data folder into a single
analysis-ready dataset: flood_dataset_2021_final.csv

Usage:
  cd ml_pipeline/
  python Cleaning.py
"""

import os
import pandas as pd

# ── Resolve paths relative to this script ────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR    = os.path.join(SCRIPT_DIR, "raw_data")
OUT_FILE   = os.path.join(SCRIPT_DIR, "flood_dataset_2021_final.csv")


# ─────────────────────────────────────────────
# HELPER: FORCE DATETIME
# ─────────────────────────────────────────────
def force_datetime(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df


# ─────────────────────────────────────────────
# 1. LOAD FILES
# ─────────────────────────────────────────────
rain      = pd.read_csv(os.path.join(RAW_DIR, "Rainfall.csv"))
discharge = pd.read_csv(os.path.join(RAW_DIR, "river_discharge_2021.csv"))
water     = pd.read_csv(os.path.join(RAW_DIR, "Water_level(2021).csv"))
flow      = pd.read_csv(os.path.join(RAW_DIR, "flow_velocity_2021.csv"))
soil      = pd.read_csv(os.path.join(RAW_DIR, "soil_moisture_2021.csv"))
turb      = pd.read_csv(os.path.join(RAW_DIR, "turbidity_2021.csv"))


# ─────────────────────────────────────────────
# 2. CLEAN RAINFALL (IMD FORMAT)
# ─────────────────────────────────────────────
rain = rain.rename(columns={"Date": "timestamp", "Avg_rainfall": "rainfall_24h_mm"})
rain = rain[["timestamp", "rainfall_24h_mm"]]
rain = force_datetime(rain)
rain = rain.set_index("timestamp").resample("D").mean().reset_index()


# ─────────────────────────────────────────────
# 3. CLEAN DISCHARGE
# ─────────────────────────────────────────────
discharge = discharge[["timestamp", "discharge_m3s"]]
discharge = force_datetime(discharge)


# ─────────────────────────────────────────────
# 4. CLEAN WATER LEVEL (CRITICAL FIX)
# ─────────────────────────────────────────────
water = water.rename(columns={
    "Data Acquisition Time": "timestamp",
    "River Water Level Telemetry Hourly (meter)": "water_level_m",
})
water["timestamp"] = pd.to_datetime(
    water["timestamp"], format="%d-%m-%Y %H:%M", errors="coerce"
)
water = water.dropna(subset=["timestamp"])
water = water[["timestamp", "water_level_m"]]
water = water.drop_duplicates(subset=["timestamp"])
water = water.set_index("timestamp").resample("D").mean().reset_index()


# ─────────────────────────────────────────────
# 5. CLEAN SIMPLE FILES
# ─────────────────────────────────────────────
def clean_simple(df, col_name):
    df.columns = ["timestamp", col_name]
    df = force_datetime(df)
    df = df.drop_duplicates(subset="timestamp")
    return df

flow = clean_simple(flow, "flow_velocity_ms")
soil = clean_simple(soil, "soil_moisture_pct")
turb = clean_simple(turb, "turbidity_ntu")


# ─────────────────────────────────────────────
# 6. MERGE ALL DATASETS
# ─────────────────────────────────────────────
df = (
    rain
    .merge(discharge, on="timestamp", how="outer")
    .merge(water,     on="timestamp", how="outer")
    .merge(flow,      on="timestamp", how="outer")
    .merge(soil,      on="timestamp", how="outer")
    .merge(turb,      on="timestamp", how="outer")
)


# ─────────────────────────────────────────────
# 7. SORT + HANDLE MISSING VALUES
# ─────────────────────────────────────────────
df = df.sort_values("timestamp").reset_index(drop=True)
df = df.ffill()
df = df.fillna(df.median(numeric_only=True))


# ─────────────────────────────────────────────
# 8. FINAL CLEANING
# ─────────────────────────────────────────────
df = df[df["water_level_m"]   >= 0]
df = df[df["flow_velocity_ms"] >= 0]
df["soil_moisture_pct"] = df["soil_moisture_pct"].clip(0, 100)
df["turbidity_ntu"]     = df["turbidity_ntu"].clip(0, 2000)


# ─────────────────────────────────────────────
# 9. CREATE FLOOD LABEL  (3-class)
# ─────────────────────────────────────────────
NORMAL_MAX  = 5.0   # Below this  → NORMAL (0)
WARNING_MAX = 10.0  # Below this  → WATCH  (1)
                    # Above or eq → ALERT  (2)

def classify(wl):
    if wl < NORMAL_MAX:
        return 0
    elif wl < WARNING_MAX:
        return 1
    else:
        return 2

df["flood_event"] = df["water_level_m"].apply(classify)


# ─────────────────────────────────────────────
# 10. SAVE
# ─────────────────────────────────────────────
df.to_csv(OUT_FILE, index=False)

print("\n✓  DONE!")
print(f"   Saved  : {OUT_FILE}")
print(f"   Shape  : {df.shape}")
print(f"   Labels : {df['flood_event'].value_counts().to_dict()}")
print(df.head())
