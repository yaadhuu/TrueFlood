"""
Rescale the synthetic 2021 flood dataset onto the Wokwi sensor physical ranges.

Why this exists
---------------
The original CSV was generated on scales that no simulated sensor in this
project can produce.  `water_level_m` runs 0.009 - 104.4 m while the ESP32
firmware's ultrasonic path is bounded by `TANK_DEPTH_M = 4.0`.  Training on
one scale and serving on another is training/serving skew: every live packet
lands outside the model's training support, so the model's output is
extrapolation, not prediction.

The transform
-------------
Per raw column, an affine map anchored on robust percentiles:

    x' = (x - p0.5) / (p99.5 - p0.5) * (sensor_max - sensor_min) + sensor_min
    x' = clip(x', sensor_min, sensor_max)

Anchoring on p0.5/p99.5 rather than min/max keeps a single outlier from
compressing the whole useful range into a sliver.  The map is monotone, so
percentile ranks are preserved exactly -- which is what lets the labels be
re-derived by pushing the original decision boundaries through the same map.

Labels
------
`flood_event` in the source CSV is not an observed flood.  It is a
deterministic threshold rule; a depth-3 decision tree recovers it at 99.1%
accuracy:

    water_level_m <= 1.50                        -> 0   (1 if flow > 1.67)
    1.50 < water_level_m <= 25.35                -> 1   (2 if turbidity > 665.5)
    water_level_m > 25.35                        -> 2

Rather than keep the original 0/1/2 labels on top of rescaled features
(which would be a new and subtler mismatch), the same rule is re-applied
using breakpoints pushed through the same affine map.  The label distribution
is therefore preserved, and the label stays consistent with the features.

This does NOT fix the underlying problem that the label is a function of the
features (see MODEL_CARD.md).  It fixes the range mismatch only.

Usage:
    python ml_pipeline/rescale_dataset.py
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_FILE = os.path.join(SCRIPT_DIR, "flood_dataset_2021_final.csv")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "flood_dataset_wokwi_scale.csv")

# Target ranges: each one matches a concrete constant in firmware/sketch.ino.
#   water_level_m     <- TANK_DEPTH_M            = 4.0 m
#   rainfall_24h_mm   <- readRainfall()          0-200 mm
#   soil_moisture_pct <- readSoilMoisture()      0-100 %   (already correct)
#   flow_velocity_ms  <- readFlowVelocity()      0-10 m/s
#   turbidity_ntu     <- readTurbidity()         0-1000 NTU
SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "water_level_m": (0.0, 4.0),
    "rainfall_24h_mm": (0.0, 200.0),
    "flow_velocity_ms": (0.0, 10.0),
    "turbidity_ntu": (0.0, 1000.0),
}
# soil_moisture_pct is already 0-100 and is deliberately left untouched.
PASSTHROUGH = ["soil_moisture_pct"]

# Original label rule breakpoints, recovered from the source data.
ORIG_WL_WATCH = 1.50
ORIG_WL_ALERT = 25.35
ORIG_TURB_ALERT = 665.5
ORIG_FLOW_WATCH = 1.67

LO_Q, HI_Q = 0.005, 0.995


def fit_affine(series: pd.Series, lo_out: float, hi_out: float):
    """Return (transform_fn, p_lo, p_hi) for a percentile-anchored affine map."""
    p_lo = float(series.quantile(LO_Q))
    p_hi = float(series.quantile(HI_Q))
    if p_hi <= p_lo:  # degenerate column - fall back to min/max
        p_lo, p_hi = float(series.min()), float(series.max())
    if p_hi <= p_lo:  # still degenerate (constant column)
        return (lambda x: np.full_like(np.asarray(x, dtype=float), lo_out), p_lo, p_hi)

    span_in = p_hi - p_lo
    span_out = hi_out - lo_out

    def transform(x):
        x = np.asarray(x, dtype=float)
        return np.clip((x - p_lo) / span_in * span_out + lo_out, lo_out, hi_out)

    return transform, p_lo, p_hi


def engineer_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute every derived column from the (rescaled) raw columns."""
    df["discharge_m3s"] = df["water_level_m"] * df["flow_velocity_ms"]
    df["rainfall_72h_mm"] = df["rainfall_24h_mm"].rolling(3, min_periods=1).sum()
    df["water_level_change"] = df["water_level_m"].diff().fillna(0.0)
    df["turbidity_spike"] = df["turbidity_ntu"].diff().fillna(0.0)
    df["soil_saturated"] = (df["soil_moisture_pct"] > 85).astype(int)
    df["water_level_lag1"] = df["water_level_m"].shift(1).bfill()
    df["rainfall_lag1"] = df["rainfall_24h_mm"].shift(1).bfill()
    df["discharge_lag1"] = df["discharge_m3s"].shift(1).bfill()
    return df


def derive_labels(df: pd.DataFrame, bp: dict[str, float]) -> pd.Series:
    """Re-apply the original threshold rule with rescaled breakpoints."""
    wl = df["water_level_m"]
    turb = df["turbidity_ntu"]
    flow = df["flow_velocity_ms"]

    level = pd.Series(0, index=df.index, dtype=int)
    level[wl > bp["wl_watch"]] = 1
    level[(wl <= bp["wl_watch"]) & (flow > bp["flow_watch"])] = 1
    level[(wl > bp["wl_watch"]) & (turb > bp["turb_alert"])] = 2
    level[wl > bp["wl_alert"]] = 2
    return level


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=SOURCE_FILE)
    ap.add_argument("--out", default=OUTPUT_FILE)
    args = ap.parse_args()

    df = pd.read_csv(args.source)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    original = df.copy()

    print("=" * 78)
    print("RESCALE: source ->", os.path.basename(args.source))
    print("=" * 78)

    transforms = {}
    rows = []
    for col, (lo_out, hi_out) in SENSOR_RANGES.items():
        fn, p_lo, p_hi = fit_affine(df[col], lo_out, hi_out)
        transforms[col] = fn
        before = df[col]
        after = pd.Series(fn(before.values), index=df.index)
        rows.append(
            {
                "column": col,
                "before_min": before.min(),
                "before_max": before.max(),
                "before_mean": before.mean(),
                "p0.5": p_lo,
                "p99.5": p_hi,
                "after_min": after.min(),
                "after_max": after.max(),
                "after_mean": after.mean(),
                "clipped_rows": int(((before < p_lo) | (before > p_hi)).sum()),
            }
        )
        df[col] = after

    for col in PASSTHROUGH:
        rows.append(
            {
                "column": col + " (passthrough)",
                "before_min": df[col].min(),
                "before_max": df[col].max(),
                "before_mean": df[col].mean(),
                "p0.5": float("nan"),
                "p99.5": float("nan"),
                "after_min": df[col].min(),
                "after_max": df[col].max(),
                "after_mean": df[col].mean(),
                "clipped_rows": 0,
            }
        )

    table = pd.DataFrame(rows).set_index("column")
    print("\nBefore / after ranges")
    print(table.round(3).to_string())

    # Order matters: raw columns first (above), then derived columns.
    df = engineer_derived(df)

    breakpoints = {
        "wl_watch": float(transforms["water_level_m"](ORIG_WL_WATCH)),
        "wl_alert": float(transforms["water_level_m"](ORIG_WL_ALERT)),
        "turb_alert": float(transforms["turbidity_ntu"](ORIG_TURB_ALERT)),
        "flow_watch": float(transforms["flow_velocity_ms"](ORIG_FLOW_WATCH)),
    }

    print("\nLabel breakpoints pushed through the same transform")
    print(f"  water_level WATCH : {ORIG_WL_WATCH:8.2f} m   -> {breakpoints['wl_watch']:6.3f} m")
    print(f"  water_level ALERT : {ORIG_WL_ALERT:8.2f} m   -> {breakpoints['wl_alert']:6.3f} m")
    print(f"  turbidity   ALERT : {ORIG_TURB_ALERT:8.2f} NTU -> {breakpoints['turb_alert']:6.1f} NTU")
    print(f"  flow        WATCH : {ORIG_FLOW_WATCH:8.2f} m/s -> {breakpoints['flow_watch']:6.2f} m/s")

    df["flood_event"] = derive_labels(df, breakpoints)

    old_dist = original["flood_event"].value_counts().sort_index()
    new_dist = df["flood_event"].value_counts().sort_index()
    agree = float((original["flood_event"].values == df["flood_event"].values).mean())
    print("\nLabel distribution   (0=NORMAL 1=WATCH 2=ALERT)")
    print("  original:", dict(old_dist))
    print("  rescaled:", dict(new_dist))
    print(f"  agreement with original labels: {agree:.3%}")

    # Sanity check referenced by backend/predict.py's
    # SafetyRules docstring: the Layer-1 alarm threshold must land inside the
    # rescaled ALERT band, otherwise the two layers disagree about what
    # "flood" means on the same physical scale.
    wl_alert_m = 3.0
    inside = breakpoints["wl_alert"] <= wl_alert_m <= SENSOR_RANGES["water_level_m"][1]
    print(
        f"\nSafetyRules.WL_ALERT_M = {wl_alert_m} m is "
        f"{'INSIDE' if inside else 'OUTSIDE'} the rescaled ALERT band "
        f"[{breakpoints['wl_alert']:.3f}, {SENSOR_RANGES['water_level_m'][1]:.1f}] m"
    )
    if not inside:
        print("  WARNING: Layer 1 and Layer 2 disagree on scale - fix before shipping.")

    df.to_csv(args.out, index=False)
    print(f"\nWrote {len(df)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
