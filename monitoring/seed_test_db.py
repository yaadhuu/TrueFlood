"""
Seed a throwaway SQLite database with realistic Wokwi-range telemetry.

Used to exercise monitoring/drift.py without a live broker.  Values are what
firmware/sketch.ino actually publishes: water level bounded by
TANK_DEPTH_M = 4.0, rainfall 0-200 mm, soil 0-100 %, flow 0-10 m/s,
turbidity 0-1000 NTU.

Two modes:
  default   sample operating conditions from the rescaled training
            distribution and add sensor noise - this is "a healthy station,
            behaving as expected", so drift.py should report healthy: true.
  --storm   a monotone storm ramp - deliberately non-stationary, so drift.py
            should report a shift.  Useful for checking the monitor is not
            just always saying yes.

Usage:
    python monitoring/seed_test_db.py --db /tmp/test.db --rows 500
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = os.path.join(REPO_ROOT, "ml_pipeline",
                              "flood_dataset_wokwi_scale.csv")

# Post-hardware-refresh schema: flow velocity and turbidity left with their
# sensors; barometric pressure arrived.
SENSOR_CLIP = {
    "water_level_m": (0.0, 4.0),
    "rainfall_24h_mm": (0.0, 200.0),
    "soil_moisture_pct": (0.0, 100.0),
}
# Per-sensor noise, as a fraction of full scale. ~0.2% is the order of an
# ESP32 ADC read: a few LSB out of 4095.
NOISE_FRAC = 0.002


def sample_rows(rng, n: int, source: str) -> list[tuple[float, ...]]:
    df = pd.read_csv(source)
    idx = rng.integers(0, len(df), size=n)
    out = []
    for i in idx:
        row = df.iloc[int(i)]
        vals = []
        for feat, (lo, hi) in SENSOR_CLIP.items():
            noise = rng.normal(0, (hi - lo) * NOISE_FRAC)
            vals.append(float(np.clip(float(row[feat]) + noise, lo, hi)))
        out.append(tuple(vals))
    return out


def storm_rows(rng, n: int) -> list[tuple[float, ...]]:
    out = []
    for i in range(n):
        phase = i / max(n - 1, 1)
        out.append((
            float(np.clip(0.3 + 3.0 * phase ** 2 + rng.normal(0, 0.08), 0, 4)),
            float(np.clip(5 + 170 * phase ** 2 + rng.normal(0, 6), 0, 200)),
            float(np.clip(45 + 50 * phase + rng.normal(0, 3), 0, 100)),
        ))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--storm", action="store_true",
                    help="seed a monotone storm ramp instead of normal operation")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if os.path.exists(args.db):
        os.remove(args.db)

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT, timestamp TEXT,
            water_level_m REAL, rainfall_24h_mm REAL, soil_moisture_pct REAL,
            pressure_hpa REAL, pressure_trend_hpa_per_hr REAL, alert_level TEXT)
    """)

    samples = (storm_rows(rng, args.rows) if args.storm
               else sample_rows(rng, args.rows, args.source))

    now = datetime.now(timezone.utc)
    rows = []
    for i, (wl, rain, soil) in enumerate(samples):
        ts = (now - timedelta(seconds=5 * (args.rows - i))).isoformat()
        level = "ALERT" if wl >= 3.0 else "WATCH" if wl >= 2.0 else "NORMAL"
        # Pressure is inversely correlated with severity: storms ride in on
        # low pressure. Trend is the per-row slope of that same story.
        sev = wl / 4.0
        pressure = float(np.clip(1020 - 42 * sev + rng.normal(0, 1.0), 950, 1040))
        trend = float(np.clip(-3.0 * sev + rng.normal(0, 0.2), -6, 3))
        rows.append(("node-1", ts, wl, rain, soil, pressure, trend, level))

    cur.executemany(
        "INSERT INTO telemetry (node_id, timestamp, water_level_m, "
        "rainfall_24h_mm, soil_moisture_pct, pressure_hpa, "
        "pressure_trend_hpa_per_hr, alert_level) VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    mode = "storm ramp" if args.storm else f"sampled from {os.path.basename(args.source)}"
    print(f"seeded {len(rows)} Wokwi-range telemetry rows ({mode}) -> {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
