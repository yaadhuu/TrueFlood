"""
Drift monitor: compare live telemetry against the model's training support.

Two independent checks, both cheap and both interpretable:

1. Out-of-range         - what fraction of live values fall outside
                          [min, max] recorded in `model_card.json`'s
                          `feature_ranges`.  This is the check that catches
                          training/serving skew on day one: if the firmware
                          publishes 0-4 m water levels and the model was
                          trained on 0-104 m, every packet is an extrapolation
                          and the model's output is noise.
2. PSI (population      - Population Stability Index against the training
   stability index)       distribution, decile bins.  Conventional reading:
                            PSI < 0.10  stable
                            0.10-0.25   moderate shift, watch it
                            PSI > 0.25  significant shift, retrain

Exits 1 when either check trips, so it can be wired to a scheduled job that
alerts on nonzero exit.

Usage:
    python monitoring/drift.py --db backend/flood_data.db --hours 24
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CARD = os.path.join(REPO_ROOT, "ml_pipeline", "model_card.json")
DEFAULT_DB = os.path.join(REPO_ROOT, "backend", "flood_data.db")

# Matches the post-hardware-refresh telemetry schema. flow_velocity_ms and
# turbidity_ntu were dropped with the sensors that produced them.
FEATURES = ["water_level_m", "rainfall_24h_mm", "soil_moisture_pct",
            "pressure_hpa"]

PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
OOR_FRACTION_MAX = 0.05   # >5% of live values outside training range = alarm
MIN_ROWS = 20


DEGENERATE_MASS = 0.5   # one reference bin holding more than this -> PSI is noise


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index over quantile bins of the training data.

    Returns NaN when the reference distribution is degenerate - i.e. a single
    bin holds more than DEGENERATE_MASS of the training mass.  PSI compares
    bin proportions, so against a near-constant reference any real sensor
    noise splits that spike across bins and produces an enormous score that
    says nothing about drift.  `rainfall_24h_mm` in the synthetic demo dataset
    is exactly this case (~90% of rows share one value).  Reporting "n/a" is
    more useful than reporting a number that is guaranteed to be alarming.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)
    if e_hist.sum() == 0:
        return float("nan")
    if e_hist.max() / e_hist.sum() > DEGENERATE_MASS:
        return float("nan")
    eps = 1e-6
    e_pct = np.maximum(e_hist / max(e_hist.sum(), 1), eps)
    a_pct = np.maximum(a_hist / max(a_hist.sum(), 1), eps)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def load_live(db_path: str, hours: int) -> pd.DataFrame:
    if not os.path.exists(db_path):
        raise SystemExit(f"drift: database not found: {db_path}")
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, " + ", ".join(FEATURES) +
            " FROM telemetry WHERE timestamp >= ?", conn, params=(cutoff,))
    finally:
        conn.close()
    return df


def load_reference(card: dict, card_path: str) -> pd.DataFrame | None:
    """Training rows, for PSI. Optional - range checks work without them."""
    data_file = card.get("data_file")
    if not data_file:
        return None
    path = os.path.join(os.path.dirname(card_path), data_file)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=os.getenv("DB_PATH", DEFAULT_DB))
    ap.add_argument("--card", default=DEFAULT_CARD)
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    with open(args.card, encoding="utf-8") as fh:
        card = json.load(fh)
    ranges = card.get("feature_ranges", {})
    if not ranges:
        print("drift: model card has no feature_ranges - retrain to generate it")
        return 1

    live = load_live(args.db, args.hours)
    ref = load_reference(card, args.card)

    report: dict = {
        "db": args.db,
        "window_hours": args.hours,
        "rows": int(len(live)),
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "features": {},
        "alarms": [],
    }

    if len(live) < MIN_ROWS:
        report["healthy"] = None
        report["alarms"].append(
            f"only {len(live)} telemetry rows in the last {args.hours}h "
            f"(need {MIN_ROWS}) - not enough data to judge")
        _emit(report, args.json)
        return 1

    print("=" * 78)
    print(f"DRIFT REPORT  db={args.db}  window={args.hours}h  rows={len(live)}")
    print("=" * 78)
    print(f"{'feature':<20}{'live min':>11}{'live max':>11}"
          f"{'train range':>22}{'out%':>8}{'PSI':>8}")

    healthy = True
    for feat in FEATURES:
        if feat not in live.columns or feat not in ranges:
            continue
        vals = pd.to_numeric(live[feat], errors="coerce").dropna().values
        if vals.size == 0:
            continue
        lo, hi = float(ranges[feat]["min"]), float(ranges[feat]["max"])
        oor = float(((vals < lo) | (vals > hi)).mean())

        p = float("nan")
        if ref is not None and feat in ref.columns:
            p = psi(pd.to_numeric(ref[feat], errors="coerce").dropna().values, vals)

        report["features"][feat] = {
            "live_min": float(vals.min()), "live_max": float(vals.max()),
            "train_min": lo, "train_max": hi,
            "out_of_range_fraction": oor,
            "psi": None if np.isnan(p) else p,
        }

        print(f"{feat:<20}{vals.min():>11.2f}{vals.max():>11.2f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>22}{oor * 100:>7.1f}%"
              f"{'   n/a' if np.isnan(p) else f'{p:>8.3f}'}")

        if oor > OOR_FRACTION_MAX:
            healthy = False
            report["alarms"].append(
                f"{feat}: {oor:.0%} of live values fall outside the training "
                f"range [{lo:.2f}, {hi:.2f}] - the model is extrapolating")
        if not np.isnan(p) and p > PSI_SIGNIFICANT:
            healthy = False
            report["alarms"].append(
                f"{feat}: PSI {p:.3f} > {PSI_SIGNIFICANT} - significant "
                "distribution shift, retrain")
        elif not np.isnan(p) and p > PSI_MODERATE:
            report["alarms"].append(
                f"{feat}: PSI {p:.3f} > {PSI_MODERATE} - moderate shift, monitor")

    report["healthy"] = healthy
    _emit(report, args.json)
    return 0 if healthy else 1


def _emit(report: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2))
        return
    print()
    if report["alarms"]:
        print("ALARMS")
        for a in report["alarms"]:
            print(f"  - {a}")
    print(f"\nhealthy: {str(report['healthy']).lower()}")


if __name__ == "__main__":
    sys.exit(main())
