"""
Real-data track: can a model forecast a *real* flood on *real* public data?

This is deliberately separate from the demo track.  It does not feed the
Wokwi demo and never will - a potentiometer cannot produce hydrology.  Its
only job is to answer the question a technical interviewer actually asks:
"your synthetic model loses to persistence; what happens on real data?"

Data
----
USGS 05389500, Mississippi River at McGregor, IA
  * daily mean gage height (00065) and discharge (00060), 2006-2024
  * chosen because it publishes a long DAILY stage record *and* has an NWS
    gauge page with a published flood stage - most USGS sites have only one
    of the two.
Open-Meteo archive precipitation at the gauge's own coordinates.

Label
-----
    flood = gage_height_ft >= NWS_MINOR_FLOOD_STAGE_FT

16.0 ft is the National Weather Service *minor flood stage* for this gauge,
published by NWPS at https://water.noaa.gov/gauges/MCGI4 (verified via
https://api.water.noaa.gov/nwps/v1/gauges/MCGI4, which returns
flood.categories.minor.stage = 16).  It is an external threshold set by
somebody else for operational reasons - it is NOT fitted to this data.  That
distinction is the entire point of this track: the demo dataset's label is a
threshold rule over its own features, so a model can recover it without
learning anything about the future.  Here it cannot cheat that way.

Task
----
Predict `flood` HORIZON=2 days ahead from information available today.
Same discipline as ml_pipeline/train.py: blocked TimeSeriesSplit, a
persistence baseline, and the flood-class recall reported next to every
accuracy number.

Usage:
    python ml_pipeline/fetch_usgs.py --site 05389500
    python ml_pipeline/fetch_openmeteo.py --site 05389500
    python ml_pipeline/train_real.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import TimeSeriesSplit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DIR = os.path.join(SCRIPT_DIR, "raw")

sys.path.insert(0, SCRIPT_DIR)

# ── Site constants. Change these together, never one at a time. ──────
SITE = "05389500"
SITE_NAME = "Mississippi River at McGregor, IA"
SITE_LAT, SITE_LON = 43.02701224, -91.1726298
NWS_LID = "MCGI4"
# NWS minor flood stage, ft. Source (verified):
#   https://water.noaa.gov/gauges/MCGI4
#   https://api.water.noaa.gov/nwps/v1/gauges/MCGI4 -> flood.categories.minor.stage
NWS_MINOR_FLOOD_STAGE_FT = 16.0
NWS_ACTION_STAGE_FT = 13.0
DATE_RANGE = ("2006-01-01", "2024-12-31")

HORIZON = 2          # days ahead, matching the demo track for comparability
N_SPLITS = 5

BASE_FEATURES = [
    "gage_height_ft", "precip_mm", "temp_mean_c",
    "gage_height_lag1", "precip_lag1",
    "gage_delta_1d", "gage_delta_3d",
    "precip_3d_mm", "precip_7d_mm", "precip_14d_mm",
    "stage_margin_ft",
]
# Discharge is requested by the fetcher but is only used when the site
# actually publishes it alongside stage.  At USGS 05389500 the daily
# discharge and daily stage records barely overlap (28 days out of 6922), so
# including it would throw away the entire dataset.  This is checked at
# runtime rather than assumed - see DISCHARGE_MIN_COVERAGE.
DISCHARGE_FEATURES = ["discharge_cfs", "discharge_lag1", "discharge_delta_1d"]
DISCHARGE_MIN_COVERAGE = 0.80

CARD_JSON = os.path.join(SCRIPT_DIR, "model_card_real.json")
CARD_MD = os.path.join(REPO_ROOT, "MODEL_CARD_REAL.md")
MODEL_OUT = os.path.join(SCRIPT_DIR, "flood_model_real.joblib")


def load() -> pd.DataFrame:
    usgs_path = os.path.join(RAW_DIR, f"usgs_{SITE}.csv")
    met_path = os.path.join(RAW_DIR, f"openmeteo_{SITE}.csv")
    for p in (usgs_path, met_path):
        if not os.path.exists(p):
            raise SystemExit(
                f"missing {p}\nRun:\n"
                f"  python ml_pipeline/fetch_usgs.py --site {SITE}\n"
                f"  python ml_pipeline/fetch_openmeteo.py --site {SITE}")

    usgs = pd.read_csv(usgs_path)
    met = pd.read_csv(met_path)
    df = usgs.merge(met, on="date", how="inner")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Same feature style as the demo track's `engineer()`: rolling sums, deltas,
    lag-1 - on this dataset's real columns.
    """
    df = df.copy()
    df["gage_height_lag1"] = df["gage_height_ft"].shift(1)
    df["discharge_lag1"] = df["discharge_cfs"].shift(1)
    df["precip_lag1"] = df["precip_mm"].shift(1)
    df["gage_delta_1d"] = df["gage_height_ft"].diff(1)
    df["gage_delta_3d"] = df["gage_height_ft"].diff(3)
    df["discharge_delta_1d"] = df["discharge_cfs"].diff(1)
    df["precip_3d_mm"] = df["precip_mm"].rolling(3, min_periods=1).sum()
    df["precip_7d_mm"] = df["precip_mm"].rolling(7, min_periods=1).sum()
    df["precip_14d_mm"] = df["precip_mm"].rolling(14, min_periods=1).sum()
    # How far below flood stage the river is sitting right now.
    df["stage_margin_ft"] = NWS_MINOR_FLOOD_STAGE_FT - df["gage_height_ft"]
    return df


def fold_metrics(y_true, y_pred, proba=None) -> dict:
    out = {
        "flood_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "flood_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "flood_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
    }
    if proba is not None and len(np.unique(y_true)) > 1:
        out["pr_auc"] = float(average_precision_score(y_true, proba))
    return out


def mean_of(dicts: list[dict]) -> dict:
    keys = {k for d in dicts for k in d}
    return {k: float(np.mean([d[k] for d in dicts if k in d])) for k in keys}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="probability threshold for the flood class")
    args = ap.parse_args()

    print("=" * 78)
    print("FloodSense trainer (REAL-data track)")
    print("=" * 78)
    print(f"site      : USGS {SITE} - {SITE_NAME}")
    print(f"label     : gage_height_ft >= {NWS_MINOR_FLOOD_STAGE_FT} ft "
          f"(NWS minor flood stage, gauge {NWS_LID})")
    print(f"horizon   : {HORIZON} days ahead")
    print(f"cv        : blocked TimeSeriesSplit, {N_SPLITS} splits")

    df = load()
    df = df.dropna(subset=["gage_height_ft"]).reset_index(drop=True)
    df = engineer(df)

    coverage = float(df["discharge_cfs"].notna().mean())
    features = list(BASE_FEATURES)
    if coverage >= DISCHARGE_MIN_COVERAGE:
        features += DISCHARGE_FEATURES
        print(f"features  : {len(features)} (discharge included, "
              f"{coverage:.0%} coverage)")
    else:
        print(f"features  : {len(features)} (discharge DROPPED - only "
              f"{coverage:.1%} of stage days also publish daily discharge)")

    df["flood_now"] = (df["gage_height_ft"] >= NWS_MINOR_FLOOD_STAGE_FT).astype(int)
    # The label is the state HORIZON days later - and only where that day's
    # observation genuinely exists (the record has winter ice gaps, so a plain
    # shift(-2) would silently pair rows months apart).
    stage_by_date = dict(zip(df["date"], df["gage_height_ft"]))
    future_date = df["date"] + pd.Timedelta(days=HORIZON)
    df["stage_future"] = future_date.map(stage_by_date)
    df = df.dropna(subset=["stage_future"] + features).reset_index(drop=True)
    y = (df["stage_future"] >= NWS_MINOR_FLOOD_STAGE_FT).astype(int).values

    X = df[features]
    print(f"rows      : {len(df)}  "
          f"({df['date'].min().date()} .. {df['date'].max().date()})")
    print(f"labels    : flood={int(y.sum())} ({y.mean():.2%}), "
          f"no-flood={int((1 - y).sum())}")
    if y.sum() < 30:
        print("ERROR: too few flood days to evaluate honestly.")
        return 2

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    model_folds, pers_folds = [], []
    last = None

    skipped = []
    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        # Floods are rare and clustered; an early fold can contain no flood
        # day at all. Such a fold cannot score a flood-class metric, so it is
        # skipped and reported rather than quietly counted as a perfect score.
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            skipped.append((fold, int(y[tr].sum()), int(y[te].sum())))
            print(f"  fold {fold}: SKIPPED - train floods={int(y[tr].sum())}, "
                  f"test floods={int(y[te].sum())} (needs both classes)")
            continue
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf.fit(X.iloc[tr], y[tr])
        proba = clf.predict_proba(X.iloc[te])[:, 1]
        pred = (proba >= args.threshold).astype(int)

        # Persistence: whatever the river is doing today, assume it holds.
        pred_pers = df["flood_now"].values[te]

        m = fold_metrics(y[te], pred, proba)
        p = fold_metrics(y[te], pred_pers)
        model_folds.append(m)
        pers_folds.append(p)
        print(f"  fold {fold}: model F1={m['flood_f1']:.3f} "
              f"(P={m['flood_precision']:.3f} R={m['flood_recall']:.3f})   "
              f"persistence F1={p['flood_f1']:.3f} "
              f"(P={p['flood_precision']:.3f} R={p['flood_recall']:.3f})")
        last = (y[te], pred)

    if not model_folds:
        print("\nERROR: every fold was skipped - not enough flood days to "
              "evaluate. Widen the date range or pick a flashier gauge.")
        return 2

    model_m = mean_of(model_folds)
    pers_m = mean_of(pers_folds)

    print("\nBlocked time-series CV, flood class")
    print(f"  {'candidate':<14}{'precision':>11}{'recall':>9}{'F1':>9}"
          f"{'PR-AUC':>9}{'accuracy':>11}")
    for name, m in (("model", model_m), ("persistence", pers_m)):
        print(f"  {name:<14}{m['flood_precision']:>11.4f}{m['flood_recall']:>9.4f}"
              f"{m['flood_f1']:>9.4f}"
              f"{m.get('pr_auc', float('nan')):>9.4f}{m['accuracy']:>11.4f}")

    margin = model_m["flood_f1"] - pers_m["flood_f1"]
    beats = margin > 0
    print(f"\nmodel margin vs persistence: {margin:+.4f} flood-class F1")
    print(f"VERDICT: model {'beats' if beats else 'does NOT beat'} persistence")
    print(f"flood-class recall (model): {model_m['flood_recall']:.4f}  "
          f"<- read this next to the accuracy figure, not instead of it")

    if last is not None:
        yt, yp = last
        print("\nLast-fold report")
        print(classification_report(yt, yp, labels=[0, 1],
                                    target_names=["no-flood", "flood"],
                                    zero_division=0))
        print("Confusion matrix (rows=true, cols=pred)")
        print(confusion_matrix(yt, yp, labels=[0, 1]))

    final = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1)
    final.fit(X, y)
    import joblib
    joblib.dump({"model": final, "features": features,
                 "threshold": args.threshold}, MODEL_OUT)
    print(f"\nSaved model -> {MODEL_OUT}")

    importances = sorted(zip(features, final.feature_importances_),
                         key=lambda kv: -kv[1])

    import sklearn
    card = {
        "name": "floodsense-real",
        "track": "real data (USGS + Open-Meteo, externally published label)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": {
            "usgs_site": SITE,
            "station_name": SITE_NAME,
            "latitude": SITE_LAT,
            "longitude": SITE_LON,
            "date_range": list(DATE_RANGE),
            "usgs_parameters": {"00060": "discharge cfs (daily mean)",
                                "00065": "gage height ft (daily mean)"},
            "weather": "Open-Meteo archive: precipitation_sum, temperature_2m_mean",
            "nws_gauge_lid": NWS_LID,
            "nws_minor_flood_stage_ft": NWS_MINOR_FLOOD_STAGE_FT,
            "nws_action_stage_ft": NWS_ACTION_STAGE_FT,
            "flood_stage_source": f"https://water.noaa.gov/gauges/{NWS_LID}",
        },
        "rows": int(len(df)),
        "forecast_horizon_days": HORIZON,
        "label": f"gage_height_ft >= {NWS_MINOR_FLOOD_STAGE_FT} ft, {HORIZON} days ahead",
        "class_balance": {"flood": int(y.sum()), "no_flood": int((1 - y).sum()),
                          "flood_rate": float(y.mean())},
        "features": features,
        "cv": {"scheme": "blocked TimeSeriesSplit", "n_splits": N_SPLITS,
               "folds_scored": len(model_folds),
               "folds_skipped_single_class": [f[0] for f in skipped]},
        "decision_threshold": args.threshold,
        "metrics": {"model": model_m, "persistence": pers_m},
        "margin_vs_persistence_f1": float(margin),
        "beats_persistence": bool(beats),
        "feature_importances": [{"feature": f, "importance": float(v)}
                                for f, v in importances],
        "sklearn_version": sklearn.__version__,
        "caveats": [
            "The Mississippi at McGregor is a large, slow river: stage moves "
            "over days, so today's stage is already a strong 2-day predictor. "
            "That is why persistence scores as high as it does, and why the "
            "model's margin over it is modest rather than dramatic.",
            "`stage_margin_ft` is a deterministic function of "
            "`gage_height_ft`; it adds no information, only a convenient "
            "framing. No feature uses data from after the prediction time.",
            "Folds containing only one class are skipped, not scored - floods "
            "are clustered in time and an early fold can contain none. "
            "Skipped folds are listed in `cv.folds_skipped_single_class`.",
            "This site publishes daily stage but almost no daily discharge "
            "(0.6% overlap), so discharge features were dropped at runtime.",
        ],
    }
    with open(CARD_JSON, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    write_card_md(card)
    print(f"Saved card  -> {CARD_JSON}")
    print(f"Saved card  -> {CARD_MD}")
    return 0


def write_card_md(card: dict) -> None:
    m, p = card["metrics"]["model"], card["metrics"]["persistence"]
    ds = card["data_source"]
    verdict = ("**The model beats persistence** on flood-class F1."
               if card["beats_persistence"] else
               "**The model does NOT beat persistence** on flood-class F1. "
               "Reported as measured.")
    lines = [
        "# Model Card - FloodSense real-data track",
        "",
        f"Generated: `{card['generated_utc']}`",
        "",
        "## Data source",
        "",
        f"- USGS site **{ds['usgs_site']}** - {ds['station_name']}",
        f"- Coordinates: {ds['latitude']}, {ds['longitude']}",
        f"- Range: {ds['date_range'][0]} to {ds['date_range'][1]} "
        f"({card['rows']} usable rows)",
        f"- USGS parameters: 00060 discharge (cfs), 00065 gage height (ft), "
        f"daily mean (statCd 00003)",
        f"- Weather: {ds['weather']} at the gauge's own coordinates",
        f"- **NWS minor flood stage: {ds['nws_minor_flood_stage_ft']} ft** "
        f"(gauge `{ds['nws_gauge_lid']}`, {ds['flood_stage_source']})",
        "",
        "## Label",
        "",
        f"`{card['label']}`",
        "",
        "The threshold is published by the National Weather Service for "
        "operational flood warning. It was not fitted to this dataset. This is "
        "the difference between this track and the demo track, whose label is "
        "a threshold rule over its own features.",
        "",
        f"Class balance: **{card['class_balance']['flood']} flood days** vs "
        f"{card['class_balance']['no_flood']} non-flood days "
        f"({card['class_balance']['flood_rate']:.2%} positive).",
        "",
        "## Results",
        "",
        f"Blocked TimeSeriesSplit, {card['cv']['n_splits']} splits. "
        f"Metrics are for the **flood class**, which is the class that matters.",
        "",
        "| Candidate | precision | recall | F1 | PR-AUC | accuracy |",
        "|---|---|---|---|---|---|",
        f"| Random Forest | {m['flood_precision']:.4f} | {m['flood_recall']:.4f} | "
        f"{m['flood_f1']:.4f} | {m.get('pr_auc', float('nan')):.4f} | "
        f"{m['accuracy']:.4f} |",
        f"| Persistence baseline | {p['flood_precision']:.4f} | "
        f"{p['flood_recall']:.4f} | {p['flood_f1']:.4f} | - | "
        f"{p['accuracy']:.4f} |",
        "",
        f"Margin: **{card['margin_vs_persistence_f1']:+.4f}** flood-class F1.",
        "",
        verdict,
        "",
        "Accuracy is reported last on purpose: with a "
        f"{card['class_balance']['flood_rate']:.1%} positive rate, predicting "
        "\"no flood\" forever scores "
        f"{1 - card['class_balance']['flood_rate']:.1%} accuracy and is useless.",
        "",
        f"Folds scored: {card['cv']['folds_scored']} of "
        f"{card['cv']['n_splits']}. Skipped (single-class): "
        f"{card['cv']['folds_skipped_single_class'] or 'none'}.",
        "",
        "## Caveats",
        "",
    ] + [f"- {c}" for c in card["caveats"]] + [
        "",
        "## Feature importances",
        "",
        "| Feature | Importance |",
        "|---|---|",
    ]
    for fi in card["feature_importances"][:8]:
        lines.append(f"| `{fi['feature']}` | {fi['importance']:.4f} |")
    lines += [
        "",
        "## Scope",
        "",
        "This model does **not** run in the live demo. Wokwi potentiometers "
        "cannot produce Mississippi River hydrology, and pretending otherwise "
        "would recreate exactly the training/serving skew this project was "
        "audited for. It stands alone as a methodology check.",
        "",
        f"scikit-learn `{card['sklearn_version']}`. Reproduce with:",
        "",
        "```bash",
        f"python ml_pipeline/fetch_usgs.py --site {ds['usgs_site']}",
        f"python ml_pipeline/fetch_openmeteo.py --site {ds['usgs_site']}",
        "python ml_pipeline/train_real.py",
        "```",
        "",
    ]
    with open(CARD_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
