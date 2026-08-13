"""
FloodSense trainer (demo track).

What this does that the previous trainer did not
------------------------------------------------
1. Computes metrics at all.  The original script fit a RandomForest and
   dumped it without ever scoring it.
2. Benchmarks against two baselines under blocked time-series CV:
     * persistence  -- yhat(t+H) = y(t).  The "do nothing" model.
     * threshold    -- a hand-written rule on water level + rainfall.
   A forecasting model that cannot beat persistence has not learned anything
   about the future; it has learned to copy the present.
3. Applies a cost-sensitive decision rule instead of argmax.  Missing an
   ALERT is far more expensive than raising a false one, so the predicted
   class is the one minimising expected cost under COST_MATRIX.
4. Fails the build (exit 1) when the model does not beat the best baseline,
   unless --no-gate is passed.  This is the check that would have caught the
   original model shipping while losing to a one-liner.
5. Writes model_card.json / MODEL_CARD.md including `feature_ranges`, which
   monitoring/drift.py and backend/predict.py's OOD check both read.

Usage:
    python ml_pipeline/train.py               # gate enforced
    python ml_pipeline/train.py --no-gate     # train anyway, record the loss
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

# Default to the Wokwi-scale dataset (ml_pipeline/rescale_dataset.py).  The
# original CSV stays available for reproducibility via TRAIN_DATA_FILE.
DATA_FILE = os.path.join(
    SCRIPT_DIR, os.getenv("TRAIN_DATA_FILE", "flood_dataset_wokwi_scale.csv")
)
MODEL_OUT = os.path.join(SCRIPT_DIR, "flood_model.joblib")
CARD_JSON = os.path.join(SCRIPT_DIR, "model_card.json")
CARD_MD = os.path.join(REPO_ROOT, "MODEL_CARD.md")

LABEL_NAMES = ["NORMAL", "WATCH", "ALERT"]
FORECAST_HORIZON = 2  # days ahead
N_SPLITS = 5

# Hardware refresh: the flow-velocity and turbidity sensors were removed from
# the node, so those columns no longer exist in live telemetry. They are gone
# from the feature vector rather than being fed silent 0.0 defaults, which
# would have produced confident garbage that looked like it was working.
# discharge_m3s / discharge_lag1 went with them (discharge was
# water_level x flow_velocity, undefined without a velocity reading - and
# mislabelled anyway: depth x velocity is m2/s, not m3/s), as did
# turbidity_spike.
FEATURE_COLS = [
    "water_level_m", "rainfall_24h_mm", "soil_moisture_pct",
    "rainfall_72h_mm", "water_level_change",
    "soil_saturated", "water_level_lag1", "rainfall_lag1",
]
RAW_COLS = [
    "water_level_m", "rainfall_24h_mm", "soil_moisture_pct",
]

# COST_MATRIX[true][pred].  Asymmetric on purpose: a missed ALERT is a flooded
# town, a false ALERT is an annoyed operator.
COST_MATRIX = np.array(
    [
        [0.0, 1.0, 2.0],    # true NORMAL
        [4.0, 0.0, 1.0],    # true WATCH
        [20.0, 8.0, 0.0],   # true ALERT
    ]
)


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the modelling features. Mirrored by backend/predict.py."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["rainfall_72h_mm"] = df["rainfall_24h_mm"].rolling(3, min_periods=1).sum()
    df["water_level_change"] = df["water_level_m"].diff().fillna(0.0)
    df["soil_saturated"] = (df["soil_moisture_pct"] > 85).astype(int)
    df["water_level_lag1"] = df["water_level_m"].shift(1).bfill()
    df["rainfall_lag1"] = df["rainfall_24h_mm"].shift(1).bfill()
    return df


def cost_sensitive_predict(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Pick the class minimising expected cost, not the most likely class."""
    full = np.zeros((proba.shape[0], len(LABEL_NAMES)))
    for i, c in enumerate(classes):
        full[:, int(c)] = proba[:, i]
    expected = full @ COST_MATRIX  # [n, pred] expected cost of each prediction
    return expected.argmin(axis=1)


def threshold_baseline(X: pd.DataFrame) -> np.ndarray:
    """
    Hand-written rule on the current reading. No learning involved.

    Rewritten for the post-hardware-refresh feature set: the old version keyed
    off turbidity, which is no longer a column, so it raised KeyError rather
    than quietly scoring wrong. Rainfall stands in as the second signal.
    """
    wl = X["water_level_m"].values
    rain = X["rainfall_24h_mm"].values
    out = np.zeros(len(X), dtype=int)
    out[wl > 0.10] = 1
    out[(wl > 0.10) & (rain > 60.0)] = 2
    out[wl > 1.67] = 2
    return out


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-gate", action="store_true",
                    help="train and save even if the model loses to a baseline")
    ap.add_argument("--data", default=DATA_FILE)
    ap.add_argument("--model-out", default=MODEL_OUT)
    ap.add_argument("--card-json", default=CARD_JSON)
    ap.add_argument("--card-md", default=CARD_MD)
    args = ap.parse_args()

    print("=" * 78)
    print("FloodSense trainer (demo track)")
    print("=" * 78)
    print(f"data      : {args.data}")
    print(f"horizon   : predict state {FORECAST_HORIZON} steps ahead")
    print(f"cv        : blocked TimeSeriesSplit, {N_SPLITS} splits")

    if not os.path.exists(args.data):
        print(f"\nERROR: {args.data} not found. Run ml_pipeline/rescale_dataset.py first.")
        return 2

    df = pd.read_csv(args.data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = engineer(df)

    # Forecast target: the state FORECAST_HORIZON steps in the future.
    df["y_future"] = df["flood_event"].shift(-FORECAST_HORIZON)
    df["y_now"] = df["flood_event"]
    df = df.dropna(subset=FEATURE_COLS + ["y_future"]).reset_index(drop=True)

    X = df[FEATURE_COLS]
    y = df["y_future"].astype(int).values
    y_now = df["y_now"].astype(int).values

    print(f"rows      : {len(df)}")
    print(f"labels    : " + ", ".join(
        f"{LABEL_NAMES[k]}={int((y == k).sum())}" for k in range(3)))

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = {"model": [], "persistence": [], "threshold": []}
    costs = {"model": [], "persistence": [], "threshold": []}
    last_fold = None

    for fold, (tr, te) in enumerate(tscv.split(X), start=1):
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        clf.fit(X.iloc[tr], y[tr])
        proba = clf.predict_proba(X.iloc[te])
        pred_model = cost_sensitive_predict(proba, clf.classes_)

        pred_pers = y_now[te]                       # yhat(t+H) = y(t)
        pred_thr = threshold_baseline(X.iloc[te])

        for name, pred in (("model", pred_model),
                           ("persistence", pred_pers),
                           ("threshold", pred_thr)):
            scores[name].append(macro_f1(y[te], pred))
            costs[name].append(float(COST_MATRIX[y[te], pred].mean()))

        print(f"  fold {fold}: model={scores['model'][-1]:.4f}  "
              f"persistence={scores['persistence'][-1]:.4f}  "
              f"threshold={scores['threshold'][-1]:.4f}")
        last_fold = (y[te], pred_model)

    summary = {
        name: {
            "macro_f1_mean": float(np.mean(v)),
            "macro_f1_std": float(np.std(v)),
            "mean_cost": float(np.mean(costs[name])),
        }
        for name, v in scores.items()
    }

    print("\nBlocked time-series CV results")
    print(f"  {'candidate':<14}{'macro-F1':>10}{'std':>8}{'mean cost':>12}")
    for name in ("model", "persistence", "threshold"):
        s = summary[name]
        print(f"  {name:<14}{s['macro_f1_mean']:>10.4f}{s['macro_f1_std']:>8.4f}"
              f"{s['mean_cost']:>12.4f}")

    best_baseline = max(("persistence", "threshold"),
                        key=lambda n: summary[n]["macro_f1_mean"])
    margin = summary["model"]["macro_f1_mean"] - summary[best_baseline]["macro_f1_mean"]
    beats = margin > 0.0

    print(f"\nbest baseline : {best_baseline} "
          f"({summary[best_baseline]['macro_f1_mean']:.4f})")
    print(f"model margin  : {margin:+.4f} macro-F1")
    print(f"GATE          : {'PASS' if beats else 'FAIL'} "
          f"(model {'beats' if beats else 'does NOT beat'} the best baseline)")

    if last_fold is not None:
        yt, yp = last_fold
        print("\nLast-fold classification report (cost-sensitive decisions)")
        print(classification_report(yt, yp, labels=[0, 1, 2],
                                    target_names=LABEL_NAMES, zero_division=0))
        print("Confusion matrix (rows=true, cols=pred)")
        print(confusion_matrix(yt, yp, labels=[0, 1, 2]))

    if not beats and not args.no_gate:
        print("\nBuild failed: model loses to the baseline. "
              "Re-run with --no-gate to save it anyway (and say so in the model card).")
        return 1

    # Final fit on all data for the shipped artifact.
    final = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    final.fit(X, y)
    joblib.dump(
        {"model": final, "features": FEATURE_COLS, "label_names": LABEL_NAMES,
         "cost_matrix": COST_MATRIX.tolist()},
        args.model_out,
    )
    print(f"\nSaved model -> {args.model_out}")

    feature_ranges = {
        c: {"min": float(df[c].min()), "max": float(df[c].max()),
            "p01": float(df[c].quantile(0.01)), "p99": float(df[c].quantile(0.99)),
            "mean": float(df[c].mean()), "std": float(df[c].std())}
        for c in RAW_COLS
    }

    import sklearn  # local import: only needed for the version stamp
    card = {
        "name": "floodsense-demo",
        "track": "demo (synthetic, Wokwi-scale)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "data_file": os.path.basename(args.data),
        "rows": int(len(df)),
        "forecast_horizon_steps": FORECAST_HORIZON,
        "cadence": "daily (one row per day)",
        "features": FEATURE_COLS,
        "label_names": LABEL_NAMES,
        "class_counts": {LABEL_NAMES[k]: int((y == k).sum()) for k in range(3)},
        "cv": {"scheme": "blocked TimeSeriesSplit", "n_splits": N_SPLITS},
        "metrics": summary,
        "best_baseline": best_baseline,
        "margin_vs_best_baseline": float(margin),
        "gate_passed": bool(beats),
        "gate_enforced": not args.no_gate,
        "cost_matrix": COST_MATRIX.tolist(),
        "decision_rule": "expected-cost minimisation over COST_MATRIX",
        "feature_ranges": feature_ranges,
        "sklearn_version": sklearn.__version__,
        "limitations": [
            "The label is a deterministic threshold rule over the same features "
            "the model sees, not an observed flood event. High in-sample scores "
            "measure rule recovery, not forecasting skill.",
            "The dataset is synthetic and rescaled to the Wokwi sensor ranges "
            "for demo consistency. It is not hydrology.",
            "This model is ADVISORY ONLY. Alarms are driven by the "
            "deterministic Layer-1 SafetyRules in backend/predict.py.",
        ],
    }
    with open(args.card_json, "w", encoding="utf-8") as fh:
        json.dump(card, fh, indent=2)
    print(f"Saved card  -> {args.card_json}")

    write_markdown_card(card, args.card_md)
    print(f"Saved card  -> {args.card_md}")
    return 0


def write_markdown_card(card: dict, path: str = CARD_MD) -> None:
    m = card["metrics"]
    verdict = (
        "**The model beats the best baseline.**"
        if card["gate_passed"]
        else "**The model LOSES to the best baseline.** It is shipped as an "
             "advisory signal only, and this loss is reported rather than hidden."
    )
    lines = [
        "# Model Card - FloodSense demo track",
        "",
        f"Generated: `{card['generated_utc']}`  ",
        f"Data: `{card['data_file']}` ({card['rows']} rows, {card['cadence']})  ",
        f"Task: predict flood state {card['forecast_horizon_steps']} steps ahead  ",
        f"Validation: {card['cv']['scheme']}, {card['cv']['n_splits']} splits  ",
        f"scikit-learn: `{card['sklearn_version']}`",
        "",
        "## Results",
        "",
        "| Candidate | macro-F1 | std | mean cost |",
        "|---|---|---|---|",
    ]
    for name in ("model", "persistence", "threshold"):
        lines.append(
            f"| {name} | {m[name]['macro_f1_mean']:.4f} | "
            f"{m[name]['macro_f1_std']:.4f} | {m[name]['mean_cost']:.4f} |"
        )
    lines += [
        "",
        f"Best baseline: **{card['best_baseline']}**. "
        f"Model margin: **{card['margin_vs_best_baseline']:+.4f}** macro-F1.",
        "",
        verdict,
        "",
        "## Decision rule",
        "",
        "Predictions are not `argmax(proba)`. The predicted class minimises "
        "expected cost under this matrix (rows = true, cols = predicted):",
        "",
        "```",
        str(np.array(card["cost_matrix"])),
        "```",
        "",
        "Missing an ALERT costs 20; a false ALERT on a NORMAL day costs 2.",
        "",
        "## Honest limitations",
        "",
    ]
    lines += [f"- {x}" for x in card["limitations"]]
    lines += [
        "",
        "This dataset is synthetic and scale-matched to the Wokwi simulation for "
        "demo consistency; see the real-data track "
        "([MODEL_CARD_REAL.md](MODEL_CARD_REAL.md)) for a genuinely benchmarked "
        "forecast attempt on public river-gauge data.",
        "",
        "## Training feature ranges",
        "",
        "Used by `monitoring/drift.py` and by the Layer-2 out-of-distribution "
        "check in `backend/predict.py`.",
        "",
        "| Feature | min | p01 | mean | p99 | max |",
        "|---|---|---|---|---|---|",
    ]
    for feat, r in card["feature_ranges"].items():
        lines.append(
            f"| `{feat}` | {r['min']:.3f} | {r['p01']:.3f} | {r['mean']:.3f} | "
            f"{r['p99']:.3f} | {r['max']:.3f} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(main())
