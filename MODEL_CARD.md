# Model Card - FloodSense demo track

Generated: `2026-08-13T12:29:14.955436+00:00`  
Data: `flood_dataset_wokwi_scale.csv` (1820 rows, daily (one row per day))  
Task: predict flood state 2 steps ahead  
Validation: blocked TimeSeriesSplit, 5 splits  
scikit-learn: `1.9.0`

## Results

| Candidate | macro-F1 | std | mean cost |
|---|---|---|---|
| model | 0.4543 | 0.1655 | 0.9947 |
| persistence | 0.7068 | 0.1436 | 0.5333 |
| threshold | 0.7246 | 0.1397 | 0.5340 |

Best baseline: **threshold**. Model margin: **-0.2703** macro-F1.

**The model LOSES to the best baseline.** It is shipped as an advisory signal only, and this loss is reported rather than hidden.

## Decision rule

Predictions are not `argmax(proba)`. The predicted class minimises expected cost under this matrix (rows = true, cols = predicted):

```
[[ 0.  1.  2.]
 [ 4.  0.  1.]
 [20.  8.  0.]]
```

Missing an ALERT costs 20; a false ALERT on a NORMAL day costs 2.

## Honest limitations

- The label is a deterministic threshold rule over the same features the model sees, not an observed flood event. High in-sample scores measure rule recovery, not forecasting skill.
- The dataset is synthetic and rescaled to the Wokwi sensor ranges for demo consistency. It is not hydrology.
- This model is ADVISORY ONLY. Alarms are driven by the deterministic Layer-1 SafetyRules in backend/predict.py.

This dataset is synthetic and scale-matched to the Wokwi simulation for demo consistency; see the real-data track ([MODEL_CARD_REAL.md](MODEL_CARD_REAL.md)) for a genuinely benchmarked forecast attempt on public river-gauge data.

## Training feature ranges

Used by `monitoring/drift.py` and by the Layer-2 out-of-distribution check in `backend/predict.py`.

| Feature | min | p01 | mean | p99 | max |
|---|---|---|---|---|---|
| `water_level_m` | 0.000 | 0.002 | 0.766 | 3.998 | 4.000 |
| `rainfall_24h_mm` | 0.000 | 0.000 | 24.660 | 158.229 | 200.000 |
| `soil_moisture_pct` | 18.000 | 26.600 | 62.681 | 97.962 | 99.000 |
