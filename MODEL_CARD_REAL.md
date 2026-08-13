# Model Card - FloodSense real-data track

Generated: `2026-08-12T06:41:34.235239+00:00`

## Data source

- USGS site **05389500** - Mississippi River at McGregor, IA
- Coordinates: 43.02701224, -91.1726298
- Range: 2006-01-01 to 2024-12-31 (4714 usable rows)
- USGS parameters: 00060 discharge (cfs), 00065 gage height (ft), daily mean (statCd 00003)
- Weather: Open-Meteo archive: precipitation_sum, temperature_2m_mean at the gauge's own coordinates
- **NWS minor flood stage: 16.0 ft** (gauge `MCGI4`, https://water.noaa.gov/gauges/MCGI4)

## Label

`gage_height_ft >= 16.0 ft, 2 days ahead`

The threshold is published by the National Weather Service for operational flood warning. It was not fitted to this dataset. This is the difference between this track and the demo track, whose label is a threshold rule over its own features.

Class balance: **234 flood days** vs 4480 non-flood days (4.96% positive).

## Results

Blocked TimeSeriesSplit, 5 splits. Metrics are for the **flood class**, which is the class that matters.

| Candidate | precision | recall | F1 | PR-AUC | accuracy |
|---|---|---|---|---|---|
| Random Forest | 0.8332 | 0.9912 | 0.9016 | 0.9518 | 0.9856 |
| Persistence baseline | 0.8591 | 0.8456 | 0.8519 | - | 0.9813 |

Margin: **+0.0497** flood-class F1.

**The model beats persistence** on flood-class F1.

Accuracy is reported last on purpose: with a 5.0% positive rate, predicting "no flood" forever scores 95.0% accuracy and is useless.

Folds scored: 3 of 5. Skipped (single-class): [1, 4].

## Caveats

- The Mississippi at McGregor is a large, slow river: stage moves over days, so today's stage is already a strong 2-day predictor. That is why persistence scores as high as it does, and why the model's margin over it is modest rather than dramatic.
- `stage_margin_ft` is a deterministic function of `gage_height_ft`; it adds no information, only a convenient framing. No feature uses data from after the prediction time.
- Folds containing only one class are skipped, not scored - floods are clustered in time and an early fold can contain none. Skipped folds are listed in `cv.folds_skipped_single_class`.
- This site publishes daily stage but almost no daily discharge (0.6% overlap), so discharge features were dropped at runtime.

## Feature importances

| Feature | Importance |
|---|---|
| `stage_margin_ft` | 0.3497 |
| `gage_height_ft` | 0.3008 |
| `gage_height_lag1` | 0.2357 |
| `gage_delta_3d` | 0.0352 |
| `precip_14d_mm` | 0.0330 |
| `gage_delta_1d` | 0.0239 |
| `temp_mean_c` | 0.0127 |
| `precip_7d_mm` | 0.0061 |

## Scope

This model does **not** run in the live demo. Wokwi potentiometers cannot produce Mississippi River hydrology, and pretending otherwise would recreate exactly the training/serving skew this project was audited for. It stands alone as a methodology check.

scikit-learn `1.9.0`. Reproduce with:

```bash
python ml_pipeline/fetch_usgs.py --site 05389500
python ml_pipeline/fetch_openmeteo.py --site 05389500
python ml_pipeline/train_real.py
```
