# 🌊 TrueFlood — multi-node flood monitoring with an auditable alarm layer

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![MQTT](https://img.shields.io/badge/MQTT-paho%202.1-orange?logo=mqtt&logoColor=white)](https://www.hivemq.com/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn%201.9-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An ESP32 (Wokwi) sensor mesh publishes over MQTT to a Flask bridge that decides
an alert level, stores history, and escalates over WhatsApp/SMS. A dashboard
renders the live state.

What makes it more than a demo is the **separation of concerns in the decision
path**:

| Layer | What it is | Does it fire alarms? |
|---|---|---|
| **1 — Safety rules** | Named constants + hysteresis + corroboration bypass, in `backend/predict.py` | **Yes.** Every alarm comes from here, and every alarm carries a human-readable reason. |
| **2 — ML forecast** | Random Forest, benchmarked against a persistence baseline | No. Advisory only, shown de-emphasised in the UI. |
| **3 — Weather fusion** | Open-Meteo forecast rain folded into Layer 1's reasoning | No. It can add a *reason*, never override one. |

That split is not decoration. It exists because the ML model, when it was
finally measured, **lost to a one-line persistence baseline** — and a model in
that state must not be wired to a siren. The numbers are below, unrounded.

---

## The honest headline

**Demo track** (synthetic dataset, rescaled to the Wokwi sensor ranges),
predicting the flood state 2 steps ahead under blocked `TimeSeriesSplit`:

| Candidate | macro-F1 | std | mean cost |
|---|---|---|---|
| Random Forest | **0.4576** | 0.1668 | 0.9842 |
| Persistence baseline (`ŷ(t+2) = y(t)`) | **0.7068** | 0.1436 | 0.5333 |
| Threshold baseline | 0.7053 | 0.1418 | 0.5584 |

**The model loses to doing nothing, by −0.2493 macro-F1.** The reason is in
[`MODEL_CARD.md`](MODEL_CARD.md): that dataset's label is a deterministic
threshold rule over the very features the model is given, so there is no
forecasting skill to learn — only a rule to memorise. `ml_pipeline/train.py`
**fails the build** (`exit 1`) on this result; the artifact is produced with
an explicit `--no-gate`, and the loss is written into the model card.

**Real-data track** (USGS gauge + Open-Meteo rainfall, label = the NWS's own
published flood stage), predicting a real flood 2 days ahead:

| Candidate | precision | recall | F1 | PR-AUC | accuracy |
|---|---|---|---|---|---|
| Random Forest | 0.8332 | **0.9912** | **0.9016** | 0.9518 | 0.9856 |
| Persistence baseline | 0.8591 | 0.8456 | 0.8519 | — | 0.9813 |

**On real data the model beats persistence by +0.0497 flood-class F1**, at 99%
recall on the class that matters. Site: USGS 05389500, Mississippi River at
McGregor, IA; label `gage_height_ft ≥ 16.0` — the NWS minor flood stage for
gauge `MCGI4`, a threshold published by somebody else for operational reasons
and *not* fitted to this data. Full detail, caveats included, in
[`MODEL_CARD_REAL.md`](MODEL_CARD_REAL.md).

### Two datasets, on purpose

The demo needs data whose *ranges* match what a Wokwi potentiometer can
actually produce (0–4 m of water, not 0–104 m). Real river data does not match
those ranges and never will. Forcing one dataset to serve both purposes is
exactly what produced the original training/serving skew — daily training rows
versus 5-second live packets, and a `water_level_m` that meant something 25×
different at inference time. So there are two tracks: a **demo track**
(synthetic, rescaled to the simulator, honestly labelled weak) that drives the
live system, and a **real-data track** (public hydrology, externally defined
label) that stands alone as a methodology check and never touches the demo.

---

## Architecture

```text
ESP32 nodes (Wokwi, sensors labelled on-canvas)
   │ MQTT  flood/sensor/<node_id>        │ MQTT  flood/status/<node_id>  (retained LWT)
   ▼                                     ▼
┌──────────────────────────────────────────────────────────────────┐
│ Flask + paho bridge  (backend/mqtt_bridge.py)                    │
│                                                                  │
│  Layer 1  SAFETY RULES  ── thresholds + hysteresis               │  ──► alarms
│           (backend/predict.py)   + corroboration bypass          │      + reasons
│                                                                  │
│  Layer 2  ML FORECAST   ── resampled to training cadence,        │      advisory
│           (flood_model.joblib)      OOD-checked vs model card    │      only
│                                                                  │
│  Layer 3  WEATHER       ── Open-Meteo forecast rain, cached 1 h  │      a reason,
│           (backend/weather.py)                                   │      not a veto
└───────┬──────────────────────────────────────────────────────────┘
        ├─► SQLite (absolute path, WAL, indexed) ─► REST API ─► dashboard
        ├─► Prometheus metrics at /api/metrics
        └─► Twilio WhatsApp → SMS fallback (key + allowlist + global budget)

offline, separate:
USGS gauge + Open-Meteo archive + NWS flood stage
   └─► ml_pipeline/train_real.py ─► MODEL_CARD_REAL.md
```

---

## What each Wokwi part represents

Board: `wokwi-esp32-devkit-v1`. The simulator has no soil-moisture model, so
potentiometers stand in. Every part carries a descriptive id and an on-canvas
`wokwi-text` label, so a screen recording is self-explanatory.

| Wokwi part | Simulates | Real hardware | Range | Pin |
|---|---|---|---|---|
| `hcsr04_water_level` | Water level (inverted distance) | Ultrasonic / radar level sensor | 0–4.0 m | D5 / D18 |
| `pot_rainfall` | Rainfall, 24 h | Tipping-bucket rain gauge | 0–200 mm | D34 (ADC1) |
| `pot_soil_moisture` | Ground saturation | Capacitive probe | 0–100 % | D35 (ADC1) |
| `bmp_pressure` | Barometric pressure + trend | BMP280 / BME280 | 950–1040 hPa | I2C 0x77 |
| `dht_humidity` | Humidity / temperature | DHT22 (real, no proxy) | % / °C | D15 |

Plus the output stack: LCD1602 (I2C 0x27), three LEDs (D25/D26/D27) and a
buzzer (D14) — 5 sensing elements and local alerting.

**Analog sensors are on ADC1 (GPIO32–39) by necessity, not preference.** The
ESP32's ADC2 is owned by the WiFi radio: once WiFi is up, `analogRead()` on an
ADC2 pin returns garbage. Since every node publishes over MQTT, ADC2 is
off-limits for sensors, and `firmware/make_diagrams.py` fails generation if a
potentiometer is ever wired to one. GPIO34/35 are additionally input-only with
no internal pull-ups — correct for a potentiometer, wrong for a button.

**A previous hardware revision carried a flow-velocity potentiometer and a
turbidity photoresistor.** Both were removed, and because they were two of the
five features the model consumed, the model was retrained without them rather
than fed silent `0.0` defaults. `discharge_m3s` went too: it was
water level × flow velocity, undefined without a velocity reading — and
mislabelled anyway, since depth × velocity is m²/s, not m³/s. The barometric
pressure trend replaced them in Layer 1, and is a better predictor than
either: it is the only signal that can fire before the water level moves.

Full mapping, wiring and per-state reproduction recipes:
[`firmware/NODES_SETUP.md`](firmware/NODES_SETUP.md).

---

## What was broken, and what fixed it

| Problem | Fix |
|---|---|
| Model shipped without ever being scored; loses to persistence | `ml_pipeline/train.py` benchmarks it and **fails the build**; the loss is published in the model card |
| Training on daily rows, serving 5-second packets | `rescale_dataset.py` + Layer 2 resamples the packet buffer to the training cadence, with an OOD check against `feature_ranges` |
| Float-switch fallback returned 0.3/10/20/55 m while ultrasonic returned 0–4 m — storm and calm packets scored identically | One `TANK_DEPTH_M` for every path in `sketch.ino` |
| `POST /api/alert/send` was unauthenticated — anyone could spend your Twilio credit | Shared key + recipient allowlist + per-node cooldown + **global hourly budget** |
| `on_disconnect` had the wrong paho signature, so `_mqtt_connected` never went false and `/api/health` lied | Correct 5-arg VERSION2 signature; `/api/health` returns **503** when degraded |
| A dead sensor rendered exactly like a healthy one | MQTT Last Will + `NODE_STALE_SEC`; stale nodes render as **NO DATA**, never as a stale alert level |
| `DB_PATH` was relative (two `.db` files got committed) | Absolute, `DB_PATH`-configurable, WAL, indexed, gitignored |
| Turbidity read backwards (more light ⇒ higher NTU) | Inverted: `(4095 - raw) * 1000 / 4095` |
| A `delay(2000)` inside the MQTT callback stalled `mqtt.loop()` | Latched flag rendered from `loop()` |
| Nothing on the Wokwi canvas said which pot was which sensor | Descriptive ids + on-canvas labels, generated by `firmware/make_diagrams.py` |

---

## Layout

```text
├── backend/
│   ├── mqtt_bridge.py       # MQTT ingest, REST API, alerting, metrics
│   ├── predict.py           # Layer 1 SafetyRules + Layer 2 FloodPredictor
│   ├── weather.py           # Layer 3 Open-Meteo forecast fusion
│   └── db.py                # SQLite: absolute path, WAL, indexed
├── ml_pipeline/
│   ├── rescale_dataset.py   # demo data -> Wokwi sensor ranges, labels re-derived
│   ├── train.py             # demo trainer: baselines, CV, cost matrix, gate
│   ├── fetch_usgs.py        # USGS daily values (cached)
│   ├── fetch_openmeteo.py   # Open-Meteo archive at the gauge (cached)
│   ├── train_real.py        # real-data trainer, NWS flood-stage label
│   └── raw/                 # committed API cache, so CI needs no network
├── monitoring/
│   ├── drift.py             # PSI + out-of-training-range, exit 1 on drift
│   └── seed_test_db.py      # seeds a throwaway DB to exercise the monitor
├── firmware/
│   ├── sketch.ino           # ESP32: one water-level scale, LWT, non-blocking LCD
│   ├── make_diagrams.py     # generates all 4 diagrams so they cannot drift
│   └── NODES_SETUP.md       # sensor mapping table + ALERT reproduction recipe
├── frontend/index.html      # dashboard: reasons, staleness, advisory, regional note
├── tests/                   # safety-rule and API tests
├── docker/                  # OPTIONAL container path (not the deploy path)
└── .github/workflows/ci.yml # tests + train-and-gate + real-data track
```

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example backend/.env          # fill in DASHBOARD_KEY, ALERT_RECIPIENTS, Twilio…

python ml_pipeline/rescale_dataset.py # data -> Wokwi ranges (prints before/after)
python ml_pipeline/train.py --no-gate # trains; prints the baseline comparison
python backend/mqtt_bridge.py         # http://localhost:8080
python -m http.server 3000 --directory frontend
```

Real-data track (needs network on first run; afterwards it uses `ml_pipeline/raw/`):

```bash
python ml_pipeline/fetch_usgs.py --site 05389500
python ml_pipeline/fetch_openmeteo.py --site 05389500
python ml_pipeline/train_real.py
```

---

## REST API

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/api/health` | GET | — | **503 when degraded**, 200 when healthy. MQTT state, node counts, model availability |
| `/api/metrics` | GET | — | Prometheus text exposition (no extra dependency) |
| `/api/nodes` | GET | — | All nodes annotated with `stale`, `seconds_since_update`, `reasons`, `forecast_advisory`, `ood_features`, plus a top-level `regional_note` |
| `/api/nodes/<id>` | GET | — | One node + 20-point history |
| `/api/history/<id>` | GET | — | Telemetry trend for charts |
| `/api/alerts/log` | GET | — | Last 50 outbound attempts, channel and Twilio codes |
| `/api/weather/<id>` | GET | — | Layer 3 forecast for a node (needs `NODE_COORDS`) |
| `/api/simulate` | POST | `DEMO_MODE=true` | Inject a reading through the full three-layer pipeline |
| `/api/alert/send` | POST | `X-Dashboard-Key` **+ allowlist** | Send an alert for a node's latest reading |
| `/api/test-alert` | POST | `X-Test-Key` | Fire a message directly, bypassing ML and MQTT |

```bash
# Unauthenticated -> 401, never 200
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8080/api/alert/send \
  -H 'Content-Type: application/json' -d '{"node_id":"x","to":"+911111111111"}'

# Right key, wrong recipient -> 403
curl -s -X POST localhost:8080/api/alert/send -H "X-Dashboard-Key: $DASHBOARD_KEY" \
  -H 'Content-Type: application/json' -d '{"node_id":"x","to":"+15550001111"}'
```

---

## Operations

**Drift monitoring.** `monitoring/drift.py` compares live telemetry against the
model card's `feature_ranges` (out-of-range fraction) and the training
distribution (PSI, decile bins), exiting 1 when either trips. It is the check
that catches training/serving skew on day one — run against the pre-rescale
model card, live Wokwi-range data is 40% out of range on `flow_velocity_ms`
alone. Schedule it daily:

```bash
python monitoring/drift.py --db $DB_PATH --hours 24
```

PSI is reported as `n/a` where the training reference is degenerate (one bin
holding >50% of the mass), because against a near-constant reference any real
sensor noise produces an enormous and meaningless score.

**Deployment.** Plain Python on Render/Railway — no Docker required.
`--workers 1` is load-bearing (the MQTT client and the predictor's hysteresis
state are process-local) and `mqtt_bridge.py` refuses to boot if
`WEB_CONCURRENCY != 1`. `render.yaml` carries every environment variable,
including a commented persistent-disk block, since the free tier's filesystem
is ephemeral; Postgres is the production-scale alternative and `db.py`'s
function signatures are stable enough to swap underneath. `docker/` is kept as
an alternative path for anyone who wants containers.

**Broker.** `broker.hivemq.com` is public — anyone can publish fake sensor data
to your topics, which makes API authentication pointless on its own. Use a
private HiveMQ Cloud cluster (free tier) with `MQTT_USER`/`MQTT_PASS`.

**Twilio sandbox.** The WhatsApp sandbox session expires after 3 days of
inactivity (error 63016). Send `join <your-code>` to +1 415 523 8886. If
WhatsApp delivery fails for any reason other than cooldown or budget, the
bridge falls back to SMS via `SMS_FALLBACK_FROM`.

---

## Testing

```bash
pytest tests/ -v
```

25 tests: the Layer 1 rule engine (calm, storm, rapid rise, compound
saturation, debounce, corroboration bypass, missing-model degradation, slow
de-escalation, Layer 3 fusion, and the invariant that Layer 2 never overrides
Layer 1) plus the API surface (auth gates, allowlist, global budget, the 503
health contract, Prometheus output, staleness, and regional consensus).
