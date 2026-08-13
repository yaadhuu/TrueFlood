# FloodSense / TrueFlood — Full Implementation Plan

**Audience:** Claude Code, working autonomously on this repo end-to-end.
**Repo:** `yaadhuu/TrueFlood` (Python/Flask + MQTT + scikit-learn + ESP32/Wokwi + vanilla JS frontend).
**Do not stop to ask the human clarifying questions.** Every decision needed to
execute this plan is made below. Where you must pick a concrete value
(a threshold, a port, a filename) that isn't specified, pick the most
conventional option, write down what you picked and why in the relevant
model card / README, and move on.

**Ground rule that overrides convenience:** never report a check as passing
without having actually run it and seen the output. This project's current
state is a demo whose central claim (an ML model beats a naive baseline)
turned out to be false when actually measured. Do not repeat that mistake in
the other direction — if a gate fails, say so in the model card and ship the
honest number. A documented failure is a stronger deliverable than a silent
success.

---

## 0. What's wrong today (context, condensed from a full audit)

1. **The ML model's labels are a deterministic threshold rule**, not observed
   flood events. A depth-3 decision tree on 5 raw features reproduces the
   label at 99.1%. On a proper forecast task (predict state 2 days ahead,
   blocked time-series CV), a one-line **persistence baseline** (`ŷ(t+2)=y(t)`)
   scores macro-F1 ≈ 0.72; the trained Random Forest scores ≈ 0.65. **The
   model currently loses to doing nothing.**
2. **Training/serving feature skew.** Training data is daily; the ESP32
   publishes every 5 seconds. Features like `rainfall_72h_mm` and the `*_lag1`
   columns mean something ~17,280x different at serving time than at training
   time.
3. **The firmware's water-level scale is internally inconsistent.** The
   ultrasonic path returns 0–4 m; the float-switch fallback returns
   0.3/10/20/55 m. The training data's ALERT class averages 48 m. Realistic
   storm packets and calm fallback packets currently produce the *same*
   predicted alert level.
4. **`POST /api/alert/send` has no authentication** — anyone can make the
   server send a WhatsApp message to an arbitrary number via your Twilio
   account. This is the one item that costs real money if skipped.
5. `paho` `on_disconnect` callback signature is wrong for `CallbackAPIVersion.VERSION2`
   (5 args expected, 4 given) → raises on every disconnect →
   `_mqtt_connected` never flips false → `/api/health` reports healthy while
   the bridge is deaf.
6. No node liveness/staleness detection — a dead sensor renders identically
   to a healthy one.
7. `DB_PATH = "flood_data.db"` is a relative path (two copies already
   committed to git as a result) and Render's free-tier filesystem is
   ephemeral.
8. The Wokwi diagrams technically include many parts (potentiometers,
   photoresistor, ultrasonic, DHT22, 3 pushbuttons, LCD, LEDs, buzzer) but
   nothing on the canvas identifies *which* simulated sensor each part
   represents — anyone watching a demo just sees "ESP32 + some pots." This
   needs fixing so the sensor-fusion story (soil moisture, rainfall,
   turbidity, flow velocity, water level, temperature/humidity) is legible on
   sight.

Full detail lives in `AUDIT.md` / `PATCHES.md` if present in the repo from a
prior review; if absent, this plan is self-sufficient — proceed directly.

---

## 1. Target architecture

**Two-layer prediction, one dataset strategy split into two tracks, sensors
that are unmistakably labeled, and a deploy path that works with or without
Docker.**

```
ESP32 (Wokwi, multi-sensor, clearly labeled)
   │ MQTT (flood/sensor/<node_id>)
   ▼
Flask/paho bridge
   │
   ├─► Layer 1 — SAFETY RULES (deterministic, auditable)  ──► fires alarms
   │     thresholds + hysteresis + corroboration bypass
   │
   ├─► Layer 2 — ML FORECAST (advisory only, never alarm-driving)
   │     trained on a DEMO dataset rescaled to match the Wokwi sensor ranges
   │
   ├─► Layer 3 — WEATHER FUSION (advisory, new)
   │     Open-Meteo forecast API: rain expected in next 24-48h at the node's
   │     lat/lon, folded into the safety-rule reasons as corroborating context
   │
   ├─► SQLite (absolute path, WAL, indexed) ──► REST API ──► frontend
   └─► Twilio WhatsApp (authenticated, allowlisted, rate-budgeted) + SMS fallback

separately, offline:
REAL DATA TRACK (USGS gauge + Open-Meteo historical rainfall + NWS flood
stage) ──► train_real.py ──► model_card_real.json
This does NOT feed the live demo. It is a standalone, honestly-benchmarked
proof of methodology for the README / resume, because Wokwi sliders cannot
generate real hydrology no matter what you train on.
```

Why two dataset tracks instead of one: the demo needs data whose *ranges*
match what a slider in Wokwi can actually produce (0–4 m, not 0–104 m). Real
river data doesn't match those ranges and never will. Trying to force one
dataset to serve both purposes is what created problem #2 and #3 above. Keep
them separate and be explicit in the docs about which is which.

---

## 2. Reference implementations (already built and tested — port these in verbatim, adapt only file paths)

These five files were written and validated in a prior session (8/8 tests
passed, trainer correctly fails its own gate on the current data, drift
monitor correctly flags real skew). Treat them as ground truth for their
respective concerns; do not redesign them, only integrate and extend.

### 2.1 `ml_pipeline/train.py`
Rewritten trainer. Reports metrics (the original computed none), benchmarks
against a persistence baseline and a threshold baseline via blocked
`TimeSeriesSplit` CV, uses a cost-sensitive decision rule (missing an ALERT
costs far more than a false alarm), fails the build (`exit 1`) if the model
doesn't beat the baseline unless `--no-gate` is passed, and writes
`model_card.json` + `MODEL_CARD.md` including `feature_ranges` (needed by the
drift monitor and by Layer 2's OOD check).

### 2.2 `backend/predict.py`
Two-layer predictor. `SafetyRules` is a named-constant, auditable threshold
engine with hysteresis (`ESCALATE_AFTER_SEC=10`, `DEESCALATE_AFTER_SEC=600`)
and a corroboration bypass (≥2 independent trigger reasons skip the debounce
and escalate immediately — a real flood shouldn't wait out a noise filter).
`FloodPredictor` resamples the packet buffer to the model's training cadence
before calling Layer 2, runs an OOD check against `model_card.json`'s
`feature_ranges`, and returns Layer 1's decision as `status` with Layer 2's
output as `forecast_advisory` only. Thread-safe (`threading.RLock`),
degrades safely to Layer-1-only if the model file is missing.

### 2.3 `monitoring/drift.py`
PSI (population stability index) + out-of-training-range check comparing live
`telemetry` rows against `model_card.json`'s `feature_ranges`. Exits 1 and
prints human-readable alarms if drift is detected. Verified to correctly flag
79% of rainfall values and 59% of velocity values as out-of-range when fed
simulated realistic firmware output against the original (unrescaled)
dataset — this is exactly the check that would have caught problem #2 on day
one.

### 2.4 `tests/test_safety_rules.py`
8 tests covering: calm→NORMAL, storm→ALERT with multiple reasons, rapid-rise
alone triggers ALERT, compound saturated-soil+rain escalates, every alert
carries a human-readable reason, single-sensor spikes are debounced,
corroborated alerts are immediate, missing model still fails safe (alarms via
Layer 1, doesn't silently return NORMAL). All passing — keep them passing
after every change in this plan; add new tests alongside, don't replace these.

### 2.5 `docker/Dockerfile`, `docker/docker-compose.yml`, `.github/workflows/ci.yml`
Docker is **optional** per the human's request (see Phase 6) — keep these
working for anyone who wants them, but they are not the deploy path. The CI
`train-and-gate` job (plain `pip install` + `python ml_pipeline/train.py`, no
Docker) is the one piece of CI that must always run regardless of the deploy
target — it is the check that would have caught the original model's failure
before it shipped.

**If any of these five files/directories are not already present in the
repo, recreate them exactly per the descriptions above before proceeding —
everything else in this plan depends on them.**

---

## 3. Phase 1 — Dataset rescale (demo track)

**Goal:** the synthetic training data and the Wokwi firmware's sensor ranges
describe the same physical scale, so Layer 2's OOD check stops firing on
every packet and the model's (advisory, honestly-labeled-as-weak) output is
at least self-consistent.

Create `ml_pipeline/rescale_dataset.py`:

- Load `flood_dataset_2021_final.csv`.
- Rescale each raw column to the Wokwi sensor's physical range using a
  **percentile-preserving** transform (map p0.5→sensor_min, p99.5→sensor_max,
  clip outliers), not a naive min-max — a naive min-max lets one outlier
  compress the whole useful range.
  - `water_level_m`: → 0–4 m (matches `TANK_DEPTH_M` in the firmware patch,
    §5 below)
  - `rainfall_24h_mm`: → 0–200 mm (matches `readRainfall()`'s ADC mapping)
  - `soil_moisture_pct`: already 0–100, leave as is
  - `flow_velocity_ms`: → 0–10 m/s (matches `readFlowVelocity()`)
  - `turbidity_ntu`: → 0–1000 NTU (matches `readTurbidity()`)
- Recompute `discharge_proxy`, `rainfall_72h_mm`, the deltas, and the lag
  columns from the *rescaled* values (order matters — rescale raw columns
  first, then re-derive).
- **Re-derive `flood_event` from the rescaled `water_level_m` and
  `turbidity_ntu`** using thresholds proportional to the new range (e.g. the
  original 1.5 m / 25.35 m breakpoints on a 0–104 m range become
  proportionally `1.5/104*4 ≈ 0.058 m` / `25.35/104*4 ≈ 0.975 m` on the new
  0–4 m range) — do not just keep the old 0/1/2 label sitting on top of new
  feature values, that reintroduces a mismatch of a different kind.
- Save as `ml_pipeline/flood_dataset_wokwi_scale.csv`.
- Print a before/after range table for every column so the transform is
  auditable in the CI log.

Update `ml_pipeline/train.py`'s `DATA_FILE` to default to the new rescaled
CSV via an env var (`TRAIN_DATA_FILE`, default
`flood_dataset_wokwi_scale.csv`), keeping the original CSV path available for
reference/reproducibility.

**Also update `backend/predict.py`'s `SafetyRules` class docstring** to note
it was already calibrated to the 0–4 m Wokwi range independently of this
dataset work — confirm the two now agree (`WL_ALERT_M = 3.0` should sit
comfortably inside the rescaled ALERT band; adjust if the rescale produces a
different breakpoint and document why in a code comment).

**Acceptance:**
```bash
python ml_pipeline/rescale_dataset.py     # prints before/after ranges
python ml_pipeline/train.py               # trains on the rescaled data
python monitoring/drift.py --db <fresh test db seeded with 0-4m/0-200mm/etc rows>
# drift.py must report healthy: true against realistic Wokwi-range input
# (it correctly reported healthy: false against the OLD dataset — that
#  flip is the acceptance signal for this phase)
```
The trainer will likely **still fail its baseline gate** on the rescaled
data — that's expected and fine, rescaling fixes the range mismatch, not the
label-leakage problem. Run with `--no-gate` to get a demo artifact, and say
so plainly in `MODEL_CARD.md`: *"This dataset is synthetic and scale-matched
to the Wokwi simulation for demo consistency; see the real-data track for a
genuinely benchmarked forecast attempt."*

---

## 4. Phase 2 — Real-data validation track (resume/credibility track)

**Goal:** a second, independent, honestly-benchmarked model trained on real
public river-gauge and rainfall data, with a real (not fitted) definition of
"flood," clearly documented as a standalone methodology demonstration that
does **not** drive the live Wokwi demo. This is the artifact that survives a
technical interview question.

### 4.1 `ml_pipeline/fetch_usgs.py`

USGS Water Services `dv` (daily values) endpoint, no API key required:

```
GET https://waterservices.usgs.gov/nwis/dv/?format=json&sites=<SITE>&startDT=<YYYY-MM-DD>&endDT=<YYYY-MM-DD>&parameterCd=00060,00065&statCd=00003
```
- `parameterCd=00060` = discharge (cfs), `00065` = gage height (ft),
  `statCd=00003` = daily mean.
- Response is nested JSON (`value.timeSeries[i].values[0].value[]` gives
  `{value, dateTime, qualifiers}` per series). Parse into a flat
  `date, discharge_cfs, gage_height_ft` dataframe, one row per day, joining
  the two parameter series on date.
- Default site: pick a USGS gauge with (a) a long daily record (10+ years),
  (b) a publicly documented NWS flood stage. Verify site choice via
  `https://waterservices.usgs.gov/nwis/site/?format=rdb&sites=<SITE>` and
  cross-check the flood stage at `https://water.noaa.gov/gauges/<SITE>`
  (NOAA's AHPS pages publish official action/minor/moderate/major flood
  stage thresholds per gauge — this is the external, non-fitted label source
  that avoids the leakage problem in §0.1). Document the chosen site number,
  its name, and its official flood stage (in feet) as constants at the top
  of `train_real.py` with a comment linking the AHPS page you verified it on.
- Handle the API's per-request date-range limits by chunking requests
  (e.g. 3-year windows) and concatenating.
- Cache the raw fetch to `ml_pipeline/raw/usgs_<site>.csv` so re-runs (e.g. in
  CI) don't hammer the API — check for the cache file first, only fetch if
  missing or `--refresh` is passed.

### 4.2 `ml_pipeline/fetch_openmeteo.py`

Open-Meteo Historical Weather API, free, no key:
```
GET https://archive-api.open-meteo.com/v1/archive?latitude=<LAT>&longitude=<LON>&start_date=<YYYY-MM-DD>&end_date=<YYYY-MM-DD>&daily=precipitation_sum,temperature_2m_mean&timezone=UTC
```
- Use the lat/lon of the chosen USGS gauge (from its site metadata,
  `geoLocation` in the site service response).
- Same caching pattern as above:
  `ml_pipeline/raw/openmeteo_<site>.csv`.

### 4.3 `ml_pipeline/train_real.py`

- Merge USGS + Open-Meteo on date.
- Label: `flood = gage_height_ft >= NWS_FLOOD_STAGE_FT` (the externally
  published threshold from §4.1 — **not** a threshold you fit yourself; that
  distinction is the whole point of this track).
- Forecast horizon: predict `flood` `N` days ahead (`N=2`, matching the demo
  track for comparability) using lagged discharge, gage height, and
  precipitation features (mirror the feature-engineering style of
  `ml_pipeline/train.py`'s `engineer()` — rolling sums, deltas, lag-1 — but
  on this dataset's actual columns).
- Same evaluation discipline as `train.py`: blocked `TimeSeriesSplit`,
  persistence baseline, cost-sensitive decision rule if the label is
  multi-class, or a precision/recall/PR-AUC report if binary (binary is fine
  and arguably more honest here — don't force a 3-class scheme onto real data
  that doesn't naturally have one).
- Handle severe class imbalance (floods are rare) explicitly: report
  precision/recall/F1 for the flood class specifically, not just accuracy;
  consider `class_weight="balanced"`; do **not** claim a high accuracy number
  without also reporting recall on the flood class next to it.
- Save `ml_pipeline/model_card_real.json` / `MODEL_CARD_REAL.md` in the same
  shape as the demo track's model card, plus a `"data_source"` block
  recording the site number, name, coordinates, date range fetched, and the
  NWS flood stage used, so the result is independently checkable.
- **Whatever the result is, report it as-is.** If the real model also loses
  to persistence, that's still a legitimate and interesting finding — say so.
  If it wins, that's the headline result for the README.

**Acceptance:**
```bash
python ml_pipeline/fetch_usgs.py --site <chosen site>
python ml_pipeline/fetch_openmeteo.py --site <chosen site>
python ml_pipeline/train_real.py
# must print a persistence-baseline-vs-model comparison table, same discipline
# as train.py, and must print the flood-class recall explicitly
```

### 4.4 README section

Add a "Two datasets, on purpose" section to `README.md` explaining the split
in 3–4 sentences, linking both model cards, and stating the real-data
benchmark numbers directly in the table (not just "see model card").

---

## 5. Phase 3 — Firmware: fix the scale bug, make every sensor legible, don't block MQTT

### 5.1 Water-level scale (BLOCKER — fixes the "storm and calm look identical" bug)

```c
#define TANK_DEPTH_M   4.0f     // single source of truth for the level scale

float readWaterLevelSwitches() {
  // SAME 0..TANK_DEPTH_M scale as the ultrasonic path — this is the fix.
  if (digitalRead(PIN_SW3)) return TANK_DEPTH_M * 0.90f;   // 3.60 m
  if (digitalRead(PIN_SW2)) return TANK_DEPTH_M * 0.55f;   // 2.20 m
  if (digitalRead(PIN_SW1)) return TANK_DEPTH_M * 0.25f;   // 1.00 m
  return 0.05f;
}

float readWaterLevel() {
  return constrain(readWaterLevelUltrasonic(), 0.0f, TANK_DEPTH_M);
}
```
Publish which path produced the reading so the backend can log it:
`doc["sensor_source"] = usedFallback ? "float_switch" : "ultrasonic";`

### 5.2 Non-blocking alert display

Replace the `delay(2000)` inside `mqttCallback` (it stalls `mqtt.loop()` and
risks keepalive timeouts) with a latched flag rendered from `loop()`:
```c
String pendingAlertMsg = ""; unsigned long alertMsgUntil = 0;
// in mqttCallback: pendingAlertMsg = alertLevel; alertMsgUntil = millis()+2000;
// in loop(): if (millis() < alertMsgUntil) show alert screen; else show normal screen;
```

### 5.3 Last Will and Testament (fixes silent-death nodes, backend §6.3 depends on this)

```c
char TOPIC_STATUS[64];
snprintf(TOPIC_STATUS, sizeof(TOPIC_STATUS), "flood/status/%s", NODE_ID);

if (mqtt.connect(CLIENT_ID, MQTT_USER, MQTT_PASS,
                 TOPIC_STATUS, /*qos*/1, /*retain*/true, "offline")) {
  mqtt.publish(TOPIC_STATUS, "online", true);
  mqtt.subscribe(TOPIC_ALERT);
}
```

### 5.4 Make every simulated sensor unmistakable — this is what the human explicitly asked for

The current diagrams have the right *parts* (potentiometers, photoresistor,
ultrasonic, DHT22, pushbuttons) but nothing visually says which pot is
"rainfall" vs "soil moisture" vs "flow velocity." Fix by:

1. **Research the current Wokwi `diagram.json` schema** (fetch
   `https://docs.wokwi.com/diagram-format` — schema/label support changes
   between Wokwi versions, don't assume). Check whether the part types in use
   (`wokwi-potentiometer`, `wokwi-photoresistor-sensor`, `wokwi-hc-sr04`,
   `wokwi-pushbutton`) support a `"label"` or `"attrs": {"label": "..."}`
   field that renders on-canvas.
2. **At minimum** (guaranteed to work regardless of label support): give
   every part a descriptive `id`, since Wokwi shows the `id` on hover/click
   and it's what appears in any generated wiring diagram or BOM export:
   - `pot_soil_moisture`, `pot_rainfall`, `pot_flow_velocity`
   - `ldr_turbidity`
   - `sw_water_low`, `sw_water_mid`, `sw_water_high`
   - `hcsr04_water_level`
   - `dht22_temp_humidity`
3. **If label rendering is supported**, add on-canvas text so a screen
   recording of the simulation is self-explanatory without narration:
   `"Soil Moisture (capacitive sensor sim)"`, `"Rain Gauge (tipping-bucket
   sim)"`, `"Flow Velocity (YF-S201 sim)"`, `"Turbidity (LDR proxy)"`.
4. Reposition parts on the canvas into a labeled grid (water-level group,
   weather group, flow group) rather than the current scattered layout, so a
   viewer can read the diagram top-to-bottom as "here's how we simulate a
   flood monitoring station."
5. Rewrite `firmware/NODES_SETUP.md` with an explicit sensor-mapping table:

   | Wokwi part | Simulates | Real hardware it stands in for |
   |---|---|---|
   | HC-SR04 ultrasonic | Water level (primary) | Ultrasonic or radar level sensor |
   | 3x pushbutton (float switches) | Water level (fallback) | Mechanical float switches |
   | Potentiometer #1 | Soil moisture | Capacitive soil moisture probe |
   | Potentiometer #2 | Rainfall (24h accumulation) | Tipping-bucket rain gauge |
   | Potentiometer #3 | Flow velocity | YF-S201 hall-effect flow sensor |
   | Photoresistor (LDR) | Turbidity | Optical turbidity sensor (note: inverted — see §5.5) |
   | DHT22 | Temperature / humidity | DHT22 (this one's real, no proxy needed) |

   State plainly in this file: *"This is a Wokwi simulation. The
   potentiometers and photoresistor stand in for sensors that don't have
   Wokwi models; sliding them simulates a live sensor reading changing. The
   mapping above is the ground truth for what each control represents."*
   This turns the "it just looks like ESP32+DHT22" problem into a legible,
   narratable demo.
6. Apply the same relabeling to all four `diagram_node*.json` files (they're
   currently byte-identical — keep them identical after the edit too, that's
   fine, it's the `NODE_ID` `#define` that differentiates nodes, not the
   diagram).

### 5.5 Fix the inverted turbidity mapping

An LDR reads *higher* with *more* light, meaning clearer water. The current
code maps raw ADC directly to NTU, which is backwards. Invert:
```c
float readTurbidity() {
  int raw = analogRead(PIN_LDR);              // higher raw = more light = clearer
  return (4095 - raw) * (1000.0f / 4095.0f);  // invert: less light = higher NTU
}
```

### 5.6 Housekeeping
- Remove unused `flowPulseCount` / `FLOW_PULSES_PER_L` (no ISR attached, this
  implies pulse-counting that doesn't exist).
- `StaticJsonDocument` → `JsonDocument` (ArduinoJson 7 API).
- `char payload[320]` → `[512]` (current payload serializes to ~230 bytes and
  will truncate silently as fields grow).
- Pin library versions in `firmware/libraries.txt`.

**Acceptance:** open each of the 4 Wokwi projects, confirm every simulated
sensor is identifiable without reading the .ino source; run the simulation
with the potentiometers set to storm values and confirm the LCD/dashboard
shows ALERT (this was previously stuck at WATCH — this is the regression
test for the original bug).

---

## 6. Phase 4 — Backend: security, reliability, weather fusion

### 6.1 BLOCKER — authenticate and allowlist `/api/alert/send`

```python
ALERT_RECIPIENTS = {n.strip() for n in os.getenv("ALERT_RECIPIENTS", "").split(",") if n.strip()}
DASHBOARD_KEY = os.getenv("DASHBOARD_KEY", "")

@app.route("/api/alert/send", methods=["POST"])
def api_alert_send():
    if not DASHBOARD_KEY or request.headers.get("X-Dashboard-Key","") != DASHBOARD_KEY:
        return jsonify({"error": "unauthorized"}), 401
    # ... existing validation ...
    if digits not in ALERT_RECIPIENTS:
        return jsonify({"error": "recipient not in allowlist"}), 403
```
Add a **global** send budget (not just per-node cooldown — per-node scales
with node count, raising an attacker's ceiling as you add nodes):
```python
_global_sends: deque = deque(maxlen=200)
GLOBAL_MAX_PER_HOUR = 20
def _global_budget_ok() -> bool:
    now = time.monotonic()
    with _state_lock:
        while _global_sends and now - _global_sends[0] > 3600:
            _global_sends.popleft()
        if len(_global_sends) >= GLOBAL_MAX_PER_HOUR:
            return False
        _global_sends.append(now)
        return True
```
Call this inside `send_whatsapp()` before every Twilio call, including
`force=True` paths.

### 6.2 Fix the paho `on_disconnect` signature

```python
def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
    global _mqtt_connected
    _mqtt_connected = False
    log.warning("[MQTT] Disconnected rc=%s", reason_code)
```
Make `/api/health` return 503 (not 200) when degraded, so uptime monitors
actually catch it.

### 6.3 Node liveness (depends on firmware §5.3's LWT)

```python
NODE_STALE_SEC = int(os.getenv("NODE_STALE_SEC", 60))
def _is_stale(node: dict) -> bool:
    try:
        last = datetime.fromisoformat(node["last_updated"])
    except Exception:
        return True
    return (datetime.now(timezone.utc) - last).total_seconds() > NODE_STALE_SEC
```
Subscribe to `flood/status/+` in the bridge; mark nodes offline on the LWT
payload. Attach `"stale"` and `"seconds_since_update"` to every node in
`/api/nodes`. Frontend must render stale nodes as "NO DATA — last seen Xm
ago," never as their last known alert level.

### 6.4 Absolute, configurable DB path + WAL + index

```python
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "flood_data.db"))
```
`PRAGMA journal_mode=WAL`, add
`CREATE INDEX IF NOT EXISTS idx_telemetry_node_ts ON telemetry(node_id, timestamp DESC);`
Remove committed `.db` files from git (`git rm --cached`), add `*.db` /
`*.sqlite3` to `.gitignore`.

### 6.5 New — `backend/weather.py`: forecast fusion (standout feature)

Open-Meteo **forecast** API (distinct from the historical one used in Phase
2), free, no key:
```
GET https://api.open-meteo.com/v1/forecast?latitude=<LAT>&longitude=<LON>&daily=precipitation_sum&forecast_days=2&timezone=UTC
```
- Per node, look up (or accept as config) the node's lat/lon.
- Poll once per hour (cache — do not call this per-MQTT-packet), store
  `forecast_rain_24h_mm` and `forecast_rain_48h_mm`.
- Feed into `SafetyRules.evaluate()` as an additional corroborating reason:
  if current soil is already near `SOIL_SATURATED_PCT` **and** the forecast
  shows heavy incoming rain, that's a legitimate early-warning signal a
  sensor-only system can't see yet. Add as a new rule:
  ```python
  if soil >= cls.SOIL_SATURATED_PCT - 10 and forecast_rain_24h_mm >= cls.RAIN_WATCH_MM:
      level = max(level, 1)
      reasons.append(f"forecast rain {forecast_rain_24h_mm:.0f} mm on already-wet ground")
  ```
- This is a real differentiator: local-sensor-only flood systems are
  reactive; this makes it partly anticipatory using free public data. Note
  it explicitly in the README as a feature, and in the model card as
  "Layer 3."

### 6.6 New — SMS fallback (standout feature, reuses existing Twilio setup)

If `send_whatsapp` fails after retries (bad number format aside — genuine
delivery failure), fall back to Twilio SMS using the same account:
```python
def send_alert(node_id, alert, data, **kw):
    result = send_whatsapp(node_id, alert, data, **kw)
    if not result.get("success") and os.getenv("SMS_FALLBACK_FROM"):
        result = send_sms(node_id, alert, data, **kw)   # same Client, messaging body, no "whatsapp:" prefix
    return result
```
Document the additional `SMS_FALLBACK_FROM` env var (a Twilio SMS-capable
number) in `.env.example`.

### 6.7 New — `/api/metrics` (Prometheus text format, standout feature, no new dependency)

Hand-roll the exposition format (no need for `prometheus_client` dependency):
```python
@app.route("/api/metrics")
def api_metrics():
    nodes = db.get_all_nodes()
    lines = [
        "# HELP floodsense_nodes_total Registered nodes",
        "# TYPE floodsense_nodes_total gauge",
        f"floodsense_nodes_total {len(nodes)}",
        "# HELP floodsense_mqtt_connected MQTT broker connection state",
        "# TYPE floodsense_mqtt_connected gauge",
        f"floodsense_mqtt_connected {1 if _mqtt_connected else 0}",
    ]
    for n in nodes:
        lvl = {"NORMAL": 0, "WATCH": 1, "ALERT": 2}.get(n.get("alert_level"), -1)
        lines.append(f'floodsense_node_alert_level{{node_id="{n["node_id"]}"}} {lvl}')
    return "\n".join(lines) + "\n", 200, {"Content-Type": "text/plain; version=0.0.4"}
```
This is a genuine, low-cost resume point ("exposes Prometheus-compatible
metrics") and needs zero new infra to demonstrate — `curl` is enough.

### 6.8 Multi-node regional consensus escalation (standout feature)

If you're running 2+ nodes (the repo supports up to 4), a real system's most
convincing feature is cross-node corroboration: one sensor spiking might be
a glitch, two nearby nodes agreeing is a real event.

```python
def regional_check(all_nodes: list[dict]) -> str | None:
    watch_or_higher = [n for n in all_nodes if n.get("alert_level") in ("WATCH","ALERT") and not n.get("stale")]
    if len(watch_or_higher) >= 2:
        return f"REGIONAL WATCH: {len(watch_or_higher)} nodes elevated simultaneously"
    return None
```
Surface this on `/api/nodes` as a top-level `"regional_note"` field and
render it prominently on the dashboard when present. Do not use it to
override an individual node's Layer-1 decision — it's additional context,
not a fourth alarm layer.

### 6.9 Dependency pinning + Twilio client reuse
Pin `requirements.txt` to exact versions (a model artifact must be unpickled
by the exact scikit-learn version that wrote it — unpinned `>=` breaks this
silently). Build the Twilio `Client` once at module scope instead of per
message.

**Acceptance:**
```bash
pytest tests/ -v                     # all existing + new tests pass
curl -X POST /api/alert/send -d '{"node_id":"x","to":"+911234567"}'  # -> 403 (not on allowlist)
curl /api/health                     # -> 503 while MQTT disconnected, 200 when connected
curl /api/metrics                    # -> valid prometheus text format
```

---

## 7. Phase 5 — Frontend

- Render `stale` nodes distinctly (greyed out, "NO DATA — last seen Xm ago"),
  never showing a stale last-known level as if current.
- Render Layer 1's `reasons[]` array as a readable bullet list next to each
  node's alert badge — this is the auditability payoff, use it.
- Render `forecast_advisory` and `ood_features` separately from the primary
  `status`, visually de-emphasized, labeled "ML forecast (advisory)" with a
  tooltip explaining it does not drive alarms.
- Render `regional_note` (§6.8) as a banner when present.
- Add a small "why did this fire?" expandable log view backed by
  `/api/alerts/log` (already exists server-side, currently unused by the UI).

---

## 8. Phase 6 — Deployment (no Docker required, Docker optional)

Docker is not required — every feature runs as plain Python on
Render/Railway exactly like the original Procfile-based setup.

- **`Procfile`** (dedupe the current duplicate line):
  `web: gunicorn --chdir backend mqtt_bridge:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
  `--workers 1` is load-bearing (the MQTT client is process-local); add the
  runtime assertion from the patch notes so a misconfigured `WEB_CONCURRENCY`
  fails loudly instead of silently double-subscribing.
- **`render.yaml`**: add the new env vars —
  `ALERT_RECIPIENTS`, `DASHBOARD_KEY`, `ALLOWED_ORIGINS`, `DB_PATH`,
  `SMS_FALLBACK_FROM` (optional), `MODEL_PATH`. Move `MQTT_BROKER`/`MQTT_PORT`
  to a private, authenticated HiveMQ Cloud cluster (free tier) instead of the
  public `broker.hivemq.com` — required for §6.1 to mean anything, since an
  open broker lets anyone publish fake sensor data regardless of API auth.
- **Persistent disk**: Render's free tier has no disk; either provision a
  paid instance with a mounted disk for `DB_PATH`, or migrate `db.py` to
  Postgres (`DATABASE_URL` env var, swap `sqlite3` for `psycopg2` — keep the
  same function signatures in `db.py` so nothing else in the codebase
  changes). Prefer the disk for a demo project; note Postgres as the
  production-scale option in the README.
- **Drift monitoring without a compose stack**: Render/Railway both support
  scheduled jobs — `python monitoring/drift.py --db $DB_PATH --hours 24`,
  daily, alerting via a webhook (Slack incoming webhook is a 5-line
  `requests.post`) on nonzero exit.
- **CI**: strip the Docker `build` job from `.github/workflows/ci.yml` if
  Docker isn't being used for deployment, but **keep** `test` and
  `train-and-gate` — they're plain `pip install` + `pytest` +
  `python ml_pipeline/train.py`, no container involved, and `train-and-gate`
  is the single highest-value check in the whole pipeline.
- **Local dev broker**: use a free HiveMQ Cloud cluster directly (matches
  prod exactly) instead of local Mosquitto — one less moving part without
  Docker.

Keep `docker/` in the repo as an alternative path for anyone who does want
containers (some reviewers will specifically look for a Dockerfile) — just
don't make it load-bearing for the actual deploy.

---

## 9. Phase 7 — Docs

- **`README.md`**: rewrite the claims to match reality. Include the baseline
  comparison table from `train.py`'s output directly in the README (not just
  linked), the real-data-track table from Phase 2, the architecture diagram
  updated to show 3 layers (safety/forecast/weather), and the sensor-mapping
  table from §5.4. Replace "production-grade" language with an accurate
  description; the honest version ("deterministic safety layer for
  auditability, ML forecast benchmarked and honestly reported, weather-fusion
  for anticipatory signal") reads as *more* sophisticated, not less.
- **`MODEL_CARD.md`** (demo track, from Phase 1's rescaled data) and
  **`MODEL_CARD_REAL.md`** (Phase 2) both present and linked from the README.
- **`firmware/NODES_SETUP.md`**: the sensor-mapping table from §5.4.
- **`.env.example`**: add every new env var introduced in this plan with a
  one-line comment each.

---

## 10. Final self-verification checklist

Run all of these and paste the actual output into a final summary — do not
mark any item done without having run it:

```bash
# Correctness
python ml_pipeline/rescale_dataset.py
python ml_pipeline/train.py --no-gate      # demo track, expected to still lose the gate — that's fine
python ml_pipeline/fetch_usgs.py --site <SITE> && python ml_pipeline/fetch_openmeteo.py --site <SITE>
python ml_pipeline/train_real.py           # real track — report the actual number, win or lose
pytest tests/ -v                           # all green, including new tests you added

# Safety
python -c "from backend.predict import predictor; print(predictor.predict('t', {'water_level_m':3.4,'rainfall_24h_mm':180,'soil_moisture_pct':95,'flow_velocity_ms':8,'turbidity_ntu':900}))"
# must print status=ALERT with >=2 reasons

# Security
curl -s -X POST localhost:8080/api/alert/send -d '{"node_id":"x","to":"+911111111111"}' -H 'Content-Type: application/json'
# must be 401 or 403, never 200, without the dashboard key + allowlisted number

# Reliability
curl -s localhost:8080/api/health | python -m json.tool
curl -s localhost:8080/api/metrics

# Drift
python monitoring/drift.py --db <db seeded with Wokwi-range values> --hours 24
# must report healthy: true post-rescale (contrast with pre-rescale run, which must report healthy: false — run both and show the diff)
```

Ship a final summary that states, in plain language: what the demo track's
model achieves and doesn't, what the real-data track's model achieves and
doesn't, and which security/reliability items are now closed. Do not round a
"loses to baseline" result up to "working" anywhere in that summary.
