# Running 4 TrueFlood Nodes Together on Wokwi

Wokwi projects simulate ONE ESP32 board each — you can't put 4 boards in one
project file. To get 4 nodes "working together," run 4 separate Wokwi projects
that all publish to the same broker under different NODE_IDs. The backend
subscribes to the wildcard `flood/sensor/+` so it picks up all of them
automatically.

## What each part on the canvas actually represents

Board: **`wokwi-esp32-devkit-v1`** (DOIT ESP32 DevKit V1).

**This is a Wokwi simulation. The potentiometers stand in for sensors Wokwi
has no model for; sliding them simulates a live reading changing. This table
is the ground truth for what each control represents.**

| Wokwi part | Simulates | Real hardware equivalent | Range | Pin |
|---|---|---|---|---|
| `hcsr04_water_level` (HC-SR04) | Water level (inverted distance) | Ultrasonic / radar level sensor | 0 – 4.0 m | D5 trig / D18 echo |
| `pot_rainfall` (potentiometer) | Rainfall intensity | Tipping-bucket rain gauge | 0 – 200 mm/24h | D34 (ADC1) |
| `pot_soil_moisture` (potentiometer) | Ground saturation | Capacitive soil moisture probe | 0 – 100 % | D35 (ADC1) |
| `bmp_pressure` (BMP180) | Barometric pressure + trend | BMP280 / BME280 | 950 – 1040 hPa | D21/D22 (I2C 0x77) |
| `dht_humidity` (DHT22) | Humidity + temperature | DHT22 (real, no proxy) | % / °C | D15 |
| `lcd_dashboard` (LCD1602 I2C) | Local status readout | Same | — | D21/D22 (I2C 0x27) |
| `led_safe`/`led_warning`/`led_critical` + `buzzer_alarm` | Local alarm stack | Beacon + siren | — | D25/D26/D27, D14 |

**Removed in this refresh:** the flow-velocity potentiometer and the turbidity
photoresistor, along with the three float switches. Flow velocity and
turbidity were two of the five features the trained model consumed, so this
was not a firmware-only change — the model was retrained without them rather
than being fed silent `0.0` defaults, which would have produced confident
garbage that looked like it was working. See MODEL_CARD.md.

### Three hardware constraints that are easy to get wrong

**Both potentiometers must be on ADC1 (GPIO 32–39).** The ESP32's **ADC2 is
owned by the WiFi radio** — once WiFi is up, `analogRead()` on an ADC2 pin
returns garbage. This node publishes over MQTT, so ADC2 is off-limits for
sensors. GPIO34/35 are also input-only and have **no internal pull-ups**,
which is correct for an externally-driven potentiometer but means they must
not be reused for buttons later.

**This board's pin names differ from the devkit-c-v4.** GPIOs are `D`-prefixed
(`D21`, not `21`), GPIO36/39 are `VP`/`VN`, GPIO16/17 are `RX2`/`TX2`, and
there are only **two** grounds (`GND.1`, `GND.2`) — there is no bare `GND`.

**The LCD needs `"attrs": {"pins": "i2c"}`** — not `"pinout"`. With the wrong
key the part renders in 16-pin parallel mode and has no SDA/SCL to wire to.

`firmware/make_diagrams.py` validates all three at generation time, plus
double-booked GPIOs, so a mistake fails the build instead of silently
producing a diagram that will not wire up.

### Reproducing each state by hand

| Want | Do |
|---|---|
| **CRITICAL** | Soil pot to max, rain pot to max, and move the HC-SR04 object close (high water). Red flashing, continuous 3.5 kHz alarm, `!!! EVACUATE !!!`. |
| **WARNING** (predictive) | Leave the water low. Needs humidity > 85 %, rain > 5 mm and a falling barometer — see the note below. Yellow, slow chirp, `RISK: ELEVATED`. |
| **SAFE** | All nominal. Green solid, silent, `SYSTEM: SAFE`. |

**The pressure trend needs time, and possibly a debug flag.** The trend is a
rate, so it needs `TREND_WINDOW_MS` (60 s in this build) of samples before it
means anything. If Wokwi's BMP180 model does not let you vary pressure
interactively, the predictive WARNING path cannot be triggered by hand at all
— set `#define SIMULATE_PRESSURE_DROP 1` in `sketch.ino`, which ramps pressure
down ~4 hPa over 30 s so the path is demonstrable. Never ship that enabled.

**`TREND_WINDOW_MS` is 60 s here; production uses 3 hours.** A barometric
trend is a slow, hours-long quantity, and deriving hPa/hr from a short window
amplifies sensor noise enormously. The short window exists only so the
behaviour is visible inside a Wokwi session — it is not a real hourly trend.

### Serial-only build

`#define ENABLE_MQTT 0` drops WiFi and PubSubClient entirely. The state
machine, LCD, LEDs, buzzer and the one-JSON-line-per-second serial output all
still run, so the Wokwi share link works for a visitor with no broker
credentials.

### Two brains, and which one is driving

The backend decides whenever it is reachable: `backend/predict.py`'s Layer 1
has hysteresis, multi-sensor corroboration, cross-node consensus and weather
fusion, none of which a single node can do alone.

But a warning device that goes silent when its uplink dies is not a warning
device. So the sketch also computes its **own** SAFE / WARNING / CRITICAL
verdict every second from the same thresholds, and falls back to it after
`BACKEND_TIMEOUT_MS` (30 s) without a backend message. The LCD shows `SRV` or
`LOC` so you can always tell which one you are looking at, and every telemetry
packet carries `decision_source`.

The two threshold sets are kept identical by
`tests/test_firmware_sync.py` — if somebody tunes one side, the test fails
rather than letting the node and the dashboard quietly contradict each other.

### Water level has one scale, 0 – 4 m, everywhere

`TANK_DEPTH_M` in `sketch.ino` is the single source of truth, and
`ml_pipeline/rescale_dataset.py` puts the training data on that same scale.
`tests/test_firmware_sync.py` asserts the two still agree.

## Steps (repeat 4 times, once per node)

1. Go to [wokwi.com](https://wokwi.com) → **New Project** → **ESP32**.
2. Delete the default `diagram.json` content and paste in
   `firmware/diagram.json` from this repo.
3. Paste `firmware/sketch.ino` into the code editor.
4. Paste `firmware/libraries.txt` into the project's `libraries.txt`
   (versions are pinned — the sketch uses the ArduinoJson **7** API).
5. Change exactly two lines at the top of the sketch:
   ```cpp
   #define NODE_ID    "node-3"          // node-1, node-2, node-3, node-4
   #define NODE_LABEL "River Station C" // any readable name
   ```
6. Click the green **Run / Play** button. Watch the Serial Monitor for
   `[MQTT] Connected` — if it doesn't connect, Wokwi-GUEST wifi can be flaky;
   click Restart.
7. Save the project (top-left) and copy its share URL — paste that URL into the
   matching "Node N" tab on the dashboard's **Wokwi** panel (Topology tab).
8. Repeat for node-2 … node-4 in **separate browser tabs**, running
   simultaneously. Leave all tabs open — closing a tab stops that node's
   simulated hardware from publishing.

## Taking a screenshot that actually shows every sensor

The diagram is deliberately laid out wide (four labelled groups around the
ESP32 — water level top-left, weather bottom-left, flow/quality top-right,
outputs centre) so nothing overlaps, but Wokwi's default zoom after loading a
project is usually zoomed IN past what fits that whole layout in frame. If a
screenshot only shows the ESP32 and one or two parts, that's a zoom problem,
not a missing-sensor problem — every part is already on the canvas.

1. Load the diagram, then before running (or while it's running) find the
   **zoom-to-fit** control in the Wokwi editor toolbar (a square/frame icon,
   usually near the zoom +/− controls in the top-right of the diagram pane) —
   or scroll/pinch-zoom out manually until all four labelled groups and the
   ESP32 are visible with no part clipped at the edges.
2. Click **Run** so the LEDs/LCD reflect a live alert state, and (optionally)
   move a couple of potentiometers toward storm values first — see
   "Reproducing an ALERT on demand" below — so the screenshot shows the red
   ALERT LED lit and the LCD's alert screen, not just idle green.
3. Capture the whole canvas in one shot (browser screenshot tool or OS
   screenshot), not just the ESP32 close-up.
4. For a "full system" screenshot (sensors + live graph + ML output together),
   put the Wokwi tab and the dashboard's **Live Telemetry** tab side by side —
   the node card under the matching `NODE_ID` shows every live sensor value
   (water level, rainfall, soil moisture, pressure + trend, temp/humidity),
   the Layer 1 reasons, and the Layer 2 advisory probabilities updating from
   the same packets the Wokwi tab is publishing.

## Verifying it worked

- Open the dashboard's **Live** tab — 4 node cards should appear within ~5–10 s
  of each Wokwi sim starting (`PUBLISH_MS` is 5000 ms).
- `GET /api/nodes` should list all 4 `node_id`s with fresh `last_updated`
  timestamps and `"stale": false`.
- Kill one Wokwi tab. Within `NODE_STALE_SEC` (default 60 s) that node must
  render as **NO DATA**, not as its last known alert level. The retained
  `offline` message on `flood/status/<NODE_ID>` (the MQTT Last Will) usually
  gets there first.

## The diagram file

`firmware/diagram.json` is the wiring for every node; nodes differ only by
`NODE_ID` / `NODE_LABEL` in `sketch.ino`. It is generated by
`python firmware/make_diagrams.py` - edit that script, not the JSON, so the
wiring and the sketch's pin numbers stay in sync.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| No `[MQTT] Connected` in Serial Monitor | Wokwi GUEST wifi unavailable | Click Restart; try a different browser |
| Node card appears, then disappears | Another Wokwi tab with the SAME `NODE_ID` | Ensure each tab uses a unique NODE_ID |
| `NORMAL` forever with maxed potentiometers | Only one threshold crossed, so the 10 s debounce is holding | Cross two thresholds at once (see the table above) |
| Compile error on `StaticJsonDocument` | ArduinoJson 6 pinned instead of 7 | Use `firmware/libraries.txt` as-is |
| Twilio error 63016 on Send Alert | Phone hasn't joined the sandbox | Send `join <sandbox-code>` to +1 415 523 8886 on WhatsApp |
