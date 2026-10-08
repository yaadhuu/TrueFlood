"""
Generate firmware/diagram.json, the Wokwi wiring for every sensor node.

All nodes share this wiring; each node is told apart only by the NODE_ID
#define in sketch.ino. Generating the file (instead of hand-editing it) keeps
the pin numbers here and in sketch.ino in sync - tests/test_firmware_sync.py
checks that.

Sensor roster (hardware refresh)
--------------------------------
    HC-SR04        water level, downward-facing over the channel bed
    potentiometer  rainfall intensity   (tipping-bucket gauge sim)
    potentiometer  ground saturation    (capacitive probe sim)
    BMP180         barometric pressure + temperature (storm-front detection)
    DHT22          relative humidity + temperature
    LCD1602 (I2C)  local status readout
    3x LED + buzzer  local alarm stack

Flow-velocity and turbidity sensors were REMOVED in this refresh. They were
two of the five features the trained model consumed, so removing them is not
a firmware-only change -- see MODEL_CARD.md. The model was
retrained without them rather than being fed silent 0.0 defaults.

Board / pin naming
------------------
`wokwi-esp32-devkit-v1` is the DOIT ESP32 DevKit V1. Two things bite anyone
porting a diagram from `board-esp32-devkit-c-v4` or writing pins from memory:

  * its GPIO pins are named with a **D prefix** (`D21`, not `21`), and
    GPIO36/39 are `VP`/`VN`, GPIO16/17 are `RX2`/`TX2`;
  * it exposes `GND.1` and `GND.2` -- there is no bare `GND` pin.

Verified against wokwi/wokwi-boards -> boards/esp32-devkit-v1/board.json.
`validate()` below rejects any pin name this board does not have, so a typo
fails generation instead of silently producing a diagram that will not wire up.

ADC allocation (a correctness issue, not a style one)
-----------------------------------------------------
The ESP32's **ADC2 is unavailable while WiFi is active** -- the radio owns
that peripheral, and `analogRead()` on an ADC2 pin returns garbage once WiFi
is up. This firmware publishes over MQTT, so both potentiometers sit on
**ADC1**: GPIO34 (rainfall) and GPIO35 (soil). Both are also input-only,
which is correct for a sensor and is why they must not be reused for buttons
later -- they have no internal pull-ups.

Usage:
    python firmware/make_diagrams.py
"""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

BOARD_TYPE = "wokwi-esp32-devkit-v1"


def text(part_id: str, body: str, top: float, left: float) -> dict:
    return {"type": "wokwi-text", "id": part_id, "top": top, "left": left,
            "attrs": {"text": body}}


PARTS = [
    {"type": BOARD_TYPE, "id": "esp", "top": 200, "left": 240, "attrs": {}},

    # ── cluster 1: water level ────────────────────────────────────────
    text("txt_group_level", "WATER LEVEL", -60, -280),
    {"type": "wokwi-hc-sr04", "id": "hcsr04_water_level", "top": -10, "left": -280,
     "attrs": {"label": "Water Level - HC-SR04, faces DOWN at the channel"}},
    text("txt_level_note",
         "Mounted above the water, so it measures the AIR GAP:\n"
         "shorter distance = higher water. 0 - 4.0 m (TANK_DEPTH_M).",
         100, -280),

    # ── cluster 2: weather front ──────────────────────────────────────
    text("txt_group_weather", "WEATHER FRONT", 200, -280),
    {"type": "wokwi-bmp180", "id": "bmp_pressure", "top": 250, "left": -280,
     "attrs": {"label": "Barometric Pressure + Temp (BMP180) - storm front"}},
    {"type": "wokwi-dht22", "id": "dht_humidity", "top": 250, "left": -100,
     "attrs": {"label": "Humidity + Temp (DHT22)"}},
    text("txt_pressure_note",
         "A FALLING barometer is the storm precursor.\n"
         "The rate (hPa/hr) is the signal, not the reading.",
         390, -280),

    # ── cluster 3: ground saturation & rainfall ───────────────────────
    text("txt_group_ground", "RAINFALL & GROUND SATURATION", 480, -280),
    {"type": "wokwi-potentiometer", "id": "pot_rainfall", "top": 520, "left": -280,
     "attrs": {"label": "Rain Gauge (tipping-bucket sim) 0-200 mm/24h"}},
    {"type": "wokwi-potentiometer", "id": "pot_soil_moisture", "top": 520, "left": -80,
     "attrs": {"label": "Soil Moisture (capacitive probe sim) 0-100 %"}},
    text("txt_ground_note",
         "Saturated ground has no absorption capacity left:\n"
         "rain on 98% soil runs straight off into the channel.",
         650, -280),

    # ── cluster 4: local alerting ─────────────────────────────────────
    text("txt_group_out", "LOCAL ALERTING  (backend verdict, or local if offline)",
         20, 470),
    {"type": "wokwi-resistor", "id": "r1", "top": 90, "left": 480,
     "attrs": {"value": "220"}},
    {"type": "wokwi-resistor", "id": "r2", "top": 160, "left": 480,
     "attrs": {"value": "220"}},
    {"type": "wokwi-resistor", "id": "r3", "top": 230, "left": 480,
     "attrs": {"value": "220"}},
    {"type": "wokwi-led", "id": "led_safe", "top": 90, "left": 580,
     "attrs": {"color": "green", "label": "SAFE"}},
    {"type": "wokwi-led", "id": "led_warning", "top": 160, "left": 580,
     "attrs": {"color": "yellow", "label": "WARNING"}},
    {"type": "wokwi-led", "id": "led_critical", "top": 230, "left": 580,
     "attrs": {"color": "red", "label": "CRITICAL"}},
    {"type": "wokwi-buzzer", "id": "buzzer_alarm", "top": 300, "left": 480,
     "attrs": {"volume": "0.1"}},
    # NOTE: "pins": "i2c" is the attribute that switches this part to the
    # 4-wire PCF8574 backpack. "pinout" is NOT a valid key -- with it the part
    # renders in 16-pin parallel mode and has no SDA/SCL to wire to.
    {"type": "wokwi-lcd1602", "id": "lcd_dashboard", "top": 390, "left": 450,
     "attrs": {"pins": "i2c", "address": "0x27"}},
    text("txt_sim_note",
         "Wokwi simulation: the potentiometers stand in for sensors Wokwi has\n"
         "no model for. See firmware/NODES_SETUP.md for the full mapping.",
         570, 450),
]

# wokwi-esp32-devkit-v1 exposes GND.1 and GND.2 (no bare "GND").
CONNECTIONS = [
    # ── I2C bus: BMP180 @ 0x77 and LCD @ 0x27 share SDA/SCL ──
    ["bmp_pressure:VCC", "esp:3V3", "red", ["v0"]],
    ["bmp_pressure:GND", "esp:GND.1", "black", ["v0"]],
    ["bmp_pressure:SDA", "esp:D21", "blue", ["v0"]],
    ["bmp_pressure:SCL", "esp:D22", "green", ["v0"]],

    ["lcd_dashboard:VCC", "esp:3V3", "red", ["v0"]],
    ["lcd_dashboard:GND", "esp:GND.1", "black", ["v0"]],
    ["lcd_dashboard:SDA", "esp:D21", "blue", ["v0"]],
    ["lcd_dashboard:SCL", "esp:D22", "green", ["v0"]],

    # ── digital sensors ──
    ["dht_humidity:VCC", "esp:3V3", "red", ["v0"]],
    ["dht_humidity:GND", "esp:GND.1", "black", ["v0"]],
    ["dht_humidity:SDA", "esp:D15", "orange", ["v0"]],

    ["hcsr04_water_level:VCC", "esp:3V3", "red", ["v0"]],
    ["hcsr04_water_level:GND", "esp:GND.2", "black", ["v0"]],
    ["hcsr04_water_level:TRIG", "esp:D5", "purple", ["v0"]],
    ["hcsr04_water_level:ECHO", "esp:D18", "orange", ["v0"]],

    # ── analog sensors: ADC1 only (see the ADC note in the module docstring) ──
    ["pot_rainfall:VCC", "esp:3V3", "red", ["v0"]],
    ["pot_rainfall:GND", "esp:GND.2", "black", ["v0"]],
    ["pot_rainfall:SIG", "esp:D34", "green", ["v0"]],

    ["pot_soil_moisture:VCC", "esp:3V3", "red", ["v0"]],
    ["pot_soil_moisture:GND", "esp:GND.2", "black", ["v0"]],
    ["pot_soil_moisture:SIG", "esp:D35", "purple", ["v0"]],

    # ── local alerting ──
    ["esp:D25", "r1:1", "green", ["v0"]],
    ["r1:2", "led_safe:A", "green", ["v0"]],
    ["led_safe:C", "esp:GND.1", "black", ["v0"]],

    ["esp:D26", "r2:1", "yellow", ["v0"]],
    ["r2:2", "led_warning:A", "yellow", ["v0"]],
    ["led_warning:C", "esp:GND.1", "black", ["v0"]],

    ["esp:D27", "r3:1", "red", ["v0"]],
    ["r3:2", "led_critical:A", "red", ["v0"]],
    ["led_critical:C", "esp:GND.2", "black", ["v0"]],

    ["esp:D14", "buzzer_alarm:1", "purple", ["v0"]],
    ["buzzer_alarm:2", "esp:GND.2", "black", ["v0"]],
]

# Every board pin this diagram may reference, from wokwi/wokwi-boards ->
# boards/esp32-devkit-v1/board.json. Guessing pin names is how a diagram
# silently fails to wire up, so they are checked at generation time.
BOARD_PINS = {
    "3V3", "EN", "VIN", "VP", "VN", "GND.1", "GND.2",
    "RX0", "TX0", "RX2", "TX2",
    "D2", "D4", "D5", "D12", "D13", "D14", "D15", "D18", "D19", "D21",
    "D22", "D23", "D25", "D26", "D27", "D32", "D33", "D34", "D35",
}

# GPIOs on ADC2 - unusable for analogRead() while WiFi is active.
ADC2_PINS = {"D0", "D2", "D4", "D12", "D13", "D14", "D15", "D25", "D26", "D27"}
ANALOG_SIGNALS = {"pot_rainfall:SIG", "pot_soil_moisture:SIG"}


def validate() -> None:
    ids = {p["id"] for p in PARTS}
    if len(ids) != len(PARTS):
        raise SystemExit("duplicate part id in PARTS")

    used_gpio: dict[str, str] = {}
    for conn in CONNECTIONS:
        for endpoint in conn[:2]:
            part = endpoint.split(":")[0]
            if part not in ids:
                raise SystemExit(f"connection references unknown part id: {part}")

        for endpoint in conn[:2]:
            part, _, pin = endpoint.partition(":")
            if part != "esp":
                continue
            if pin not in BOARD_PINS:
                raise SystemExit(
                    f"'{pin}' is not a pin on {BOARD_TYPE} "
                    f"(D-prefixed GPIOs; grounds are GND.1/GND.2, no bare GND)")
            # Signal pins must not be double-booked. Power/ground rails are
            # shared on purpose, and so is the I2C bus.
            if pin in ("3V3", "GND.1", "GND.2", "D21", "D22"):
                continue
            other = conn[1] if endpoint == conn[0] else conn[0]
            if pin in used_gpio and used_gpio[pin] != other:
                raise SystemExit(
                    f"pin {pin} is allocated twice: {used_gpio[pin]} and {other}")
            used_gpio[pin] = other

        a, b = conn[0], conn[1]
        for sensor_end, board_end in ((a, b), (b, a)):
            if sensor_end in ANALOG_SIGNALS and board_end.startswith("esp:"):
                pin = board_end.split(":", 1)[1]
                if pin in ADC2_PINS:
                    raise SystemExit(
                        f"{sensor_end} is wired to {pin}, which is ADC2. "
                        "ADC2 cannot be read while WiFi is active on the "
                        "ESP32 - move it to ADC1 (D32-D35/VP/VN).")

    # The LCD only has SDA/SCL to wire to when it is in I2C mode.
    for part in PARTS:
        if part["type"] == "wokwi-lcd1602":
            if part.get("attrs", {}).get("pins") != "i2c":
                raise SystemExit(
                    "lcd1602 needs attrs.pins == 'i2c' (NOT 'pinout') or it "
                    "renders in 16-pin parallel mode with no SDA/SCL")


def main() -> int:
    validate()
    diagram = {
        "version": 1,
        "author": "FloodSense",
        "editor": "wokwi",
        "parts": PARTS,
        "connections": CONNECTIONS,
        "dependencies": {},
    }
    blob = json.dumps(diagram, indent=2) + "\n"
    # Every node uses the same wiring; nodes differ only by NODE_ID in sketch.ino.
    path = os.path.join(HERE, "diagram.json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(blob)
    print(f"wrote {path}")
    print(f"{len(PARTS)} parts, {len(CONNECTIONS)} connections, "
          f"board={BOARD_TYPE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
