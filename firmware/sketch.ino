
// ═══════════════════════════════════════════════════════════════════
//  FloodSense ESP32 – Flood Prediction & Early-Warning Node
//  ► Change NODE_ID and NODE_LABEL for each physical/simulated node
//
//  Board: wokwi-esp32-devkit-v1 (DOIT ESP32 DevKit V1)
//
//  SENSOR ROSTER
//    HC-SR04        water level (faces DOWN: shorter distance = higher water)
//    potentiometer  rainfall intensity        GPIO34 (ADC1)
//    potentiometer  ground saturation         GPIO35 (ADC1)
//    BMP180         barometric pressure+temp  I2C 0x77
//    DHT22          humidity + temp           GPIO15
//    LCD1602 (I2C)  local readout             I2C 0x27
//    3x LED, buzzer local alarm stack
//
//  WHERE THE DECISION IS MADE
//  --------------------------
//  Normally the backend decides: this node publishes readings over MQTT,
//  backend/predict.py's Layer 1 evaluates them with hysteresis, cross-node
//  consensus and weather fusion, and publishes a verdict back to
//  flood/alert/<NODE_ID> which this sketch renders.
//
//  But a warning device that goes silent when its uplink dies is not a
//  warning device. So this sketch ALSO runs the full risk state machine
//  locally and falls back to it whenever the backend has gone quiet, or
//  whenever ENABLE_MQTT is 0. Which brain is driving is shown on the LCD
//  (SRV/LOC) and published as "decision_source".
// ═══════════════════════════════════════════════════════════════════

#include <stdarg.h>            // va_list, for the fixed-width LCD formatter
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <DHT.h>
#include <Adafruit_BMP085.h>   // correct driver for BMP180 - register-compatible
#include <ArduinoJson.h>

// ── BUILD MODE ────────────────────────────────────────────────────
// 1 = networked node (WiFi + MQTT).  0 = standalone Wokwi demo: the state
// machine, LCD, LEDs, buzzer and serial JSON all still run, so a visitor
// opening the share link with no broker credentials sees a working device.
#define ENABLE_MQTT 1

// Wokwi's BMP180 model may not expose an interactive pressure control, which
// would make the predictive WARNING path impossible to demonstrate by hand.
// Set to 1 to ramp pressure down ~4 hPa over 30 s after boot so the
// storm-precursor branch is reproducible. NEVER ship this enabled.
#define SIMULATE_PRESSURE_DROP 0

#if ENABLE_MQTT
  #include <WiFi.h>
  #include <PubSubClient.h>
#endif

// ── NODE IDENTITY  ◄─── CHANGE THIS FOR EACH NODE ─────────────────
#define NODE_ID    "node-1"          // Unique: "node-1", "node-2", …
#define NODE_LABEL "River Station A" // Human-readable name for LCD/UI
// ──────────────────────────────────────────────────────────────────

#if ENABLE_MQTT
const char* WIFI_SSID     = "Wokwi-GUEST";
const char* WIFI_PASSWORD = "";
const char* MQTT_BROKER   = "broker.hivemq.com";
const int   MQTT_PORT     = 1883;
char TOPIC_SENSOR[64];   // flood/sensor/<NODE_ID>
char TOPIC_ALERT[64];    // flood/alert/<NODE_ID>
char TOPIC_STATUS[64];   // flood/status/<NODE_ID>  — retained LWT
char CLIENT_ID[32];
#endif

// ── Pin map (GPIO numbers; the diagram names them D5, D15, … ) ─────
// Analog sensors MUST be on ADC1 (GPIO32-39): the ESP32's ADC2 is owned by
// the WiFi radio, so analogRead() on an ADC2 pin returns garbage once WiFi
// is up. GPIO34/35 are also input-only (no internal pull-ups), which is
// correct for a potentiometer and why they must not be reused for buttons.
#define PIN_TRIG      5     // HC-SR04 TRIG
#define PIN_ECHO      18    // HC-SR04 ECHO
#define PIN_DHT       15    // DHT22 data
#define PIN_RAIN      34    // rainfall pot     ADC1_CH6, input-only
#define PIN_SOIL      35    // soil pot         ADC1_CH7, input-only
#define PIN_LED_SAFE      25
#define PIN_LED_WARNING   26
#define PIN_LED_CRITICAL  27
#define PIN_BUZZ      14
#define PIN_I2C_SDA   21
#define PIN_I2C_SCL   22

#define DHT_TYPE      DHT22   // swap to DHT11 for cheaper hardware, same wiring

// ── Schedulers (no delay() anywhere below setup) ───────────────────
const unsigned long SENSOR_INTERVAL_MS  = 1000;   // read + serial JSON
const unsigned long PUBLISH_INTERVAL_MS = 5000;   // MQTT publish
const unsigned long BLINK_INTERVAL_MS   = 250;    // critical LED flash
const unsigned long CHIRP_PERIOD_MS     = 2000;   // warning chirp period
const unsigned long CHIRP_ON_MS         = 100;    // warning chirp length
const unsigned long SCROLL_INTERVAL_MS  = 400;    // LCD marquee step
unsigned long lastSensorRead = 0, lastPublish = 0, lastBlink = 0, lastScroll = 0;

// ── Scale + thresholds ─────────────────────────────────────────────
#define TANK_DEPTH_M       4.0f   // single source of truth for the level scale

// These MIRROR SafetyRules in backend/predict.py. If the node and the server
// disagree about what 3 metres means, the device contradicts the dashboard in
// front of whoever is watching. tests/test_firmware_sync.py enforces parity.
#define WL_WATCH_M              2.0f
#define WL_ALERT_M              3.0f
#define RAIN_WATCH_MM          90.0f
#define RAIN_ALERT_MM         180.0f
#define SOIL_SATURATED_PCT     98.0f
#define HUMIDITY_STORM_PCT     85.0f
#define RAIN_STARTED_MM         5.0f
#define PRESSURE_FALL_WARNING  -1.0f   // hPa/hr, rapid fall = storm approaching
#define PRESSURE_FALL_SEVERE   -2.5f   // hPa/hr, flash-flood-producing systems

// Hysteresis: a single noisy HC-SR04 read must not fire an evacuation alarm.
#define ESCALATE_AFTER_READS    2
#define DEESCALATE_AFTER_READS 10

// Rolling window for the pressure trend. PRODUCTION USES 3 HOURS
// (10800000 ms) - a barometric trend is a slow, hours-long quantity and
// deriving hPa/hr from a short window amplifies sensor noise enormously.
// This is set to 1 minute ONLY so the storm-precursor path is demonstrable
// inside a Wokwi session. Do not mistake this for a real hourly trend.
#define TREND_WINDOW_MS       60000UL
#define TREND_SAMPLES            32

enum RiskLevel { RISK_SAFE = 0, RISK_WARNING = 1, RISK_CRITICAL = 2 };

// ── Objects ───────────────────────────────────────────────────────
LiquidCrystal_I2C lcd(0x27, 16, 2);
DHT              dht(PIN_DHT, DHT_TYPE);
Adafruit_BMP085  bmp;
#if ENABLE_MQTT
WiFiClient       wifiClient;
PubSubClient     mqtt(wifiClient);
unsigned long    lastMqttRetry  = 0;
unsigned long    lastBackendMsg = 0;
bool             hadBackendMsg  = false;
RiskLevel        backendRisk    = RISK_SAFE;
String           pendingAlertMsg = "";
unsigned long    alertMsgUntil  = 0;
bool             alertScreenUp  = false;
const unsigned long MQTT_RETRY_DELAY    = 3000;
const unsigned long BACKEND_TIMEOUT_MS  = 30000;
const unsigned long ALERT_MSG_MS        = 2000;
#endif

// ── Sensor state ──────────────────────────────────────────────────
float sWaterLevel = 0.0f, sRainfall = 0.0f, sSoil = 0.0f;
float sPressure = 1013.25f, sTemp = 28.0f, sHumidity = 65.0f;
float lastGoodWaterLevel = 0.0f;
bool  bmpReady = false;
unsigned long sensorErrors = 0;
const char* sensorSource = "ultrasonic";   // "ultrasonic" | "stale"

// Pressure ring buffer -> trend in hPa/hr
float         trendP[TREND_SAMPLES];
unsigned long trendT[TREND_SAMPLES];
int           trendHead = 0, trendCount = 0;
float         pressureTrend = 0.0f;
bool          pressureTrendValid = false;

// ── Risk state ────────────────────────────────────────────────────
RiskLevel localRisk = RISK_SAFE, activeRisk = RISK_SAFE;
RiskLevel candidateRisk = RISK_SAFE;
int       confirmCount = 0;
String    reasons[4];
int       reasonCount = 0;
bool      usingBackend = false;

// ── Output state ──────────────────────────────────────────────────
bool      redLedOn = false;
int       buzzerTone = 0;
int       scrollOffset = 0;
RiskLevel lastDrawnRisk = (RiskLevel)-1;
bool      forceRedraw = true;

// ─────────────────────────────────────────────────────────────────
// SENSORS
// ─────────────────────────────────────────────────────────────────
// The HC-SR04 is mounted above the channel looking DOWN, so it measures the
// air gap, not the depth: a SHORTER echo distance means a HIGHER water level.
// A 0 return means the echo never came back — hold the last good value rather
// than publishing 0.0, which the model would read as "empty channel".
float readWaterLevel() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);          // 10 us strobe per the HC-SR04 datasheet
  digitalWrite(PIN_TRIG, LOW);

  long dur = pulseIn(PIN_ECHO, HIGH, 30000);   // capped: worst case 30 ms
  if (dur == 0) {
    sensorErrors++;
    sensorSource = "stale";
    return lastGoodWaterLevel;
  }
  sensorSource = "ultrasonic";
  float dist_cm = dur * 0.0343f / 2.0f;        // speed of sound, round trip
  float level   = (TANK_DEPTH_M * 100.0f - dist_cm) / 100.0f;
  lastGoodWaterLevel = constrain(level, 0.0f, TANK_DEPTH_M);
  return lastGoodWaterLevel;
}

float readRainfall() { return analogRead(PIN_RAIN) * (200.0f / 4095.0f); }  // 0-200 mm
float readSoil()     { return analogRead(PIN_SOIL) * (100.0f / 4095.0f); }  // 0-100 %

// Push a pressure sample and recompute the trend across the window.
void updatePressureTrend(float hpa, unsigned long now) {
  trendP[trendHead] = hpa;
  trendT[trendHead] = now;
  trendHead = (trendHead + 1) % TREND_SAMPLES;
  if (trendCount < TREND_SAMPLES) trendCount++;

  // Oldest sample still inside the window.
  int   oldestIdx = -1;
  for (int i = 0; i < trendCount; i++) {
    int idx = (trendHead - 1 - i + TREND_SAMPLES * 2) % TREND_SAMPLES;
    if (now - trendT[idx] <= TREND_WINDOW_MS) oldestIdx = idx;
  }
  if (oldestIdx < 0) { pressureTrendValid = false; return; }

  unsigned long span = now - trendT[oldestIdx];
  if (span < TREND_WINDOW_MS / 2) { pressureTrendValid = false; return; }

  pressureTrend = (hpa - trendP[oldestIdx]) / (span / 3600000.0f);  // hPa/hr
  pressureTrendValid = true;
}

void readSensors(unsigned long now) {
  sWaterLevel = readWaterLevel();
  sRainfall   = readRainfall();
  sSoil       = readSoil();

  // DHT22 read failures are common and return NaN, which serializes to null
  // and breaks the backend parser. Hold the previous value instead.
  float h = dht.readHumidity();
  float t = dht.readTemperature();
  if (!isnan(h)) sHumidity = h; else sensorErrors++;
  if (!isnan(t)) sTemp = t;     else sensorErrors++;

  if (bmpReady) {
    float pa = bmp.readPressure();            // Pa
    if (!isnan(pa) && pa > 30000 && pa < 110000) sPressure = pa / 100.0f;
    float bt = bmp.readTemperature();
    if (!isnan(bt) && isnan(t)) sTemp = bt;   // BMP180 temp as DHT fallback
  }
#if SIMULATE_PRESSURE_DROP
  // Debug ramp: -4 hPa over the first 30 s so the WARNING path is provable
  // when the simulator offers no interactive pressure control.
  {
    float elapsed = min((unsigned long)30000, now) / 30000.0f;
    sPressure = 1013.25f - 4.0f * elapsed;
  }
#endif

  updatePressureTrend(sPressure, now);
}

// ─────────────────────────────────────────────────────────────────
// PREDICTIVE RISK STATE MACHINE
// ─────────────────────────────────────────────────────────────────
void addReason(const String &r) {
  if (reasonCount < 4) reasons[reasonCount++] = r;
}

RiskLevel evaluateRisk() {
  reasonCount = 0;
  RiskLevel level = RISK_SAFE;

  // ── CRITICAL ──────────────────────────────────────────────────
  if (sWaterLevel >= WL_ALERT_M) {
    level = RISK_CRITICAL;
    addReason("water level " + String(sWaterLevel, 2) + " m >= " +
              String(WL_ALERT_M, 1) + " m");
  }
  // Saturated ground has zero absorption capacity left: peak rain on 98%
  // soil runs straight off into an already-high channel.
  if (sWaterLevel >= WL_WATCH_M && sRainfall >= RAIN_ALERT_MM &&
      sSoil >= SOIL_SATURATED_PCT) {
    level = RISK_CRITICAL;
    addReason("high water + " + String(sRainfall, 0) + " mm rain on " +
              String(sSoil, 0) + "% saturated ground");
  }
  if (pressureTrendValid && pressureTrend <= PRESSURE_FALL_SEVERE) {
    level = RISK_CRITICAL;
    addReason("pressure falling " + String(-pressureTrend, 1) +
              " hPa/hr (severe)");
  }

  if (level == RISK_CRITICAL) return level;

  // ── WARNING ───────────────────────────────────────────────────
  // The predictive branch: this fires BEFORE the water level peaks, which is
  // the entire reason the barometer is on the node.
  if (pressureTrendValid && pressureTrend <= PRESSURE_FALL_WARNING &&
      sHumidity > HUMIDITY_STORM_PCT && sRainfall > RAIN_STARTED_MM) {
    level = RISK_WARNING;
    addReason("pressure falling " + String(-pressureTrend, 1) +
              " hPa/hr with " + String(sHumidity, 0) + "% humidity");
  }
  if (sWaterLevel >= WL_WATCH_M) {
    level = RISK_WARNING;
    addReason("water level " + String(sWaterLevel, 2) + " m >= " +
              String(WL_WATCH_M, 1) + " m");
  }
  if (sRainfall >= RAIN_WATCH_MM) {
    level = RISK_WARNING;
    addReason("rainfall " + String(sRainfall, 0) + " mm/24h");
  }

  if (level == RISK_SAFE) addReason("all sensors nominal");
  return level;
}

// Hysteresis: escalate after ESCALATE_AFTER_READS confirming reads,
// de-escalate only after DEESCALATE_AFTER_READS. A one-tick spike on the
// ultrasonic sensor must never reach the siren.
void applyHysteresis(RiskLevel raw) {
  if (raw == localRisk) { candidateRisk = raw; confirmCount = 0; return; }
  if (raw != candidateRisk) { candidateRisk = raw; confirmCount = 0; }
  confirmCount++;
  int needed = (raw > localRisk) ? ESCALATE_AFTER_READS : DEESCALATE_AFTER_READS;
  if (confirmCount >= needed) {
    localRisk = raw;
    confirmCount = 0;
  }
}

const char* riskName(RiskLevel r) {
  switch (r) {
    case RISK_CRITICAL: return "CRITICAL";
    case RISK_WARNING:  return "WARNING";
    default:            return "SAFE";
  }
}

#if ENABLE_MQTT
// Backend speaks NORMAL/WATCH/ALERT; the device speaks SAFE/WARNING/CRITICAL.
RiskLevel riskFromBackend(const String &s) {
  if (s == "ALERT") return RISK_CRITICAL;
  if (s == "WATCH") return RISK_WARNING;
  return RISK_SAFE;
}
#endif

void selectActiveRisk(unsigned long now) {
#if ENABLE_MQTT
  bool fresh = hadBackendMsg && mqtt.connected() &&
               (now - lastBackendMsg < BACKEND_TIMEOUT_MS);
  usingBackend = fresh;
  activeRisk   = fresh ? backendRisk : localRisk;
#else
  usingBackend = false;
  activeRisk   = localRisk;
#endif
}

// ─────────────────────────────────────────────────────────────────
// PERIPHERALS  (driven from the scheduler; never block)
// ─────────────────────────────────────────────────────────────────
void renderIndicators(unsigned long now) {
  if (activeRisk == RISK_CRITICAL) {
    if (now - lastBlink >= BLINK_INTERVAL_MS) { lastBlink = now; redLedOn = !redLedOn; }
  } else {
    redLedOn = false;
  }
  digitalWrite(PIN_LED_SAFE,     activeRisk == RISK_SAFE    ? HIGH : LOW);
  digitalWrite(PIN_LED_WARNING,  activeRisk == RISK_WARNING ? HIGH : LOW);
  digitalWrite(PIN_LED_CRITICAL, redLedOn                   ? HIGH : LOW);

  // Only touch the buzzer when the desired tone changes: calling tone() every
  // iteration re-initialises the ESP32's LEDC channel thousands of times a
  // second, which wastes cycles and audibly glitches the output.
  int want = 0;
  if (activeRisk == RISK_CRITICAL) {
    want = 3500;                                        // continuous siren
  } else if (activeRisk == RISK_WARNING) {
    if (now % CHIRP_PERIOD_MS < CHIRP_ON_MS) want = 2000;   // slow chirp
  }
  if (want != buzzerTone) {
    if (want == 0) noTone(PIN_BUZZ); else tone(PIN_BUZZ, want);
    buzzerTone = want;
  }
}

// Render exactly 16 characters, padded/truncated. A 16x2 LCD silently mangles
// anything longer, and "Depth: 4.00m P:1013" is 19.
void lcdLine(char *dst, const char *fmt, ...) {
  char tmp[96];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(tmp, sizeof(tmp), fmt, ap);
  va_end(ap);
  snprintf(dst, 17, "%-16.16s", tmp);
}

void drawStatus(unsigned long now) {
  bool changed = forceRedraw || activeRisk != lastDrawnRisk;
  char l0[17], l1[17];

  if (activeRisk == RISK_WARNING) {
    // Scroll the reason one character per tick — no blocking loop.
    if (!changed && now - lastScroll < SCROLL_INTERVAL_MS) return;
    lastScroll = now;
    if (changed) scrollOffset = 0;

    String r = reasonCount ? reasons[0] : String("elevated risk");
    r += "   ";
    int len = r.length();
    char win[17];
    for (int i = 0; i < 16; i++) win[i] = r[(scrollOffset + i) % len];
    win[16] = '\0';
    scrollOffset = (scrollOffset + 1) % len;

    lcdLine(l0, "RISK: ELEVATED");
    lcdLine(l1, "%s", win);
  } else {
    if (!changed && now - lastScroll < 1000) return;
    lastScroll = now;
    if (activeRisk == RISK_CRITICAL) lcdLine(l0, "!!! EVACUATE !!!");
    else                             lcdLine(l0, "SYSTEM: SAFE");
    lcdLine(l1, "Depth: %.2fm %s", sWaterLevel, usingBackend ? "SRV" : "LOC");
  }

  lcd.setCursor(0, 0); lcd.print(l0);
  lcd.setCursor(0, 1); lcd.print(l1);
  lastDrawnRisk = activeRisk;
  forceRedraw = false;
}

void updateLCD(unsigned long now) {
#if ENABLE_MQTT
  if (now < alertMsgUntil) {
    if (!alertScreenUp) {
      lcd.clear();
      lcd.setCursor(0, 0); lcd.print("** ML RESULT ** ");
      lcd.setCursor(0, 1); lcd.print("=> " + pendingAlertMsg + "        ");
      alertScreenUp = true;
    }
    return;
  }
  if (alertScreenUp) { alertScreenUp = false; lcd.clear(); forceRedraw = true; }
#endif
  drawStatus(now);
}

// ─────────────────────────────────────────────────────────────────
// TELEMETRY
// ─────────────────────────────────────────────────────────────────
void buildPayload(char *out, size_t outLen, unsigned long now) {
  JsonDocument doc;              // ArduinoJson 7; StaticJsonDocument deprecated
  doc["node_id"]            = NODE_ID;
  doc["node_label"]         = NODE_LABEL;
  doc["water_level_m"]      = round(sWaterLevel * 100.0f) / 100.0f;
  doc["water_level_max_m"]  = TANK_DEPTH_M;
  doc["rainfall_24h_mm"]    = round(sRainfall * 10.0f) / 10.0f;
  doc["soil_moisture_pct"]  = round(sSoil * 10.0f) / 10.0f;
  doc["pressure_hpa"]       = round(sPressure * 10.0f) / 10.0f;
  if (pressureTrendValid)
    doc["pressure_trend_hpa_per_hr"] = round(pressureTrend * 10.0f) / 10.0f;
  doc["temperature_c"]      = round(sTemp * 10.0f) / 10.0f;
  doc["humidity_pct"]       = round(sHumidity * 10.0f) / 10.0f;
  doc["risk_state"]         = riskName(localRisk);
  JsonArray arr = doc["reasons"].to<JsonArray>();
  for (int i = 0; i < reasonCount; i++) arr.add(reasons[i]);
  doc["sensor_source"]      = sensorSource;
  doc["sensor_errors"]      = sensorErrors;
  doc["decision_source"]    = usingBackend ? "backend" : "local";
  doc["uptime_s"]           = now / 1000;
  serializeJson(doc, out, outLen);
}

// ─────────────────────────────────────────────────────────────────
// MQTT
// ─────────────────────────────────────────────────────────────────
#if ENABLE_MQTT
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String msg;
  for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];

  JsonDocument doc;
  if (deserializeJson(doc, msg)) return;

  String level = "NORMAL";
  if (doc["alert_level"].is<const char*>()) level = doc["alert_level"].as<String>();

  backendRisk    = riskFromBackend(level);
  lastBackendMsg = millis();
  hadBackendMsg  = true;

  // No delay() here: this runs inside mqtt.loop() and a stall risks a
  // keepalive timeout. Latch it; updateLCD() renders it.
  pendingAlertMsg = level;
  alertMsgUntil   = millis() + ALERT_MSG_MS;
}

void connectMQTT() {
  if (WiFi.status() != WL_CONNECTED || mqtt.connected()) return;
  unsigned long now = millis();
  if (now - lastMqttRetry < MQTT_RETRY_DELAY) return;
  lastMqttRetry = now;

  // LWT: if this node dies the broker publishes "offline" retained, so the
  // backend marks it dead instead of serving a stale alert level forever.
  if (mqtt.connect(CLIENT_ID, NULL, NULL, TOPIC_STATUS, 1, true, "offline")) {
    mqtt.publish(TOPIC_STATUS, "online", true);
    mqtt.subscribe(TOPIC_ALERT);
    Serial.println("[MQTT] connected");
  }
}
#endif

// ─────────────────────────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_LED_SAFE, OUTPUT);
  pinMode(PIN_LED_WARNING, OUTPUT);
  pinMode(PIN_LED_CRITICAL, OUTPUT);
  pinMode(PIN_BUZZ, OUTPUT);

  Wire.begin(PIN_I2C_SDA, PIN_I2C_SCL);
  lcd.init(); lcd.backlight();
  lcd.setCursor(0, 0); lcd.print("FloodSense v4.0 ");
  lcd.setCursor(0, 1); lcd.print(NODE_ID);

  dht.begin();
  bmpReady = bmp.begin();
  if (!bmpReady)
    Serial.println("[BMP180] not detected - pressure held, storm rule disabled");

#if ENABLE_MQTT
  snprintf(TOPIC_SENSOR, sizeof(TOPIC_SENSOR), "flood/sensor/%s", NODE_ID);
  snprintf(TOPIC_ALERT,  sizeof(TOPIC_ALERT),  "flood/alert/%s",  NODE_ID);
  snprintf(TOPIC_STATUS, sizeof(TOPIC_STATUS), "flood/status/%s", NODE_ID);
  snprintf(CLIENT_ID,    sizeof(CLIENT_ID),    "esp32-%s",        NODE_ID);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  for (int i = 0; i < 40 && WiFi.status() != WL_CONNECTED; i++) delay(250);  // setup only
  Serial.println(WiFi.status() == WL_CONNECTED
                 ? "[WiFi] connected" : "[WiFi] failed - local mode");
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  connectMQTT();
#else
  Serial.println("[build] ENABLE_MQTT=0 - standalone demo, serial JSON only");
#endif

  lastGoodWaterLevel = 0.0f;
  readSensors(millis());
  localRisk = evaluateRisk();
  lcd.clear();
  Serial.println("[setup] ready - sensors 1 s, publish 5 s");
}

// ─────────────────────────────────────────────────────────────────
// LOOP — non-blocking; no delay() below this line
// ─────────────────────────────────────────────────────────────────
void loop() {
  unsigned long now = millis();

#if ENABLE_MQTT
  if (!mqtt.connected()) connectMQTT();
  mqtt.loop();
#endif

  // 1 Hz: read sensors, evaluate risk, emit one JSON line
  if (now - lastSensorRead >= SENSOR_INTERVAL_MS) {
    lastSensorRead = now;
    readSensors(now);
    applyHysteresis(evaluateRisk());

    char payload[512];
    buildPayload(payload, sizeof(payload), now);
    Serial.println(payload);
  }

  // every loop: pick the driving verdict and render it
  selectActiveRisk(now);
  renderIndicators(now);
  updateLCD(now);

#if ENABLE_MQTT
  // 0.2 Hz: publish the same document, so serial and MQTT never diverge
  if (now - lastPublish >= PUBLISH_INTERVAL_MS) {
    lastPublish = now;
    if (mqtt.connected()) {
      char payload[512];
      buildPayload(payload, sizeof(payload), now);
      mqtt.publish(TOPIC_SENSOR, payload);
    }
  }
#endif
}
