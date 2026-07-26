
// ═══════════════════════════════════════════════════════════════════
//  FloodSense ESP32 – Multi-Node Edition
//  ► Change NODE_ID and NODE_LABEL for each physical/simulated node
// ═══════════════════════════════════════════════════════════════════

#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <DHT.h>
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// ── NODE IDENTITY  ◄─── CHANGE THIS FOR EACH NODE ─────────────────
#define NODE_ID    "node-1"          // Unique: "node-1", "node-2", …
#define NODE_LABEL "River Station A" // Human-readable name for LCD/UI
// ──────────────────────────────────────────────────────────────────

// ── WiFi ──────────────────────────────────────────────────────────
const char* WIFI_SSID     = "Wokwi-GUEST";
const char* WIFI_PASSWORD = "";

// ── MQTT ──────────────────────────────────────────────────────────
const char* MQTT_BROKER   = "broker.hivemq.com";
const int   MQTT_PORT     = 1883;
// Topics are derived from NODE_ID at runtime (see setup())
char TOPIC_SENSOR[64];   // flood/sensor/<NODE_ID>
char TOPIC_ALERT[64];    // flood/alert/<NODE_ID>
char CLIENT_ID[32];      // esp32-<NODE_ID>  (unique per node — fixes CLIENT_ID collision bug)

// ── Pin Definitions ───────────────────────────────────────────────
#define PIN_SW1             23    // Float switch LOW
#define PIN_SW2             14    // Float switch MID
#define PIN_SW3             27    // Float switch HIGH
#define PIN_ULTRASONIC_TRIG 5     // HC-SR04 Ultrasonic TRIG
#define PIN_ULTRASONIC_ECHO 17    // HC-SR04 Ultrasonic ECHO
#define PIN_DHT             12    // DHT22
#define PIN_SOIL            34    // Soil Moisture ADC
#define PIN_RAIN            36    // Rainfall ADC (VP)
#define PIN_LDR             32    // Turbidity LDR Photoresistor ADC
#define PIN_FLOW            25    // Flow velocity ADC (pot in Wokwi)
#define PIN_LED_G           26    // Green LED – NORMAL
#define PIN_LED_Y           33    // Yellow LED – WATCH
#define PIN_LED_R           19    // Red LED – ALERT
#define PIN_BUZZER          18    // Buzzer

// ── Sensor Config ─────────────────────────────────────────────────
#define DHT_TYPE           DHT22
#define PUBLISH_MS         5000         // Publish interval (ms)
#define MQTT_RETRY_DELAY   3000         // Delay between MQTT retries (ms)  ← fixes reconnect-spam bug
#define FLOW_PULSES_PER_L  7.5f

// ── Objects ───────────────────────────────────────────────────────
LiquidCrystal_I2C lcd(0x27, 16, 2);
DHT              dht(PIN_DHT, DHT_TYPE);
WiFiClient       wifiClient;
PubSubClient     mqtt(wifiClient);

// ── State ─────────────────────────────────────────────────────────
volatile uint32_t flowPulseCount = 0;
unsigned long lastPublish    = 0;
unsigned long lastFlowTime   = 0;
unsigned long lastMqttRetry  = 0;   // rate-limit reconnect attempts
String        currentAlert   = "NORMAL";

// ─────────────────────────────────────────────────────────────────
// ADC / SENSOR HELPERS
// ─────────────────────────────────────────────────────────────────
float readWaterLevelSwitches() {
  bool sw1 = digitalRead(PIN_SW1);
  bool sw2 = digitalRead(PIN_SW2);
  bool sw3 = digitalRead(PIN_SW3);
  if (sw3) return 55.0f;   // ALERT level
  if (sw2) return 20.0f;   // WATCH level
  if (sw1) return 10.0f;   // NORMAL level
  return 0.3f;             // DRY
}

float readWaterLevelUltrasonic() {
  digitalWrite(PIN_ULTRASONIC_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_ULTRASONIC_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_ULTRASONIC_TRIG, LOW);
  long duration = pulseIn(PIN_ULTRASONIC_ECHO, HIGH, 30000); // 30ms timeout
  if (duration == 0) return readWaterLevelSwitches(); // Fallback if no pulse
  float distance_cm = duration * 0.0343f / 2.0f;
  float tank_depth_cm = 400.0f; // 4m depth reference
  float level_m = (tank_depth_cm - distance_cm) / 100.0f;
  return max(0.0f, level_m);
}

float readWaterLevel() {
  return readWaterLevelUltrasonic();
}

float readRainfall() {
  return analogRead(PIN_RAIN) * (200.0f / 4095.0f);   // 0–200 mm
}

float readSoilMoisture() {
  return analogRead(PIN_SOIL) * (100.0f / 4095.0f);   // 0–100 %
}

float readTurbidity() {
  // LDR photoresistor: murkier water scatters more light
  return analogRead(PIN_LDR) * (1000.0f / 4095.0f);  // 0–1000 NTU
}

float readFlowVelocity() {
  unsigned long now     = millis();
  unsigned long elapsed = now - lastFlowTime;
  if (elapsed < 1000) return 0.0f;
  lastFlowTime = now;
  return analogRead(PIN_FLOW) * (10.0f / 4095.0f);    // 0–10 m/s
}

// ─────────────────────────────────────────────────────────────────
// LCD
// ─────────────────────────────────────────────────────────────────
void updateLCD(float wl, float hum, float temp, String alert) {
  lcd.clear();
  lcd.setCursor(0, 0);
  // Show node label (truncated to 9 chars) + alert
  String label = String(NODE_LABEL);
  if (label.length() > 9) label = label.substring(0, 9);
  lcd.print(label + " " + alert);

  lcd.setCursor(0, 1);
  lcd.print("WL:");
  lcd.print(wl, 1);
  lcd.print("m H:");
  lcd.print((int)hum);
  lcd.print("% T:");
  lcd.print((int)temp);
  lcd.print("C");
}

// ─────────────────────────────────────────────────────────────────
// LED / BUZZER
// ─────────────────────────────────────────────────────────────────
void setAlertIndicators(String alert) {
  currentAlert = alert;
  digitalWrite(PIN_LED_G, LOW);
  digitalWrite(PIN_LED_Y, LOW);
  digitalWrite(PIN_LED_R, LOW);
  noTone(PIN_BUZZER);

  if      (alert == "NORMAL") { digitalWrite(PIN_LED_G, HIGH); }
  else if (alert == "WATCH")  { digitalWrite(PIN_LED_Y, HIGH); }
  else if (alert == "ALERT")  {
    digitalWrite(PIN_LED_R, HIGH);
    tone(PIN_BUZZER, 1000);
  }
}

// ─────────────────────────────────────────────────────────────────
// MQTT CALLBACK
// ─────────────────────────────────────────────────────────────────
void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String msg;
  for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];
  Serial.print("[MQTT IN] "); Serial.println(msg);

  StaticJsonDocument<512> doc;
  if (deserializeJson(doc, msg)) {
    Serial.println("[MQTT] JSON parse error");
    return;
  }

  String alertLevel = "NORMAL";
  if (doc.containsKey("alert_level")) {
    alertLevel = doc["alert_level"].as<String>();
  } else if (doc.containsKey("status")) {
    alertLevel = "ALERT";
  }

  setAlertIndicators(alertLevel);

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("** ML RESULT **");
  lcd.setCursor(0, 1); lcd.print("=> "); lcd.print(alertLevel);
  delay(2000);
}

// ─────────────────────────────────────────────────────────────────
// WiFi
// ─────────────────────────────────────────────────────────────────
void connectWiFi() {
  lcd.clear(); lcd.setCursor(0, 0); lcd.print("Connecting WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500); Serial.print("."); attempts++;
  }
  lcd.clear();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n[WiFi] Connected: " + WiFi.localIP().toString());
    lcd.setCursor(0, 0); lcd.print("WiFi OK");
    lcd.setCursor(0, 1); lcd.print(WiFi.localIP().toString());
  } else {
    Serial.println("\n[WiFi] FAILED – offline mode");
    lcd.setCursor(0, 0); lcd.print("WiFi FAILED");
    lcd.setCursor(0, 1); lcd.print("Offline mode");
  }
  delay(1500);
}

// ─────────────────────────────────────────────────────────────────
// MQTT CONNECT (rate-limited – no more reconnect spam)
// ─────────────────────────────────────────────────────────────────
void connectMQTT() {
  if (WiFi.status() != WL_CONNECTED) return;
  if (mqtt.connected()) return;

  unsigned long now = millis();
  if (now - lastMqttRetry < MQTT_RETRY_DELAY) return;   // ← rate-limit fix
  lastMqttRetry = now;

  Serial.print("[MQTT] Connecting as "); Serial.print(CLIENT_ID); Serial.print("...");
  if (mqtt.connect(CLIENT_ID)) {
    Serial.println(" OK");
    mqtt.subscribe(TOPIC_ALERT);
    Serial.print("[MQTT] Subscribed: "); Serial.println(TOPIC_ALERT);
  } else {
    Serial.print(" FAILED rc="); Serial.println(mqtt.state());
  }
}

// ─────────────────────────────────────────────────────────────────
// SETUP
// ─────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);

  // Build per-node topic strings (fixes hardcoded topic bug)
  snprintf(TOPIC_SENSOR, sizeof(TOPIC_SENSOR), "flood/sensor/%s", NODE_ID);
  snprintf(TOPIC_ALERT,  sizeof(TOPIC_ALERT),  "flood/alert/%s",  NODE_ID);
  snprintf(CLIENT_ID,    sizeof(CLIENT_ID),    "esp32-%s",         NODE_ID);

  Serial.printf("\n=== FloodSense [%s] – %s ===\n", NODE_ID, NODE_LABEL);
  Serial.printf("   Sensor topic : %s\n", TOPIC_SENSOR);
  Serial.printf("   Alert  topic : %s\n", TOPIC_ALERT);

  // GPIO
  pinMode(PIN_SW1,             INPUT);
  pinMode(PIN_SW2,             INPUT);
  pinMode(PIN_SW3,             INPUT);
  pinMode(PIN_ULTRASONIC_TRIG, OUTPUT);
  pinMode(PIN_ULTRASONIC_ECHO, INPUT);
  pinMode(PIN_LED_G,           OUTPUT);
  pinMode(PIN_LED_Y,           OUTPUT);
  pinMode(PIN_LED_R,           OUTPUT);
  pinMode(PIN_BUZZER,          OUTPUT);

  // LCD
  Wire.begin(21, 22);
  lcd.init(); lcd.backlight();
  lcd.setCursor(0, 0); lcd.print("FloodSense v2.0");
  lcd.setCursor(0, 1); lcd.print(NODE_ID);
  delay(1500);

  // DHT22
  dht.begin();

  // WiFi + MQTT
  connectWiFi();
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(mqttCallback);
  connectMQTT();

  setAlertIndicators("NORMAL");
  lastPublish  = millis();
  lastFlowTime = millis();
  Serial.println("[Setup] Ready – publishing every 5s");
}

// ─────────────────────────────────────────────────────────────────
// LOOP
// ─────────────────────────────────────────────────────────────────
void loop() {
  if (!mqtt.connected()) connectMQTT();
  mqtt.loop();

  unsigned long now = millis();
  if (now - lastPublish < PUBLISH_MS) return;
  lastPublish = now;

  // Read sensors
  float waterLevel   = readWaterLevel();
  float rainfall     = readRainfall();
  float soilMoisture = readSoilMoisture();
  float turbidity    = readTurbidity();
  float flowVelocity = readFlowVelocity();
  float temperature  = dht.readTemperature();
  float humidity     = dht.readHumidity();

  if (isnan(temperature)) temperature = 28.0f;
  if (isnan(humidity))    humidity    = 65.0f;

  bool   sw1 = digitalRead(PIN_SW1);
  bool   sw2 = digitalRead(PIN_SW2);
  bool   sw3 = digitalRead(PIN_SW3);
  String wlLabel = sw3 ? "HIGH" : sw2 ? "MID" : sw1 ? "LOW" : "DRY";

  // Serial debug
  Serial.printf("── [%s] ──────────────────────────\n", NODE_ID);
  Serial.printf("Water Level : %.2f m  [%s]\n", waterLevel, wlLabel.c_str());
  Serial.printf("Rainfall    : %.1f mm\n", rainfall);
  Serial.printf("Soil Moist  : %.1f %%\n", soilMoisture);
  Serial.printf("Turbidity   : %.0f NTU\n", turbidity);
  Serial.printf("Flow Vel    : %.2f m/s\n", flowVelocity);
  Serial.printf("Temp / Hum  : %.1f°C / %.0f%%\n", temperature, humidity);
  Serial.printf("Alert State : %s\n", currentAlert.c_str());

  updateLCD(waterLevel, humidity, temperature, currentAlert);

  // Build and publish JSON
  StaticJsonDocument<320> doc;
  doc["node_id"]           = NODE_ID;
  doc["node_label"]        = NODE_LABEL;
  doc["water_level_m"]     = round(waterLevel   * 100.0f) / 100.0f;
  doc["rainfall_24h_mm"]   = round(rainfall     * 10.0f)  / 10.0f;
  doc["soil_moisture_pct"] = round(soilMoisture * 10.0f)  / 10.0f;
  doc["flow_velocity_ms"]  = round(flowVelocity * 100.0f) / 100.0f;
  doc["turbidity_ntu"]     = round(turbidity);
  doc["temperature_c"]     = round(temperature  * 10.0f)  / 10.0f;
  doc["humidity_pct"]      = round(humidity     * 10.0f)  / 10.0f;
  doc["water_level_label"] = wlLabel;
  doc["uptime_s"]          = now / 1000;

  char payload[320];
  serializeJson(doc, payload);

  if (mqtt.connected()) {
    bool ok = mqtt.publish(TOPIC_SENSOR, payload);
    Serial.printf("[MQTT OUT] %s → %s\n", TOPIC_SENSOR, ok ? "OK" : "FAIL");
  } else {
    Serial.println("[MQTT] Not connected – skipping publish");
  }
}
