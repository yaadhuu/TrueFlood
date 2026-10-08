"""
Every setting the backend reads from the environment, in one place.

Values come from environment variables (or backend/.env when running locally).
See .env.example in the repo root for what each one means.
"""

import logging
import os

from dotenv import load_dotenv

_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"), override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("floodsense")


def _flag(name: str) -> bool:
    return os.getenv(name, "").lower() in ("true", "1", "yes")


# ── MQTT ──────────────────────────────────────────────────────────
BROKER = os.getenv("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", 8883))      # 8883 = TLS, 1883 = plaintext
MQTT_USER = os.getenv("MQTT_USER") or None
MQTT_PASS = os.getenv("MQTT_PASS") or None
DISABLE_MQTT = _flag("DISABLE_MQTT")               # tests run without a broker

TOPIC_IN = "flood/sensor/+"          # sensor readings, one sub-topic per node
TOPIC_STATUS = "flood/status/+"      # retained online/offline (MQTT Last Will)
TOPIC_OUT_FMT = "flood/alert/{node_id}"

# ── HTTP API ──────────────────────────────────────────────────────
API_PORT = int(os.getenv("PORT", 8080))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")
                   if o.strip()] or ["*"]
DEMO_MODE = _flag("DEMO_MODE")                      # enables POST /api/simulate
DASHBOARD_KEY = os.getenv("DASHBOARD_KEY", "")      # header for /api/alert/send
TEST_SECRET = os.getenv("TEST_SECRET", "")          # header for /api/test-alert
WEB_CONCURRENCY = int(os.getenv("WEB_CONCURRENCY", "1"))
MAX_HISTORY = 20                                    # readings returned per node

# ── Node liveness ─────────────────────────────────────────────────
NODE_STALE_SEC = int(os.getenv("NODE_STALE_SEC", 60))

# ── Alerts (Twilio WhatsApp, with SMS fallback) ──────────────────
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM")
TWILIO_TO = os.getenv("TWILIO_TO")
SMS_FALLBACK_FROM = os.getenv("SMS_FALLBACK_FROM")
TWILIO_READY = all([TWILIO_SID, TWILIO_AUTH, TWILIO_FROM, TWILIO_TO])

ALERT_COOLDOWN_SEC = 300                            # per node
GLOBAL_MAX_PER_HOUR = int(os.getenv("GLOBAL_MAX_PER_HOUR", 20))  # whole system

# Phone numbers the dashboard is allowed to message. If unset, default to the
# number that already receives automatic alerts.
ALERT_RECIPIENTS = {n.strip() for n in os.getenv("ALERT_RECIPIENTS", "").split(",")
                    if n.strip()}
if not ALERT_RECIPIENTS and TWILIO_TO:
    _default = TWILIO_TO.replace("whatsapp:", "").strip()
    if _default:
        ALERT_RECIPIENTS = {_default}

if not TWILIO_READY:
    log.warning("Twilio credentials incomplete - WhatsApp/SMS alerts are disabled.")
if not DASHBOARD_KEY:
    log.warning("DASHBOARD_KEY is unset - /api/alert/send will reject every request.")
