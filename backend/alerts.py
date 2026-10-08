"""
Outbound alerts: WhatsApp first, SMS as a fallback, both through Twilio.

Sending a message costs money, so three limits apply:
  1. per-node cooldown   - one alert per node every ALERT_COOLDOWN_SEC
  2. global hourly budget - at most GLOBAL_MAX_PER_HOUR messages across all nodes
  3. (in app.py) the dashboard endpoint needs a key and an allowlisted number
"""

import threading
import time
from collections import deque
from datetime import datetime, timezone

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

import config
from config import log

# One Twilio client for the whole process (each Client opens its own HTTP session).
_twilio = (Client(config.TWILIO_SID, config.TWILIO_AUTH)
           if (config.TWILIO_SID and config.TWILIO_AUTH) else None)

# Alerts are sent from MQTT's network thread and Flask's request threads at the
# same time, so every piece of shared state below is guarded by this lock.
_lock = threading.RLock()
_last_sent: dict[str, float] = {}         # node_id -> time of last alert
_sent_times: deque = deque(maxlen=200)    # send times, for the hourly budget
alert_log: deque = deque(maxlen=50)       # recent attempts, shown by /api/alerts/log


# ── Global hourly budget ──────────────────────────────────────────

def _drop_older_than_an_hour(now: float) -> None:
    while _sent_times and now - _sent_times[0] > 3600:
        _sent_times.popleft()


def sends_last_hour() -> int:
    with _lock:
        _drop_older_than_an_hour(time.monotonic())
        return len(_sent_times)


def global_budget_ok() -> bool:
    """Reserve one message from the hourly budget; False if it is used up."""
    now = time.monotonic()
    with _lock:
        _drop_older_than_an_hour(now)
        if len(_sent_times) >= config.GLOBAL_MAX_PER_HOUR:
            return False
        _sent_times.append(now)
        return True


def _refund_budget() -> None:
    """Give the reservation back when Twilio rejected the message."""
    with _lock:
        if _sent_times:
            _sent_times.pop()


# ── Message text ──────────────────────────────────────────────────

def _alert_body(node_id: str, alert: str, data: dict) -> str:
    trend = data.get("pressure_trend_hpa_per_hr")
    trend_txt = ("" if trend is None else
                 f" ({'falling' if trend < 0 else 'rising'} {abs(float(trend)):.1f} hPa/hr)")
    why = "\n".join(f"  - {r}" for r in (data.get("reasons") or [])[:5])
    return (
        f"FLOOD ALERT\n"
        f"Node      : {data.get('node_label', node_id)} ({node_id})\n"
        f"Status    : {alert}\n"
        f"Water Lvl : {data.get('water_level_m', 'N/A')} m\n"
        f"Rainfall  : {data.get('rainfall_24h_mm', 'N/A')} mm\n"
        f"Soil Moist: {data.get('soil_moisture_pct', 'N/A')} %\n"
        f"Humidity  : {data.get('humidity_pct', 'N/A')} %\n"
        f"Pressure  : {data.get('pressure_hpa', 'N/A')} hPa{trend_txt}\n"
        + (f"Why       :\n{why}\n" if why else "")
        + f"Time      : {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )


# ── Sending ───────────────────────────────────────────────────────

def _twilio_send(*, from_: str, to: str, body: str) -> dict:
    if _twilio is None:
        return {"success": False, "error": "Twilio not ready"}
    if not global_budget_ok():
        log.error("[alert] hourly budget of %d used up - not sending",
                  config.GLOBAL_MAX_PER_HOUR)
        return {"success": False, "error": "global_budget"}
    try:
        msg = _twilio.messages.create(from_=from_, to=to, body=body)
        return {"success": True, "sid": msg.sid}
    except TwilioRestException as exc:
        _refund_budget()
        return {"success": False, "error": str(exc),
                "twilio_code": exc.code, "twilio_msg": exc.msg}
    except Exception as exc:  # noqa: BLE001 - network errors etc.
        _refund_budget()
        return {"success": False, "error": str(exc)}


def _default_recipient() -> str:
    return (config.TWILIO_TO or "").replace("whatsapp:", "")


def send_whatsapp(node_id: str, alert: str, data: dict, *,
                  force: bool = False, to_override: str | None = None) -> dict:
    if not config.TWILIO_READY:
        return {"success": False, "error": "Twilio not ready"}

    now = time.monotonic()
    with _lock:
        if not force and now - _last_sent.get(node_id, 0.0) < config.ALERT_COOLDOWN_SEC:
            return {"success": False, "error": "cooldown"}
        _last_sent[node_id] = now

    result = _twilio_send(from_=config.TWILIO_FROM,
                          to=f"whatsapp:{to_override or _default_recipient()}",
                          body=_alert_body(node_id, alert, data))
    result["channel"] = "whatsapp"
    if result["success"]:
        log.info("[WhatsApp][%s] sent - SID %s", node_id, result["sid"])
    elif result["error"] != "global_budget":
        with _lock:                      # failed: let a retry through the cooldown
            _last_sent.pop(node_id, None)
        log.error("[WhatsApp][%s] failed: %s", node_id, result.get("error"))
    return result


def send_sms(node_id: str, alert: str, data: dict, *,
             to_override: str | None = None) -> dict:
    if not config.SMS_FALLBACK_FROM:
        return {"success": False, "error": "SMS fallback not configured"}
    result = _twilio_send(from_=config.SMS_FALLBACK_FROM,
                          to=to_override or _default_recipient(),
                          body=_alert_body(node_id, alert, data))
    result["channel"] = "sms"
    return result


def send_alert(node_id: str, alert: str, data: dict, *,
               force: bool = False, to_override: str | None = None) -> dict:
    """WhatsApp, then SMS if WhatsApp failed for a reason other than our own limits."""
    result = send_whatsapp(node_id, alert, data, force=force, to_override=to_override)
    if (not result["success"]
            and result["error"] not in ("cooldown", "global_budget")
            and config.SMS_FALLBACK_FROM):
        log.info("[alert][%s] WhatsApp failed - falling back to SMS", node_id)
        result = send_sms(node_id, alert, data, to_override=to_override)

    with _lock:
        alert_log.append({
            "ts": datetime.now(timezone.utc).isoformat(),
            "node_id": node_id,
            "alert": alert,
            "channel": result.get("channel"),
            "success": bool(result.get("success")),
            "sid": result.get("sid"),
            "error": result.get("error"),
            "twilio_code": result.get("twilio_code"),
            "twilio_msg": result.get("twilio_msg"),
        })
    return result


def send_alert_in_background(node_id: str, alert: str, data: dict) -> None:
    """Twilio calls take ~1 s; never block the MQTT thread or an HTTP request on them."""
    threading.Thread(target=send_alert, args=(node_id, alert, data), daemon=True).start()
