---
name: trueflood-ops
description: Deployment and integration gotchas specific to TrueFlood's Flask+MQTT+Twilio backend. Use when touching mqtt_bridge.py, Twilio alerts, Procfile, or Render deployment config.
---

# TrueFlood Ops Knowledge

## Gunicorn / Worker Count
- **ALWAYS** run `--workers 1`. `mqtt_bridge.py` starts a paho MQTT client at
  import time. The client_id is now a random hex suffix per process, but a second
  worker still causes double-subscribe noise and duplicate WhatsApp alerts.
- The Procfile fix: `gunicorn --chdir backend mqtt_bridge:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`

## Twilio WhatsApp Sandbox
- `TWILIO_FROM=whatsapp:+14155238886` is the shared Twilio sandbox number.
- Sandbox join sessions **expire after 3 days idle**. The recipient in `TWILIO_TO`
  must re-send `join <your-sandbox-code>` to +1 415 523 8886 before every
  test session after a gap.
- Find the sandbox code at: Twilio Console → Messaging → Try it out →
  Send a WhatsApp message.
- Before assuming a code bug, always rejoin the sandbox first.

## Twilio Error Codes to Know
| Code | Meaning |
|------|---------|
| 63016 | Number not in sandbox (rejoin required) |
| 63018 | Rate limit exceeded |
| 21211 | Invalid `To` number format (must be `whatsapp:+E.164`) |
| 21608 | Unverified number for trial account |

## TwilioRestException
- Always catch `TwilioRestException` from `twilio.base.exceptions` first.
- Log both `exc.code` and `exc.msg` — not just `str(exc)`.
- On failure, roll back `node_alert_times[node_id]` so the next attempt isn't blocked.

## Frontend API URL
- **Never** default to `localhost:8080` in production code.
- The production Render URL is the `PRODUCTION_BACKEND_URL` constant in `index.html`.
- Change it to your actual Render service URL before deploying.

## MQTT Client ID
- `client_id` is now `f"flood-bridge-{uuid4().hex[:8]}"` — unique per process restart.
- This prevents HiveMQ broker-side disconnect loops when Render restarts the dyno.

## MQTT TLS
- Backend connects on port `8883` with `mc.tls_set()` (TLS).
- Frontend already uses WSS on port `8884`. Both paths are now encrypted.
- `MQTT_PORT` env var defaults to `8883`; override to `1883` only for local dev
  without a TLS-capable broker.

## Demo / Test Endpoints
- `POST /api/simulate` — only active when `DEMO_MODE=true` env var is set.
  Use for testing the full pipeline (MQTT→ML→alert) without hardware.
- `POST /api/test-alert` — requires header `X-Test-Key: <TEST_SECRET>`.
  Calls Twilio directly, returns Twilio SID or error code. Use to isolate
  Twilio config from MQTT/ML issues.
- `GET /api/alerts/log` — last 50 send attempts with outcome, Twilio code, msg.

## Cold-Start on Render Free Tier
- Free Render services sleep after ~15 min of inactivity.
- Use an uptime pinger (e.g., cron-job.org) to hit `/api/health` every 10 min.
- The first request after sleep can take 30–50 s — document this in the README.
