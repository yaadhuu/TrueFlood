import json
import logging
import os
import sqlite3
from datetime import datetime, timezone

log = logging.getLogger("floodsense.db")

# Absolute and configurable. The previous relative "flood_data.db" resolved
# against the process's working directory, which is how two different copies
# of the database ended up committed to the repo. On Render's free tier the
# filesystem is ephemeral, so point DB_PATH at a mounted disk (or migrate to
# Postgres) if the history needs to survive a restart.
#
# os.getenv(name, default) only falls back when the var is absent — a var
# present but set to "" (exactly what .env.example's "leave blank for
# default" documents) returns "" verbatim. Blank is treated the same as unset.
_DB_PATH_ENV = os.getenv("DB_PATH")
DB_PATH = _DB_PATH_ENV if _DB_PATH_ENV else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "flood_data.db")


def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # WAL: readers (Flask request threads) do not block the writer (the MQTT
    # callback thread), which is exactly this workload.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            alert_level TEXT,
            probabilities TEXT,
            last_updated TEXT,
            data TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node_id TEXT,
            timestamp TEXT,
            water_level_m REAL,
            rainfall_24h_mm REAL,
            soil_moisture_pct REAL,
            flow_velocity_ms REAL,
            turbidity_ntu REAL,
            pressure_hpa REAL,
            pressure_trend_hpa_per_hr REAL,
            alert_level TEXT
        )
    ''')
    # Migration for databases created before the hardware refresh added the
    # BMP180. flow_velocity_ms / turbidity_ntu are deliberately LEFT in place:
    # historical rows still hold real readings and dropping the columns would
    # destroy them. Nothing writes them any more.
    cursor.execute("PRAGMA table_info(telemetry)")
    existing = {row[1] for row in cursor.fetchall()}
    for col in ("pressure_hpa", "pressure_trend_hpa_per_hr"):
        if col not in existing:
            cursor.execute(f"ALTER TABLE telemetry ADD COLUMN {col} REAL")
            log.info("migrated telemetry: added %s", col)
    # Every history query is "latest N rows for one node" — without this index
    # that is a full scan of a table that grows by 17k rows/node/day.
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_telemetry_node_ts
        ON telemetry(node_id, timestamp DESC)
    ''')
    conn.commit()
    conn.close()
    log.info("Database initialized at %s", DB_PATH)


def update_node_state(node_id, state):
    conn = get_db()
    cursor = conn.cursor()

    alert_level = state.get("alert_level", "NORMAL")
    probs = json.dumps(state.get("probabilities", {}))
    last_updated = state.get("last_updated", datetime.now(timezone.utc).isoformat())
    data_str = json.dumps(state)

    cursor.execute('''
        INSERT INTO nodes (node_id, alert_level, probabilities, last_updated, data)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(node_id) DO UPDATE SET
            alert_level=excluded.alert_level,
            probabilities=excluded.probabilities,
            last_updated=excluded.last_updated,
            data=excluded.data
    ''', (node_id, alert_level, probs, last_updated, data_str))
    conn.commit()
    conn.close()


def insert_telemetry(node_id, data, alert_level):
    conn = get_db()
    cursor = conn.cursor()
    ts = data.get("last_updated", datetime.now(timezone.utc).isoformat())
    cursor.execute('''
        INSERT INTO telemetry (
            node_id, timestamp, water_level_m, rainfall_24h_mm,
            soil_moisture_pct, pressure_hpa, pressure_trend_hpa_per_hr,
            alert_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        node_id,
        ts,
        data.get("water_level_m"),
        data.get("rainfall_24h_mm"),
        data.get("soil_moisture_pct"),
        data.get("pressure_hpa"),
        data.get("pressure_trend_hpa_per_hr"),
        alert_level
    ))
    conn.commit()
    conn.close()


def get_all_nodes():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM nodes")
    rows = cursor.fetchall()
    conn.close()
    return [json.loads(row["data"]) for row in rows]


def get_node(node_id):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM nodes WHERE node_id = ?", (node_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row["data"])
    return None


def get_node_history(node_id, limit=20):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp as ts, water_level_m, rainfall_24h_mm,
               soil_moisture_pct, pressure_hpa, pressure_trend_hpa_per_hr,
               alert_level
        FROM telemetry
        WHERE node_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (node_id, limit))
    rows = cursor.fetchall()
    conn.close()

    # Return chronologically (oldest first)
    history = [dict(row) for row in rows]
    history.reverse()
    return history
