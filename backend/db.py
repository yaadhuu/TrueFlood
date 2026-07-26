import sqlite3
import json
import logging
from datetime import datetime, timezone

log = logging.getLogger("floodsense.db")

DB_PATH = "flood_data.db"

def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cursor = conn.cursor()
    # Table for latest state of each node
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            alert_level TEXT,
            probabilities TEXT,
            last_updated TEXT,
            data TEXT
        )
    ''')
    # Table for historical telemetry
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
            alert_level TEXT
        )
    ''')
    conn.commit()
    conn.close()
    log.info("Database initialized.")

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
            soil_moisture_pct, flow_velocity_ms, turbidity_ntu, alert_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        node_id,
        ts,
        data.get("water_level_m"),
        data.get("rainfall_24h_mm"),
        data.get("soil_moisture_pct"),
        data.get("flow_velocity_ms"),
        data.get("turbidity_ntu"),
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
        SELECT timestamp as ts, water_level_m, rainfall_24h_mm, flow_velocity_ms, alert_level
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
