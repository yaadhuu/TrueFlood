# 🌊 FloodSense – AI-Powered Multi-Node Hydro-Telemetry Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![MQTT](https://img.shields.io/badge/MQTT-HiveMQ-orange?logo=mqtt&logoColor=white)](https://www.hivemq.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38bdf8?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)

**FloodSense** is an end-to-end, production-grade IoT and Machine Learning platform designed for real-time flood monitoring, risk classification, and alert broadcasting. Utilizing a multi-node ESP32 telemetry mesh, a Python/Flask MQTT streaming server, a Random Forest classification model, and a glassmorphic real-time dashboard, FloodSense provides immediate local hazard detection and automatic cellular alert escalation.

---

## 🔗 Live Deployments

*   **Live Dashboard (Frontend):** *[Insert your Vercel / GitHub Pages deployment link here]*
*   **Live Prediction Server (Backend API):** *[Insert your Render / Railway deployment link here]*
*   **ESP32 Telemetry Simulator (Wokwi Node-1):** *[Insert your Wokwi project link here]*

---

## 📐 System Architecture

The following diagram illustrates the data ingestion and inference pipeline of the FloodSense system:

```text
  ┌────────────────────────────────────────────────────────┐
  │                   ESP32 telemetry nodes                │
  │  [ Node 1: Station A ]  [ Node 2: Station B ]  [ ... ]  │
  └───────────────────────────┬────────────────────────────┘
                              │
                              │ WiFi / MQTT (JSON payload)
                              ▼
                     ┌──────────────────┐
                     │  HiveMQ Broker   │ (WSS Port: 8884 | TCP Port: 1883)
                     └────────┬─────────┘
                              │
             ┌────────────────┴────────────────┐
             │                                 │
             ▼ (MQTT WebSocket Feed)           ▼ (MQTT TCP wildcards: flood/sensor/+)
   ┌────────────────────┐            ┌────────────────────┐
   │ Glassmorphic UI    │            │ Python MQTT Bridge │ (Flask backend)
   │ (index.html)       │            │ (mqtt_bridge.py)   │
   │                    │            └─────────┬──────────┘
   │ • Live telemetry   │                      │
   │   sparklines       │                      ├─► [ ML Inference Engine (predict.py) ]
   │ • State indicators │                      │   • Random Forest (NORMAL/WATCH/ALERT)
   │ • Historical logs  │                      │
   └─────────▲──────────┘                      ├─► Publish alert to topic: flood/alert/<node_id>
             │                                 │
             └─────── REST Sync Fallback ──────┴─► Twilio WhatsApp alert dispatch (cooldown)
                       (CORS GET /api/nodes)
```

---

## 🛠️ Tech Stack

*   **Firmware & Hardware:** C++ (ESP32), Wokwi Simulation, DHT22 (Temp/Humid), Soil Moisture Sensor, Rain Gauge, Turbidity Sensor, Liquid Flow Meter.
*   **Message Broker:** HiveMQ Cloud (MQTT over WebSockets & TCP).
*   **Backend Server:** Python 3.10+, Flask (REST API), Flask-CORS (Cross-Origin Resource Sharing), Gunicorn (WSGI Server).
*   **Machine Learning:** Scikit-Learn (Random Forest Classifier), Pandas, Joblib.
*   **Alert Escalation:** Twilio Messaging API (WhatsApp Sandbox Channel).
*   **Frontend UI:** HTML5, Tailwind CSS, Chart.js (Interactive rolling sparklines), MQTT.js client.

---

## 📂 Codebase Layout

```text
flood-sense-monorepo/
├── backend/                  # Python API & MQTT ingest server
│   ├── .env.example          # Environment configuration blueprint
│   ├── mqtt_bridge.py        # MQTT listener & REST API service
│   ├── predict.py            # Feature engineering & ML prediction pipeline
│   └── requirements.txt      # Backend dependencies
├── ml_pipeline/              # Machine learning training and cleaning
│   ├── Cleaning.py           # Ingests IMD telemetry & cleans datasets
│   ├── flood_dataset_2021_final.csv   # Model training data
│   ├── flood_ml_training2.py # Random Forest model trainer
│   └── flood_model.joblib    # Serialized model artifact
├── frontend/                 # Client visualization
│   └── index.html            # Premium glassmorphic interface
├── firmware/                 # ESP32 C++ source files
│   ├── sketch.ino            # ESP32 WiFi & telemetry generator
│   ├── diagram_node1.json    # Wokwi simulation diagram for Node-1
│   ├── diagram_node2.json    # Wokwi simulation diagram for Node-2
│   └── libraries.txt         # Wokwi library installer config
├── deployment/               # Cloud orchestration
│   └── Procfile              # Gunicorn configuration for Render / Railway
├── requirements.txt          # Global project requirements
├── Procfile                  # Global root Procfile (Render fallback)
└── README.md                 # Interactive portfolio showcase
```

---

## 🚀 Local Setup Instructions

### 1. Clone & Re-organize
```bash
git clone <your-repo-link>
cd flood-sense-monorepo
```

### 2. Configure Environment Variables
Copy the template `.env.example` in `/backend` to a local `.env` file:
```bash
cp backend/.env.example backend/.env
```
Open `backend/.env` and update the values:
*   `TWILIO_ACCOUNT_SID` & `TWILIO_AUTH_TOKEN`: Find these on your [Twilio Console](https://www.twilio.com/console).
*   `TWILIO_FROM`: Sandbox WhatsApp Sender number (e.g., `whatsapp:+14155238886`).
*   `TWILIO_TO`: Your WhatsApp number verified in Sandbox (e.g., `whatsapp:+919876543210`).
*   `MQTT_BROKER` & `MQTT_PORT`: HiveMQ server details (defaults to public broker).

### 3. Run Backend API Server
Install dependencies and run the backend bridge:
```bash
pip install -r requirements.txt
python backend/mqtt_bridge.py
```
The console will start the MQTT loop in the background and expose the REST endpoints on:
`http://localhost:8080`

### 4. Run Frontend Dashboard
Open `frontend/index.html` directly in a browser or host it via a local static server:
```bash
# Optional: serve using Python's http server
python -m http.server 3000 --directory frontend
```
Open `http://localhost:3000` in your browser. Configure the API endpoint in the top bar to connect to your running backend (`http://localhost:8080`).

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/health` | `GET` | Server health diagnostics and MQTT broker connection status |
| `/api/nodes` | `GET` | List of all registered sensor nodes, current states, and ML classification |
| `/api/nodes/<node_id>` | `GET` | Focus details and 20-point rolling history for a specific station |
| `/api/history/<node_id>` | `GET` | Fetch raw telemetry trends of a single station for chart integration |

---

## ⚙️ Wokwi Hardware Simulation Setup

Each simulated node is constructed using the following configuration:
*   **Sensors:** Slide potentiometers represent soil moisture, rain, turbidity, and water velocity. A DHT22 sensor generates environment stats.
*   **Controller:** ESP32 board.
*   **Code:** Locate C++ code in `firmware/sketch.ino`.
*   **Node IDs:** Open the code in Wokwi and alter the lines below for multiple node deployment:
    ```cpp
    #define NODE_ID    "node-1"          // Change to "node-2", "node-3" per station
    #define NODE_LABEL "River Station A" // Name displayed on dashboard
    ```

---

## ☁️ Cloud Deployment Guidelines

### Backend Deployment (Render / Railway)
1. Link your repository to Render or Railway.
2. Select **Web Service** and choose **Python** as the runtime.
3. Configure the build commands:
   *   **Build Command:** `pip install -r requirements.txt`
   *   **Start Command:** `gunicorn --chdir backend mqtt_bridge:app --bind 0.0.0.0:$PORT`
4. Set up your **Environment Variables** in the service settings matching the variables inside `backend/.env.example`.

### Frontend Deployment (Vercel / GitHub Pages)
*   **Vercel:** Link your repository, specify the root directory as `frontend` (or set `frontend` as the output directory), and deploy.
*   **GitHub Pages:** Publish the content of `/frontend` directly to your `gh-pages` branch.
