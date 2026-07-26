# Running 4 TrueFlood Nodes Together on Wokwi

Wokwi projects simulate ONE ESP32 board each — you can't put 4 boards in one
project file. To get 4 nodes "working together," run 4 separate Wokwi projects
that all publish to the same public HiveMQ broker under different NODE_IDs. The
backend already subscribes to the wildcard `flood/sensor/+` so it picks up all of
them automatically.

## Steps (repeat 4 times, once per node)

1. Go to [wokwi.com](https://wokwi.com) → **New Project** → **ESP32**.
2. Delete the default `diagram.json` content and paste in
   `firmware/diagram_node<N>.json` from this repo.
3. Paste `firmware/sketch.ino` into the code editor.
4. Change exactly two lines at the top of the sketch:
   ```cpp
   #define NODE_ID    "node-3"          // node-1, node-2, node-3, node-4
   #define NODE_LABEL "River Station C" // any readable name
   ```
5. Click the green **Run / Play** button. Watch the Serial Monitor for
   `[MQTT] Connected` — if it doesn't connect, Wokwi-GUEST wifi can be flaky;
   click Restart.
6. Save the project (top-left) and copy its share URL — paste that URL into the
   matching "Node N" tab on the dashboard's **Wokwi** panel (Topology tab).
7. Repeat for node-2, node-3, node-4 in **separate browser tabs**, running
   simultaneously. Leave all 4 tabs open — closing a tab stops that node's
   simulated hardware from publishing.

## Verifying it worked

- Open the dashboard's **Live** tab — you should see 4 node cards appear within
  ~5–10 s of each Wokwi sim starting (`PUBLISH_MS` is 5000 ms per `sketch.ino`).
- `GET /api/nodes` on the backend should list all 4 `node_id`s with fresh
  `last_updated` timestamps.

## Diagram files in this repo

| File | Use for |
|------|---------|
| `diagram_node1.json` | Node 1 (node-1 / River Station A) |
| `diagram_node2.json` | Node 2 (node-2 / River Station B) |
| `diagram_node3.json` | Node 3 (node-3 / River Station C) |
| `diagram_node4.json` | Node 4 (node-4 / River Station D) |

All diagrams are identical in hardware; the only difference between nodes is the
`NODE_ID` and `NODE_LABEL` constants in `sketch.ino`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| No `[MQTT] Connected` in Serial Monitor | Wokwi GUEST wifi unavailable | Click Restart; try a different browser |
| Node card appears, then disappears | Another Wokwi tab with the SAME `NODE_ID` | Ensure each tab uses a unique NODE_ID |
| `NORMAL` forever even with max potentiometer | Feature values not in ALERT range | Push water_level slider past 8m AND rainfall past 120mm |
| Twilio error 63016 on Send Alert | Phone hasn't joined sandbox | Send `join <sandbox-code>` to +1 415 523 8886 on WhatsApp |
