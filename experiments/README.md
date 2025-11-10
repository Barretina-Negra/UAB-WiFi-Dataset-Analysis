# Conflictivity Dashboard

Static Streamlit dashboard to visualize Wi‑Fi conflictivity across UAB campus APs or aggregated building groups.

## What is "Conflictivity"?
Heuristic load/pressure indicator combining client density and radio utilization.

```
conflictivity = 0.6 * normalized_client_count + 0.4 * (max_radio_utilization / 100)
```

Where normalized_client_count is min-max scaled per snapshot (0.5 fallback if all equal).

## Features
* Snapshot selector (latest AP-info-v2 JSON auto-selected).
* Aggregation: per AP or building prefix extracted from the AP name (`AP-VET71` → `VET`).
* Map types: heatmap or point bubbles (Plotly + OpenStreetMap).
* Filters: building codes, minimum conflictivity threshold, heatmap radius.
* Ranking: top-N most conflictive places.
* Color scale inverted RdYlGn (red = higher conflict, green = lower).

## Run

You have three ways to review the dashboard, depending on your environment.

### 1) Streamlit (original UI)

Ensure dependencies. If you use a virtual environment, activate it first.

```bash
pip install -r requirements.txt
streamlit run dashboard/conflictivity_dashboard.py
```

Then open the local URL that Streamlit prints (usually http://localhost:8501).

Note: On macOS with Python 3.14, installing `streamlit` may fail due to `pyarrow` build issues. If that happens, use the Dash route below or create a Python 3.11 env.

### 2) Static preview (no server)

Generate interactive HTML maps from the latest snapshot:

```bash
python scripts/preview_conflictivity_map.py
```

This writes HTML files under `preview/` (open them in a browser).

### 3) Dash (fallback UI, no pyarrow needed)

Run the Dash app (works on Python 3.14) with:

```bash
. .venv/bin/activate
python dashboard/dash_app.py
```

Then open http://127.0.0.1:8050

Dependencies: `dash`, `plotly`, `pandas` (already in `requirements.txt`). No GDAL/pyarrow required.

## Data Expectations
* AP snapshots: `realData/ap/AP-info-v2-*.json` (list of objects, each with `name`, `client_count`, `radios` array having `utilization`).
* Geolocation: `realData/geoloc/aps_geolocalizados_wgs84.geojson` with `properties.USER_NOM_A` matching AP `name` and Point geometry.

If geolocation or snapshots are missing the app stops with an explanatory message.

## Adjusting the Metric
Edit `conflictivity_dashboard.py` to tweak weights or include additional factors (e.g. memory, cpu utilization). Keep the resulting score in [0,1] for color coherence.

## Next Ideas
* Time slider over multiple snapshots for temporal animation.
* Real-time ingestion via a lightweight API/websocket.
* Alert highlighting (blinking) when conflictivity > threshold.
* Persist historical aggregates; show sparkline per place.
* Alternative normalization (z-score or quantile scaling) to reduce outlier impact.
* Add building floor differentiation using elevation / separate layers.

## Troubleshooting
* Empty map: check filters (group selection and min conflictivity) or that snapshot file contains AP names present in the geojson.
* Missing colors: ensure conflictivity column is present; re-run after editing code.
* Performance: If snapshot is huge, consider sampling or pre-aggregating groups before rendering.

---
Maintained within the hackathon analysis repository. This is a static prototype; do not rely on the metric for production decisions.
