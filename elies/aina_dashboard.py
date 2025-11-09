"""
Enhanced Conflictivity Dashboard with Time Series Navigation

Purpose
- Show a heatmap of Wi-Fi conflictivity by AP on the UAB campus with time series navigation.
- Improved visualization with proper color layering and optimized heatmap display.

Data sources
- AP snapshots: realData/ap/AP-info-v2-*.json (fields include name, client_count, radios[].utilization, group_name, etc.)
- AP geolocations: realData/geoloc/aps_geolocalizados_wgs84.geojson (properties.USER_NOM_A = AP name, geometry.coordinates = [lon, lat])

Conflictivity metric (based on ap_problem_analysis.ipynb with radio utilization interpretation)
- Uses norm01() normalization for most metrics, with special handling for radio utilization
- Radio Utilization Scoring (non-linear based on Wi-Fi standards):
  * < 20%: Light (score 0.0-0.3) - Good conditions for throughput
  * 20-50%: Moderate (score 0.3-0.7) - Performance may degrade under load
  * > 50%: High (score 0.7-1.0) - Increased latency/packet loss likely
- Multi-factor scoring system:
  * Radio utilization (85%): Primary indicator of channel congestion and air quality
  * Client load (10%): Normalized number of connected clients
  * CPU utilization (2%): Processing overhead
  * Memory usage (3%): Available resources
- Final score normalized to [0,1] range where 1 = highest conflictivity

UI features
- Time series slider to navigate through snapshots chronologically
- Aggregation by AP (default and only option)
- Optimized heatmap visualization with proper color layering
- Filters: group prefix, minimum conflictivity threshold
- Top N most conflictive table

Run
  streamlit run dashboard/conflictivity_dashboard.py
"""

# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# -------- Paths --------
REPO_ROOT = Path(__file__).resolve().parents[1]
AP_DIR = REPO_ROOT / "realData" / "ap"
GEOJSON_PATH = REPO_ROOT / "realData" / "geoloc" / "aps_geolocalizados_wgs84.geojson"

# -------- Helpers --------
def norm01(series: pd.Series, invert: bool = False) -> pd.Series:
    """Simple min-max, falls back to 0.5 when no variance."""
    s = series.astype(float)
    rng = s.max() - s.min()
    if rng == 0 or np.isinf(rng) or np.isnan(rng):
        return pd.Series(0.5, index=s.index)
    n = (s - s.min()) / rng
    return 1 - n if invert else n

def extract_group(ap_name: Optional[str]) -> Optional[str]:
    if not isinstance(ap_name, str):
        return None
    m = re.match(r"^AP-([A-Za-z]+)", ap_name)
    return m.group(1) if m else None

def find_snapshot_files(ap_dir: Path) -> List[Tuple[Path, datetime]]:
    files = list(ap_dir.glob("AP-info-v2-*.json"))
    files_with_time = []
    for f in files:
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})", f.name)
        if m:
            y, mo, d, h, mi, s = map(int, m.groups())
            files_with_time.append((f, datetime(y, mo, d, h, mi, s)))
    files_with_time.sort(key=lambda x: x[1])
    return files_with_time

# --- New scoring utilities ----------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def airtime_score(util: float, band: str) -> float:
    """
    Map channel utilization % to [0,1] pain score.
    Stricter on 2.4 GHz because overlap/legacy effects.
    """
    u = clamp(util or 0.0, 0.0, 100.0)
    if band == "2g":
        # 2.4 GHz thresholds
        if u <= 10:
            return 0.05 * (u / 10.0)  # 0–0.05
        if u <= 25:
            return 0.05 + 0.35 * ((u - 10) / 15.0)  # 0.05–0.40
        if u <= 50:
            return 0.40 + 0.35 * ((u - 25) / 25.0)  # 0.40–0.75
        return 0.75 + 0.25 * ((u - 50) / 50.0)      # 0.75–1.00
    else:
        # 5 GHz thresholds
        if u <= 15:
            return 0.05 * (u / 15.0)  # 0–0.05
        if u <= 35:
            return 0.05 + 0.35 * ((u - 15) / 20.0)  # 0.05–0.40
        if u <= 65:
            return 0.40 + 0.35 * ((u - 35) / 30.0)  # 0.40–0.75
        return 0.75 + 0.25 * ((u - 65) / 35.0)      # 0.75–1.00

def client_pressure_score(n_clients: float, peers_p95: float) -> float:
    """
    Log-style pressure: small increases at low counts, stronger near heavy loads.
    Normalized against the 95th percentile of current snapshot to reduce outlier bias.
    """
    n = max(0.0, float(n_clients or 0.0))
    denom = max(1.0, float(peers_p95 or 1.0))
    x = math.log1p(n) / math.log1p(denom)  # 0–1
    return clamp(x, 0.0, 1.0)

def cpu_health_score(cpu_pct: float) -> float:
    """Begin to penalize after ~70%. 90%+ is severe."""
    c = clamp(cpu_pct or 0.0, 0.0, 100.0)
    if c <= 70:
        return 0.0
    if c <= 90:
        return 0.6 * ((c - 70) / 20.0)  # 0–0.6
    return 0.6 + 0.4 * ((c - 90) / 10.0)  # 0.6–1.0

def mem_health_score(mem_used_pct: float) -> float:
    """Penalize after 80% used. 95%+ is severe."""
    m = clamp(mem_used_pct or 0.0, 0.0, 100.0)
    if m <= 80:
        return 0.0
    if m <= 95:
        return 0.6 * ((m - 80) / 15.0)  # 0–0.6
    return 0.6 + 0.4 * ((m - 95) / 5.0)  # 0.6–1.0

def read_ap_snapshot(path: Path, band_mode: str = "worst") -> pd.DataFrame:
    """
    band_mode:
      - "worst": max(max_2g, max_5g)  [recommended]
      - "2.4GHz": max of band=0
      - "5GHz":   max of band=1
      - "avg": average of band maxima
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for ap in data:
        name = ap.get("name")
        client_count = ap.get("client_count", 0)
        cpu_util = ap.get("cpu_utilization", 0)
        mem_free = ap.get("mem_free", 0)
        mem_total = ap.get("mem_total", 0)
        group_name = ap.get("group_name")
        site = ap.get("site")
        radios = ap.get("radios") or []

        util_2g = []
        util_5g = []
        for r in radios:
            u = r.get("utilization")
            band = r.get("band")
            if u is None:
                continue
            if band == 0:
                util_2g.append(float(u))
            elif band == 1:
                util_5g.append(float(u))

        max_2g = max(util_2g) if util_2g else np.nan
        max_5g = max(util_5g) if util_5g else np.nan

        if band_mode == "2.4GHz":
            agg_util = max_2g
        elif band_mode == "5GHz":
            agg_util = max_5g
        elif band_mode == "avg":
            parts = [x for x in [max_2g, max_5g] if not np.isnan(x)]
            agg_util = float(np.mean(parts)) if parts else np.nan
        else:  # "worst"
            agg_util = np.nanmax([max_2g, max_5g])

        rows.append(
            {
                "name": name,
                "group_name": group_name,
                "site": site,
                "client_count": pd.to_numeric(client_count, errors="coerce"),
                "cpu_utilization": pd.to_numeric(cpu_util, errors="coerce"),
                "mem_free": pd.to_numeric(mem_free, errors="coerce"),
                "mem_total": pd.to_numeric(mem_total, errors="coerce"),
                "util_2g": max_2g,
                "util_5g": max_5g,
                "agg_util": agg_util,
            }
        )

    df = pd.DataFrame(rows)

    # Sanitize numerics
    for c in ["client_count", "cpu_utilization", "mem_free", "mem_total", "util_2g", "util_5g", "agg_util"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Memory used %
    df["mem_used_pct"] = (1 - (df["mem_free"] / df["mem_total"])).clip(0, 1) * 100
    df["mem_used_pct"] = df["mem_used_pct"].fillna(0)

    # Choose band weights for airtime score when both present
    # 2.4 GHz counts more (noisy band)
    w_2g = 0.6
    w_5g = 0.4

    # Airtime scores per band
    df["air_s_2g"] = df["util_2g"].apply(lambda u: airtime_score(u, "2g") if not np.isnan(u) else np.nan)
    df["air_s_5g"] = df["util_5g"].apply(lambda u: airtime_score(u, "5g") if not np.isnan(u) else np.nan)

    # Aggregated airtime score:
    # - worst mode: max of band scores (true worst case)
    # - avg mode: weighted average of available band scores
    if band_mode in ("2.4GHz", "5GHz"):
        df["airtime_score"] = np.where(
            band_mode == "2.4GHz", df["air_s_2g"], df["air_s_5g"]
        )
    elif band_mode == "avg":
        df["airtime_score"] = (
            (df["air_s_2g"].fillna(0) * w_2g + df["air_s_5g"].fillna(0) * w_5g)
            / ((~df["air_s_2g"].isna()) * w_2g + (~df["air_s_5g"].isna()) * w_5g).replace(0, np.nan)
        )
    else:  # worst
        df["airtime_score"] = np.nanmax(np.vstack([df["air_s_2g"].fillna(-1), df["air_s_5g"].fillna(-1)]), axis=0)
        df["airtime_score"] = df["airtime_score"].where(df["airtime_score"] >= 0, np.nan)

    # Client pressure normalized to snapshot 95th percentile
    p95_clients = float(np.nanpercentile(df["client_count"].fillna(0), 95)) if len(df) else 1.0
    df["client_score"] = df["client_count"].apply(lambda n: client_pressure_score(n, p95_clients))

    # Resource health
    df["cpu_score"] = df["cpu_utilization"].apply(cpu_health_score)
    df["mem_score"] = df["mem_used_pct"].apply(mem_health_score)

    # Idle-with-no-clients relief: if airtime is high but clients==0, reduce pain a bit
    # This distinguishes neighbor interference from true contention under load.
    def relief(a_score: float, clients: float) -> float:
        if np.isnan(a_score):
            return np.nan
        if (clients or 0) > 0:
            return a_score
        # up to 20% relief when no clients
        return a_score * 0.8

    df["airtime_score_adj"] = [
        relief(a, c) for a, c in zip(df["airtime_score"], df["client_count"])
    ]

    # Final conflictivity weights (sum ≈ 1.0)
    W_AIR = 0.75   # channel busy/quality dominates
    W_CL  = 0.15   # how many users we may hurt
    W_CPU = 0.05   # AP CPU health
    W_MEM = 0.05   # AP memory health

    # Combine; missing airtime -> treat as 0.4 (neutral-ish) to avoid rewarding missing data
    df["airtime_score_filled"] = df["airtime_score_adj"].fillna(0.4)

    df["conflictivity"] = (
        df["airtime_score_filled"] * W_AIR
        + df["client_score"].fillna(0) * W_CL
        + df["cpu_score"].fillna(0) * W_CPU
        + df["mem_score"].fillna(0) * W_MEM
    ).clip(0, 1)

    # For display
    df["max_radio_util"] = df["agg_util"].fillna(0)  # keep name expected by UI
    df["group_code"] = df["name"].apply(extract_group)
    return df

def read_geoloc_points(geojson_path: Path) -> pd.DataFrame:
    with geojson_path.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    rows = []
    for ft in feats:
        props = (ft or {}).get("properties", {})
        geom = (ft or {}).get("geometry", {})
        if (geom or {}).get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        name = props.get("USER_NOM_A")
        if not name:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        rows.append({"name": name, "lon": lon, "lat": lat})
    return pd.DataFrame(rows)

def create_optimized_heatmap(
    df: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    min_conflictivity: float = 0.0,
    radius: int = 15,
    zoom: int = 15,
) -> go.Figure:
    df_with_location = df.copy()
    df_with_location["location_key"] = (
        df_with_location["lat"].round(6).astype(str)
        + ","
        + df_with_location["lon"].round(6).astype(str)
    )

    location_groups = df_with_location.groupby("location_key").agg(
        lat=("lat", "first"),
        lon=("lon", "first"),
        name=("name", lambda x: list(x)),
        conflictivity=("conflictivity", lambda x: list(x)),
        client_count=("client_count", lambda x: list(x) if "client_count" in df.columns else None),
        max_radio_util=("max_radio_util", lambda x: list(x) if "max_radio_util" in df.columns else None),
    ).reset_index()

    location_groups["max_conflictivity"] = location_groups["conflictivity"].apply(max)
    location_groups["ap_count"] = location_groups["name"].apply(len)
    
    # Filter out locations below minimum conflictivity
    location_groups = location_groups[location_groups["max_conflictivity"] >= min_conflictivity]
    
    location_groups = location_groups.sort_values("max_conflictivity", ascending=True)

    hover_texts = []
    ap_names_list = []  # Store AP names for selection
    for _, row in location_groups.iterrows():
        ap_data = sorted(
            zip(
                row["name"],
                row["conflictivity"],
                (row["client_count"] or [None] * len(row["name"])),
                (row["max_radio_util"] or [None] * len(row["name"])),
            ),
            key=lambda x: x[1],
            reverse=True,
        )
        if len(ap_data) == 1:
            n, conf, cli, util = ap_data[0]
            t = f"<b>{n}</b><br>Conflictivity: {conf:.3f}"
            if cli is not None:
                t += f"<br>Clients: {int(cli)}"
            if util is not None and not np.isnan(util):
                t += f"<br>Radio Util: {util:.1f}%"
            ap_names_list.append([n])  # Single AP
        else:
            t = f"<b>{len(ap_data)} APs at this location</b><br><br>"
            names_at_location = []
            for i, (n, conf, cli, util) in enumerate(ap_data):
                t += f"<b>{n}</b><br>  Conflictivity: {conf:.3f}"
                if cli is not None:
                    t += f" | Clients: {int(cli)}"
                if util is not None and not np.isnan(util):
                    t += f" | Radio: {util:.1f}%"
                if i < len(ap_data) - 1:
                    t += "<br>"
                names_at_location.append(n)
            ap_names_list.append(names_at_location)  # Multiple APs
        hover_texts.append(t)

    fig = go.Figure(
        go.Scattermapbox(
            lat=location_groups["lat"],
            lon=location_groups["lon"],
            mode="markers",
            marker=dict(
                size=radius * 2,
                color=location_groups["max_conflictivity"],
                colorscale=[
                    [0.0, "rgb(0, 255, 0)"],
                    [0.5, "rgb(255, 165, 0)"],
                    [1.0, "rgb(255, 0, 0)"],
                ],
                cmin=0,
                cmax=1,
                opacity=0.85,
                showscale=True,
                colorbar=dict(
                    title="Conflictivity",
                    thickness=15,
                    len=0.7,
                    tickmode="linear",
                    tick0=0,
                    dtick=0.2,
                    tickformat=".1f",
                ),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
            customdata=ap_names_list,  # Store AP names for selection
        )
    )

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=700,
    )
    return fig

# -------- UI --------
st.set_page_config(page_title="UAB Wi‑Fi Conflictivity", page_icon="📶", layout="wide")
st.title("UAB Wi‑Fi Conflictivity Dashboard")
st.caption("Time series visualization • Heatmap of conflictivity by Access Point")

# Data availability checks
if not AP_DIR.exists():
    st.error(f"AP directory not found: {AP_DIR}")
    st.stop()
if not GEOJSON_PATH.exists():
    st.error(f"GeoJSON not found: {GEOJSON_PATH}")
    st.stop()

snapshots = find_snapshot_files(AP_DIR)
if not snapshots:
    st.warning("No AP snapshots found in realData/ap. Please add AP-info-v2-*.json files.")
    st.stop()

geo_df = read_geoloc_points(GEOJSON_PATH)

# Sidebar
with st.sidebar:
    st.header("Time Navigation")
    default_idx = len(snapshots) - 1
    selected_idx = st.slider(
        "Select Time",
        min_value=0,
        max_value=len(snapshots) - 1,
        value=default_idx,
        format="",
        help="Slide to navigate through time series data",
    )
    selected_path, selected_dt = snapshots[selected_idx]
    st.info(f"📅 **{selected_dt.strftime('%Y-%m-%d')}**\n\n⏰ **{selected_dt.strftime('%H:%M:%S')}**")

    first_dt = snapshots[0][1]
    last_dt = snapshots[-1][1]
    st.caption(f"Available data: {first_dt.strftime('%Y-%m-%d %H:%M')} to {last_dt.strftime('%Y-%m-%d %H:%M')}")
    st.caption(f"Total snapshots: {len(snapshots)}")

    st.divider()
    st.header("Visualization Settings")
    band_mode = st.radio(
        "Band Mode",
        options=["worst", "avg", "2.4GHz", "5GHz"],
        index=0,
        help="worst: max(max_2.4, max_5) • avg: weighted average of band maxima",
        horizontal=True,
    )
    radius = 5  # Fixed heatmap radius
    min_conf = st.slider("Minimum conflictivity", 0.0, 1.0, 0.0, 0.01)
    top_n = st.slider("Top N listing (table)", 5, 50, 15, step=5)

# Load and compute
ap_df = read_ap_snapshot(selected_path, band_mode=band_mode)
merged = ap_df.merge(geo_df, on="name", how="inner")
if merged.empty:
    st.info("No APs have geolocation data.")
    st.stop()

# Group filter
available_groups = sorted({g for g in merged["name"].apply(extract_group).dropna().unique().tolist()})
with st.sidebar:
    st.divider()
    st.header("Filters")
    selected_groups = st.multiselect(
        "Filter by building code",
        options=available_groups,
        default=available_groups,
    )

if selected_groups:
    merged = merged[merged["name"].apply(extract_group).isin(selected_groups)]
if merged.empty:
    st.info("No APs after applying group filter.")
    st.stop()

map_df = merged.copy()
center_lat = float(map_df["lat"].mean())
center_lon = float(map_df["lon"].mean())

fig = create_optimized_heatmap(
    df=map_df,
    center_lat=center_lat,
    center_lon=center_lon,
    min_conflictivity=min_conf,
    radius=radius,
    zoom=15,
)

# Initialize session state for selected AP
if "selected_ap" not in st.session_state:
    st.session_state.selected_ap = None
if "last_opened_ap" not in st.session_state:
    st.session_state.last_opened_ap = None
if "dialog_just_dismissed" not in st.session_state:
    st.session_state.dialog_just_dismissed = False
if "chart_key" not in st.session_state:
    st.session_state.chart_key = 0

# Function to clear selection when dialog is dismissed
def clear_selection():
    st.session_state.selected_ap = None
    st.session_state.last_opened_ap = None
    st.session_state.dialog_just_dismissed = True
    # Increment chart key to force recreation and clear selection
    st.session_state.chart_key += 1

# Dialog function for AINA AI analysis
@st.dialog("🤖 Anàlisi AINA AI", on_dismiss=clear_selection)
def show_aina_analysis(ap_name: str, ap_row: pd.Series):
    """Show AINA AI analysis in a modal dialog."""
    st.subheader(f"Access Point: {ap_name}")
    
    # Prepare AP data for AI
    util_2g = ap_row.get("util_2g", np.nan)
    util_5g = ap_row.get("util_5g", np.nan)
    client_count = ap_row.get("client_count", 0)
    cpu_util = ap_row.get("cpu_utilization", np.nan)
    mem_free = ap_row.get("mem_free", np.nan)
    mem_total = ap_row.get("mem_total", np.nan)
    mem_used_pct = ap_row.get("mem_used_pct", np.nan)
    conflictivity = ap_row.get("conflictivity", np.nan)
    
    # Format AP data for AI (handle NaN values)
    def format_value(val, format_str="{:.1f}", default="no disponible"):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return format_str.format(val)
    
    # Show AP info
    with st.expander("📊 Dades de l'Access Point", expanded=False):
        st.write(f"- **Nom:** {ap_name}")
        st.write(f"- **Utilització màxima 2.4 GHz:** {format_value(util_2g, '{:.1f}%', 'no disponible')}")
        st.write(f"- **Utilització màxima 5 GHz:** {format_value(util_5g, '{:.1f}%', 'no disponible')}")
        st.write(f"- **Nombre de clients connectats:** {int(client_count) if not (isinstance(client_count, float) and np.isnan(client_count)) else 0}")
        st.write(f"- **Utilització CPU:** {format_value(cpu_util, '{:.1f}%', 'no disponible')}")
        st.write(f"- **Memòria lliure:** {format_value(mem_free, '{:.0f} MB', 'no disponible')}")
        st.write(f"- **Memòria total:** {format_value(mem_total, '{:.0f} MB', 'no disponible')}")
        st.write(f"- **Percentatge de memòria usada:** {format_value(mem_used_pct, '{:.1f}%', 'no disponible')}")
        st.write(f"- **Puntuació de conflictivitat calculada:** {format_value(conflictivity, '{:.3f}', 'no disponible')}")
    
    ap_info_text = f"""Dades de l'Access Point:

- Nom: {ap_name}
- Utilització màxima 2.4 GHz: {format_value(util_2g, '{:.1f}%', 'no disponible')}
- Utilització màxima 5 GHz: {format_value(util_5g, '{:.1f}%', 'no disponible')}
- Nombre de clients connectats: {int(client_count) if not (isinstance(client_count, float) and np.isnan(client_count)) else 0}
- Utilització CPU: {format_value(cpu_util, '{:.1f}%', 'no disponible')}
- Memòria lliure: {format_value(mem_free, '{:.0f} MB', 'no disponible')}
- Memòria total: {format_value(mem_total, '{:.0f} MB', 'no disponible')}
- Percentatge de memòria usada: {format_value(mem_used_pct, '{:.1f}%', 'no disponible')}
- Puntuació de conflictivitat calculada: {format_value(conflictivity, '{:.3f}', 'no disponible')}

"""
    
    # AINA AI API call
    API_KEY = os.getenv("AINA_API_KEY")
    if not API_KEY:
        st.error("❌ AINA_API_KEY no trobada a les variables d'entorn. Si us plau, crea un fitxer .env amb AINA_API_KEY=tu_api_key")
        return
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "UAB-THE-HACK/1.0"
    }
    
    prompt = ap_info_text + """Aquest Access Point es conflictiu, investiga les causes tenint en compte que aquests son els criteris que s'utilitza per calcular conflictivitat:

Aquí tens el nou model de conflictivitat, pas a pas.

Entrades per AP

- util_2g, util_5g: utilització màxima del canal per banda (de radios[].utilization)

- client_count

- cpu_utilization (%)

- mem_used_pct = 100 x (1 - mem_free/mem_total)

1) Malestar d'aire (airtime) per banda

- Mapar la utilització a una puntuació de malestar no lineal en [0,1], més estricta a 2,4 GHz.

2,4 GHz (band="2g")

- 0-10% → 0-0,05

- 10-25% → 0,05-0,40

- 25-50% → 0,40-0,75

- 50-100% → 0,75-1,00

5 GHz (band="5g")

- 0-15% → 0-0,05

- 15-35% → 0,05-0,40

- 35-65% → 0,40-0,75

- 65-100% → 0,75-1,00

2) Agregació de l'airtime entre bandes

- band_mode="worst" (per defecte): airtime_score = max(airtime_2g, airtime_5g)

- band_mode="avg": mitjana ponderada (2,4 GHz 0,6, 5 GHz 0,4)

- band_mode="2.4GHz"/"5GHz": prendre la puntuació d'aquesta banda

3) Alleujament quan no hi ha clients

- Si client_count == 0, reduir airtime_score un 20% per distingir soroll veí de contenció:

  airtime_score_adj = airtime_score x 0,8

- Altrament airtime_score_adj = airtime_score

4) Pressió de clients

- Relativa a la instantània, amb escala logarítmica:

  client_score = log1p(client_count) / log1p(p95_clients)

  on p95_clients és el percentil 95 de clients entre els APs a la instantània seleccionada.

  El resultat es limita a [0,1].

5) Salut de recursos de l'AP

- CPU:

  - ≤70% → 0

  - 70-90% → lineal fins a 0,6

  - 90-100% → lineal fins a 1,0

- Memòria (percentatge usat):

  - ≤80% → 0

  - 80-95% → lineal fins a 0,6

  - 95-100% → lineal fins a 1,0

6) Combinació en conflictivitat

- Omplir airtime_score absent amb 0,4 (neutral-ish) per evitar recompensar dades absents.

- Suma ponderada (retallada a [0,1]):

  conflictivity =

    0,75 x airtime_score_filled +

    0,15 x client_score +

    0,05 x cpu_score +

    0,05 x mem_score

Intuïció

- L'airtime (canal ocupat/qualitat) predomina.

- La pressió puja amb més clients però desacelera a compts baixos (escala log).

- CPU/memòria només importen quan realment estan estressats.

- Es penalitza abans la banda de 2,4 GHz perquè es degrada abans.

- Si un canal està ocupat però no tens clients, encara importa, però una mica menys.

Ara vull que em raonis si l'AP es conflictiu per saturació d'ampla de banda ocupat (a partir de la `radio[].utilization`), per AP saturat (amb massa clients) o per ambdós.

L'AP està dissenyat per gestionar un màxim de 50 clients concurrents. Està massa carregat si s'apropa a supera aquest nombre.

La utilització de banda comença a afectar a partir de 40% de utilització.

Si n'hi ha un numero alt d'ambos, doncs clarament el raonament es ambdos. Pero 20-30 clients un AP pot gestionar facilment.
"""
    
    # Show loading state and call AINA AI
    with st.spinner("🔄 Esperant resposta d'AINA..."):
        payload = {
            "model": "BSC-LT/ALIA-40b-instruct_Q8_0",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7,
        }
        
        try:
            response = requests.post(
                "https://api.publicai.co/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                resposta = data["choices"][0]["message"]["content"]
                st.success("**Resposta d'AINA:**")
                st.markdown(resposta)
            else:
                st.error(f"❌ Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"❌ Error en la petició: {str(e)}")

# Handle map selection - make points selectable
fig.update_layout(clickmode='event+select')
# Use chart_key to force recreation when dialog is dismissed (clears selection)
selected_points = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=f"ap_map_{st.session_state.chart_key}")

# Process selection and open dialog
if selected_points and "selection" in selected_points:
    selection = selected_points["selection"]
    if "points" in selection and len(selection["points"]) > 0:
        # Get the first selected point
        point = selection["points"][0]
        ap_name = None
        
        # Try to get AP name from customdata first
        if "customdata" in point and point["customdata"]:
            ap_names = point["customdata"]
            if isinstance(ap_names, list) and len(ap_names) > 0:
                # customdata is always a list: [ap_name] for single AP or [ap1, ap2, ...] for multiple
                ap_name = ap_names[0] if isinstance(ap_names[0], str) else str(ap_names[0])
        
        # Fallback: try to extract from text/hover if customdata didn't work
        if not ap_name and "text" in point:
            text = point["text"]
            name_match = re.search(r"<b>([^<]+)</b>", text)
            if name_match:
                ap_name = name_match.group(1)
        
        if ap_name:
            # Only open dialog if this is a new selection (different from last opened)
            # and we haven't just dismissed a dialog (to prevent reopening on same selection)
            if ap_name != st.session_state.last_opened_ap and not st.session_state.dialog_just_dismissed:
                st.session_state.selected_ap = ap_name
                st.session_state.last_opened_ap = ap_name
                st.session_state.dialog_just_dismissed = False
                # Find AP data and open dialog
                ap_data = merged[merged["name"] == ap_name]
                if not ap_data.empty:
                    show_aina_analysis(ap_name, ap_data.iloc[0])
            elif st.session_state.dialog_just_dismissed:
                # Reset the flag after processing
                st.session_state.dialog_just_dismissed = False

# Top list
st.subheader("Top conflictive Access Points")
filtered_for_table = map_df[map_df["conflictivity"] >= min_conf].copy()
if filtered_for_table.empty:
    st.info(f"No APs with conflictivity >= {min_conf:.2f}")
else:
    cols = ["name", "group_code", "client_count", "max_radio_util", "conflictivity"]
    cols = [c for c in cols if c in filtered_for_table.columns]
    top_df = (
        filtered_for_table[cols]
        .sort_values("conflictivity", ascending=False)
        .head(top_n)
        .rename(
            columns={
                "name": "Access Point",
                "group_code": "Building",
                "conflictivity": "Conflictivity Score",
                "client_count": "Clients",
                "max_radio_util": "Radio Util % (agg)",
            }
        )
    )
    top_df["Conflictivity Score"] = top_df["Conflictivity Score"].map(lambda x: f"{x:.3f}")
    st.dataframe(top_df, use_container_width=True, hide_index=True)

band_info = {
    "worst": "Worst band (max of 2.4/5 GHz)",
    "avg": "Weighted average of band maxima (2.4:60%, 5:40%)",
    "2.4GHz": "2.4 GHz only",
    "5GHz": "5 GHz only",
}
st.caption(
    f"📻 Band mode: {band_info[band_mode]}  |  "
    "💡 Conflictivity measures Wi-Fi stress by combining channel congestion (75%), number of connected devices (15%), and AP resource usage (10%)  |  "
    "🟢 Low ↔ 🔴 High (0–1)  |  "
    "👆 Selecciona un AP al mapa per analitzar-lo amb AINA AI"
)
