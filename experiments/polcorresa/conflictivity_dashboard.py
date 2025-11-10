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

# -------- Paths --------
REPO_ROOT = Path(__file__).resolve().parents[2]
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
        else:
            t = f"<b>{len(ap_data)} APs at this location</b><br><br>"
            for i, (n, conf, cli, util) in enumerate(ap_data):
                t += f"<b>{n}</b><br>  Conflictivity: {conf:.3f}"
                if cli is not None:
                    t += f" | Clients: {int(cli)}"
                if util is not None and not np.isnan(util):
                    t += f" | Radio: {util:.1f}%"
                if i < len(ap_data) - 1:
                    t += "<br>"
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
st.plotly_chart(fig, use_container_width=True)

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
    "🟢 Low ↔ 🔴 High (0–1)"
)
