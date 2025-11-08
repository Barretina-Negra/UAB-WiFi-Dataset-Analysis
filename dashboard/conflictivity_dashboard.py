"""
Enhanced Conflictivity Dashboard with Time Series Navigation

Purpose
- Show a heatmap of Wi-Fi conflictivity by AP on the UAB campus with time series navigation.
- Improved visualization with proper color layering and optimized heatmap display.

Data sources
- AP snapshots: realData/ap/AP-info-v2-*.json (fields include name, client_count, radios[].utilization, group_name, etc.)
- AP geolocations: realData/geoloc/aps_geolocalizados_wgs84.geojson (properties.USER_NOM_A = AP name, geometry.coordinates = [lon, lat])

Conflictivity metric
- Combine client load and radio utilization per AP into a [0,1] score:
  score = 0.6 * norm01(client_count) + 0.4 * max_radio_utilization/100
- When min=max for client_count, use 0.5 for that component.

UI features
- Time series slider to navigate through snapshots chronologically
- Aggregation by AP (default and only option)
- Optimized heatmap visualization with proper color layering
- Filters: group prefix, minimum conflictivity threshold
- Top N most conflictive table

Run
  streamlit run dashboard/conflictivity_dashboard.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -------- Paths --------
REPO_ROOT = Path(__file__).resolve().parents[1]
AP_DIR = REPO_ROOT / "realData" / "ap"
GEOJSON_PATH = REPO_ROOT / "realData" / "geoloc" / "aps_geolocalizados_wgs84.geojson"


# -------- Helpers --------
def norm01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.empty:
        return s
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx - mn == 0:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def extract_group(ap_name: Optional[str]) -> Optional[str]:
    if not isinstance(ap_name, str):
        return None
    m = re.match(r"^AP-([A-Za-z]+)", ap_name)
    return m.group(1) if m else None


def find_snapshot_files(ap_dir: Path) -> List[Tuple[Path, datetime]]:
    """Find all snapshot files and parse their timestamps.
    Returns list of (path, datetime) tuples sorted by time."""
    files = list(ap_dir.glob("AP-info-v2-*.json"))
    files_with_time = []
    
    for f in files:
        # Parse timestamp from filename: AP-info-v2-2025-04-03T00_00_01+02_00.json
        match = re.search(r'(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})', f.name)
        if match:
            year, month, day, hour, minute, second = map(int, match.groups())
            dt = datetime(year, month, day, hour, minute, second)
            files_with_time.append((f, dt))
    
    # Sort by datetime
    files_with_time.sort(key=lambda x: x[1])
    return files_with_time


def read_ap_snapshot(path: Path) -> pd.DataFrame:
    """Load one AP snapshot JSON into a DataFrame with selected fields.
    Columns: name, client_count, group_name, site, max_radio_util, conflictivity
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for ap in data:
        name = ap.get("name")
        client_count = ap.get("client_count")
        group_name = ap.get("group_name")
        site = ap.get("site")
        radios = ap.get("radios") or []
        max_util = None
        for r in radios:
            u = r.get("utilization")
            if u is None:
                continue
            max_util = u if max_util is None else max(max_util, u)
        rows.append({
            "name": name,
            "client_count": client_count,
            "group_name": group_name,
            "site": site,
            "max_radio_util": max_util,
        })
    df = pd.DataFrame(rows)
    # sanitize
    if "client_count" in df:
        df["client_count"] = pd.to_numeric(df["client_count"], errors="coerce").fillna(0)
    if "max_radio_util" in df:
        df["max_radio_util"] = pd.to_numeric(df["max_radio_util"], errors="coerce").fillna(0)
    # conflictivity
    df["client_norm"] = norm01(df["client_count"]) if len(df) else []
    df["util_norm"] = df["max_radio_util"].clip(lower=0, upper=100) / 100.0
    df["conflictivity"] = 0.6 * df["client_norm"] + 0.4 * df["util_norm"]
    df["group_code"] = df["name"].apply(extract_group)
    return df


def read_geoloc_points(geojson_path: Path) -> pd.DataFrame:
    """Return DataFrame with columns: name, lon, lat (name from USER_NOM_A)."""
    with geojson_path.open("r", encoding="utf-8") as f:
        gj = json.load(f)
    feats = gj.get("features", [])
    rows = []
    for ft in feats:
        props = (ft or {}).get("properties", {})
        geom = (ft or {}).get("geometry", {})
        if not props or not geom:
            continue
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates")
        if not coords or len(coords) < 2:
            continue
        name = props.get("USER_NOM_A")
        if not name:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        rows.append({"name": name, "lon": lon, "lat": lat})
    return pd.DataFrame(rows)


def aggregate_by_group(df: pd.DataFrame, geo_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate AP rows into building group centroids and average conflictivity.
    Returns: DataFrame with columns group_code, lat, lon, conflictivity, n_aps
    """
    ap_geo = df
    # Ensure coordinates exist; if not, join once with geoloc
    if not {"lat", "lon"}.issubset(ap_geo.columns):
        ap_geo = ap_geo.merge(geo_df, on="name", how="inner")

    # Derive group code if missing
    if "group_code" not in ap_geo.columns:
        ap_geo["group_code"] = ap_geo["name"].apply(extract_group)

    # Keep only rows with valid group and coordinates
    ap_geo = ap_geo.dropna(subset=["group_code", "lat", "lon"])  # keep valid
    if ap_geo.empty:
        return ap_geo

    agg = (
        ap_geo.groupby("group_code")
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            conflictivity=("conflictivity", "mean"),
            n_aps=("name", "count"),
        )
        .reset_index()
    )
    return agg


def create_optimized_heatmap(df: pd.DataFrame, center_lat: float, center_lon: float, 
                              radius: int = 15, zoom: int = 15) -> go.Figure:
    """Create an optimized heatmap with proper color layering (high values on top).
    
    The key improvement is sorting data so high conflictivity values are rendered last,
    ensuring they appear on top of lower values.
    """
    # Sort by conflictivity so high values are plotted last (on top)
    df_sorted = df.sort_values('conflictivity', ascending=True).copy()
    
    # Create figure with density mapbox
    fig = go.Figure(go.Densitymapbox(
        lat=df_sorted['lat'],
        lon=df_sorted['lon'],
        z=df_sorted['conflictivity'],
        radius=radius,
        colorscale=[
            [0.0, 'rgb(0, 255, 0)'],      # Green (low conflictivity)
            [0.5, 'rgb(255, 255, 0)'],    # Yellow (medium)
            [1.0, 'rgb(255, 0, 0)']       # Red (high conflictivity)
        ],
        showscale=True,
        colorbar=dict(
            title="Conflictivity",
            thickness=15,
            len=0.7,
        ),
        hovertemplate='<b>%{text}</b><br>Conflictivity: %{z:.2f}<extra></extra>',
        text=df_sorted['name'],
        zmin=0,
        zmax=1,
    ))
    
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

# Load geolocation data once
geo_df = read_geoloc_points(GEOJSON_PATH)

# Sidebar: time slider and controls
with st.sidebar:
    st.header("Time Navigation")
    
    # Time series slider
    if len(snapshots) > 0:
        # Default to latest
        default_idx = len(snapshots) - 1
        
        # Create readable labels for the slider
        time_labels = [dt.strftime("%Y-%m-%d %H:%M") for _, dt in snapshots]
        
        # Use slider for time selection
        selected_idx = st.slider(
            "Select Time",
            min_value=0,
            max_value=len(snapshots) - 1,
            value=default_idx,
            format="",
            help="Slide to navigate through time series data"
        )
        
        # Display selected timestamp prominently
        selected_path, selected_dt = snapshots[selected_idx]
        st.info(f"📅 **{selected_dt.strftime('%Y-%m-%d')}**\n\n⏰ **{selected_dt.strftime('%H:%M:%S')}**")
        
        # Show time range info
        first_dt = snapshots[0][1]
        last_dt = snapshots[-1][1]
        total_snapshots = len(snapshots)
        st.caption(f"Available data: {first_dt.strftime('%Y-%m-%d %H:%M')} to {last_dt.strftime('%Y-%m-%d %H:%M')}")
        st.caption(f"Total snapshots: {total_snapshots}")
    
    st.divider()
    st.header("Visualization Settings")
    
    # Heatmap radius optimization (optimal value: 15)
    radius = st.slider(
        "Heatmap radius (px)", 
        min_value=5, 
        max_value=40, 
        value=15,
        help="Optimal value: 15px for clear AP differentiation"
    )
    
    min_conf = st.slider("Minimum conflictivity", 0.0, 1.0, 0.0, 0.01)
    top_n = st.slider("Top N listing", 5, 50, 15, step=5)

# Load data for selected timestamp
ap_df = read_ap_snapshot(selected_path)

# Merge AP + geoloc and filter
merged = ap_df.merge(geo_df, on="name", how="inner")
merged = merged[pd.to_numeric(merged["conflictivity"], errors="coerce") >= min_conf]

if merged.empty:
    st.info("No APs matched the current filters.")
    st.stop()

# Optional group filter (by prefix code) derived from current data
available_groups = sorted({g for g in merged["name"].apply(extract_group).dropna().unique().tolist()})
with st.sidebar:
    st.divider()
    st.header("Filters")
    selected_groups = st.multiselect(
        "Filter by building code", 
        options=available_groups, 
        default=available_groups,
        help="Select specific building codes to display"
    )

if selected_groups:
    merged = merged[merged["name"].apply(extract_group).isin(selected_groups)]

if merged.empty:
    st.info("No APs after applying group filter.")
    st.stop()

# Prepare data for map (AP level aggregation only)
map_df = merged.copy()

if map_df.empty:
    st.info("No data to display on the map.")
    st.stop()

# Center map
center_lat = float(map_df["lat"].mean())
center_lon = float(map_df["lon"].mean())

# Create optimized heatmap
fig = create_optimized_heatmap(
    df=map_df,
    center_lat=center_lat,
    center_lon=center_lon,
    radius=radius,
    zoom=15
)

st.plotly_chart(fig, use_container_width=True)


# Top conflictive listing
st.subheader("Top conflictive Access Points")
cols = [c for c in ["name", "group_code", "client_count", "max_radio_util", "conflictivity"] if c in map_df.columns]
tmp = map_df[cols].copy()
if "group_code" not in tmp.columns:
    tmp["group_code"] = tmp["name"].apply(extract_group)
top_df = tmp.sort_values("conflictivity", ascending=False).head(top_n)
top_df = top_df.rename(columns={"name": "Access Point", "group_code": "Building", "conflictivity": "Conflictivity Score"})
if "client_count" in top_df.columns:
    top_df = top_df.rename(columns={"client_count": "Clients"})
if "max_radio_util" in top_df.columns:
    top_df = top_df.rename(columns={"max_radio_util": "Max Radio Util %"})
    
# Format the score column
if "Conflictivity Score" in top_df.columns:
    top_df["Conflictivity Score"] = top_df["Conflictivity Score"].map(lambda x: f"{x:.3f}")
    
st.dataframe(top_df, use_container_width=True, hide_index=True)


# Footer
st.caption(
    "💡 **Conflictivity Formula:** 0.6 × normalized_clients + 0.4 × max_radio_utilization  |  "
    "🎨 **Color Scale:** 🟢 Green (Low) → 🟡 Yellow (Medium) → 🔴 Red (High)"
)
