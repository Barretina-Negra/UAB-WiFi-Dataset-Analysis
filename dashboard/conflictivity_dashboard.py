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

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -------- Paths --------
REPO_ROOT = Path(__file__).resolve().parents[1]
AP_DIR = REPO_ROOT / "realData" / "ap"
GEOJSON_PATH = REPO_ROOT / "realData" / "geoloc" / "aps_geolocalizados_wgs84.geojson"


# -------- Helpers --------
def norm01(series: pd.Series, invert: bool = False) -> pd.Series:
    """Normalize series to [0,1] range where 1 = worst.
    Matches the exact algorithm from ap_problem_analysis.ipynb
    """
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


def read_ap_snapshot(path: Path, band_filter: str = "average") -> pd.DataFrame:
    """Load one AP snapshot JSON into a DataFrame with selected fields.
    Columns: name, client_count, group_name, site, max_radio_util, cpu_utilization, mem_free, mem_total, conflictivity
    
    Args:
        path: Path to the JSON snapshot file
        band_filter: Frequency band to analyze - "2.4GHz", "5GHz", or "average" (default)
                     - "2.4GHz": Only consider 2.4 GHz radios (band=0), use max if multiple
                     - "5GHz": Only consider 5 GHz radios (band=1), use max if multiple
                     - "average": Average of max(2.4GHz radios) and max(5GHz radios)
    
    Note: max_radio_util column contains:
          - For single band modes: max utilization of radios on that band
          - For average mode: average(max(5GHz), max(2.4GHz))
    
    Conflictivity formula based on ap_problem_analysis.ipynb:
    - Radio utilization: 0.85 (primary indicator of channel congestion)
    - Client load: 0.10 (normalized client count)
    - CPU utilization: 0.02 (processing overhead)
    - Memory usage: 0.03 (available resources)
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for ap in data:
        name = ap.get("name")
        client_count = ap.get("client_count", 0)
        group_name = ap.get("group_name")
        site = ap.get("site")
        cpu_utilization = ap.get("cpu_utilization")
        mem_free = ap.get("mem_free")
        mem_total = ap.get("mem_total")
        
        # Extract radio metrics with band filtering
        radios = ap.get("radios") or []
        
        if band_filter == "average":
            # For average mode: compute average(max(5GHz), max(2.4GHz))
            band_24ghz = []
            band_5ghz = []
            
            for r in radios:
                u = r.get("utilization")
                band = r.get("band")
                
                if u is None:
                    continue
                
                if band == 0:  # 2.4 GHz
                    band_24ghz.append(u)
                elif band == 1:  # 5 GHz
                    band_5ghz.append(u)
            
            # Get max from each band
            max_24 = max(band_24ghz) if band_24ghz else None
            max_5 = max(band_5ghz) if band_5ghz else None
            
            # Calculate average of the max values
            if max_24 is not None and max_5 is not None:
                radio_util = (max_24 + max_5) / 2
            elif max_24 is not None:
                radio_util = max_24
            elif max_5 is not None:
                radio_util = max_5
            else:
                radio_util = None
        else:
            # For single band filters
            radio_utils = []
            
            for r in radios:
                u = r.get("utilization")
                band = r.get("band")
                
                if u is None:
                    continue
                
                # Apply band filter
                if band_filter == "2.4GHz" and band != 0:
                    continue
                elif band_filter == "5GHz" and band != 1:
                    continue
                
                radio_utils.append(u)
            
            # Use max for single band (in case AP has multiple radios on same band)
            radio_util = max(radio_utils) if radio_utils else None
        
        rows.append({
            "name": name,
            "client_count": client_count,
            "group_name": group_name,
            "site": site,
            "max_radio_util": radio_util,
            "cpu_utilization": cpu_utilization,
            "mem_free": mem_free,
            "mem_total": mem_total,
        })
    
    df = pd.DataFrame(rows)
    
    # Sanitize numeric fields
    # Note: max_radio_util None values (APs with no matching band radios) become 0
    # These APs will have low conflictivity scores, which is appropriate since they
    # don't have active radios on the selected band
    numeric_cols = ["client_count", "max_radio_util", "cpu_utilization", "mem_free", "mem_total"]
    for col in numeric_cols:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    
    # Calculate memory used percentage
    if "mem_free" in df.columns and "mem_total" in df.columns:
        df["mem_used_pct"] = ((1 - (df["mem_free"] / df["mem_total"]).clip(lower=0, upper=1)) * 100).fillna(0)
    else:
        df["mem_used_pct"] = 0
    
    # Build conflictivity score using algorithm adapted from ap_problem_analysis.ipynb
    # with proper radio utilization interpretation
    
    # Radio Utilization Score (non-linear based on thresholds):
    # < 20%: Good (score 0.0-0.3)
    # 20-50%: Moderate (score 0.3-0.7)
    # > 50%: High/Critical (score 0.7-1.0)
    def radio_util_score(util):
        """Convert radio utilization % to a problem score [0,1]"""
        if util < 20:
            # Light utilization: map 0-20% to 0.0-0.3
            return (util / 20) * 0.3
        elif util < 50:
            # Moderate utilization: map 20-50% to 0.3-0.7
            return 0.3 + ((util - 20) / 30) * 0.4
        else:
            # High utilization: map 50-100% to 0.7-1.0
            return 0.7 + ((util - 50) / 50) * 0.3
    
    # Apply radio utilization scoring
    df["util_score"] = df["max_radio_util"].apply(radio_util_score)
    
    # Normalize other metrics using norm01 (linear normalization)
    df["load_norm"] = norm01(df["client_count"]) if len(df) > 0 else pd.Series(0, index=df.index)
    df["cpu_norm"] = norm01(df["cpu_utilization"]) if len(df) > 0 else pd.Series(0, index=df.index)
    df["mem_norm"] = norm01(df["mem_used_pct"]) if len(df) > 0 else pd.Series(0, index=df.index)
    
    # Weights adapted for AP-only data:
    # - Radio utilization is our primary indicator (85%)
    # - Client load adds context about demand (10%)
    # - System resources (CPU + Mem) = 5%
    
    w_util = 0.85      # Radio utilization (primary metric)
    w_load = 0.10      # Client load
    w_cpu = 0.02       # CPU utilization
    w_mem = 0.03       # Memory usage
    
    df["conflictivity"] = (
        df["util_score"] * w_util +
        df["load_norm"] * w_load +
        df["cpu_norm"] * w_cpu +
        df["mem_norm"] * w_mem
    ).clip(lower=0, upper=1)
    
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
                              min_conflictivity: float = 0.0, radius: int = 15, zoom: int = 15) -> go.Figure:
    """Create an optimized heatmap with proper color representation per AP.
    
    Key features:
    - Each AP shows its TRUE color based on its conflictivity value (no density averaging)
    - APs below min_conflictivity have reduced opacity
    - High conflictivity values render on top
    - Color scale ALWAYS fixed to 0-1 range
    """
    # Sort by conflictivity DESCENDING so low values are plotted first,
    # then high values (red) are plotted last and appear on top
    df_sorted = df.sort_values('conflictivity', ascending=False).copy()
    
    # Calculate opacity based on threshold
    # Full opacity for APs >= threshold, reduced for others
    opacity_values = df_sorted['conflictivity'].apply(
        lambda x: 0.85 if x >= min_conflictivity else 0.15
    ).tolist()
    
    # Create scatter plot with individual colored markers
    fig = go.Figure(go.Scattermapbox(
        lat=df_sorted['lat'],
        lon=df_sorted['lon'],
        mode='markers',
        marker=dict(
            size=radius * 2,  # Convert radius to marker size
            color=df_sorted['conflictivity'],
            colorscale=[
                [0.0, 'rgb(0, 255, 0)'],        # Green (low conflictivity)
                [0.5, 'rgb(255, 165, 0)'],      # Orange (medium)
                [1.0, 'rgb(255, 0, 0)']         # Red (high conflictivity)
            ],
            cmin=0,
            cmax=1,
            opacity=opacity_values,
            showscale=True,
            colorbar=dict(
                title="Conflictivity",
                thickness=15,
                len=0.7,
                tickmode='linear',
                tick0=0,
                dtick=0.2,
                tickformat='.1f',
            ),
        ),
        text=df_sorted['name'],
        hovertemplate='<b>%{text}</b><br>Conflictivity: %{marker.color:.3f}<extra></extra>',
        showlegend=False,
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
    
    # Frequency band selector
    band_filter = st.radio(
        "Frequency Band",
        options=["average", "2.4GHz", "5GHz"],
        index=0,
        help="Select which frequency band to analyze:\n"
             "• Average: avg(max(2.4GHz), max(5GHz))\n"
             "• 2.4GHz: Only 2.4 GHz radios (band 0)\n"
             "• 5GHz: Only 5 GHz radios (band 1)"
    )
    
    # Heatmap radius optimization (optimal value: 15)
    radius = st.slider(
        "Heatmap radius (px)", 
        min_value=5, 
        max_value=40, 
        value=15,
        help="Optimal value: 15px for clear AP differentiation"
    )
    
    min_conf = st.slider("Minimum conflictivity", 0.0, 1.0, 0.0, 0.01)
    top_n = st.slider(
        "Top N listing", 
        5, 50, 15, step=5,
        help="Number of most conflictive APs to show in the table below the map"
    )

# Load data for selected timestamp with band filter
ap_df = read_ap_snapshot(selected_path, band_filter=band_filter)

# Merge AP + geoloc - DON'T filter by min_conf yet (we need all data for consistent heatmap)
merged = ap_df.merge(geo_df, on="name", how="inner")

if merged.empty:
    st.info("No APs have geolocation data.")
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

# Apply group filter if selected
if selected_groups:
    merged = merged[merged["name"].apply(extract_group).isin(selected_groups)]

if merged.empty:
    st.info("No APs after applying group filter.")
    st.stop()

# Prepare data for map (AP level aggregation only)
# Keep ALL APs for consistent heatmap calculation
map_df = merged.copy()

if map_df.empty:
    st.info("No data to display on the map.")
    st.stop()

# Center map
center_lat = float(map_df["lat"].mean())
center_lon = float(map_df["lon"].mean())

# Create optimized heatmap with ALL data, using min_conf for opacity control
fig = create_optimized_heatmap(
    df=map_df,
    center_lat=center_lat,
    center_lon=center_lon,
    min_conflictivity=min_conf,
    radius=radius,
    zoom=15
)

st.plotly_chart(fig, use_container_width=True)


# Top conflictive listing - filter here for the table
st.subheader("Top conflictive Access Points")
# Filter by minimum conflictivity for the table display
filtered_for_table = map_df[map_df["conflictivity"] >= min_conf].copy()

if filtered_for_table.empty:
    st.info(f"No APs with conflictivity >= {min_conf:.2f}")
else:
    cols = [c for c in ["name", "group_code", "client_count", "max_radio_util", "conflictivity"] if c in filtered_for_table.columns]
    tmp = filtered_for_table[cols].copy()
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
band_info = {
    "average": "All bands (average)",
    "2.4GHz": "2.4 GHz only",
    "5GHz": "5 GHz only"
}
st.caption(
    f"📻 **Band:** {band_info[band_filter]}  |  "
    "💡 **Conflictivity:** 85% radio_util (threshold-based) + 10% client_load + 2% cpu + 3% memory  |  "
    "📡 **Radio Util:** <20% good, 20-50% moderate, >50% high  |  "
    "🎨 **Colors:** 🟢 Green (Low) → 🟠 Orange (Medium) → 🔴 Red (High)"
)
