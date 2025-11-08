"""
Static Conflictivity Dashboard (UI-first)

Purpose
- Show a heatmap of Wi-Fi conflictivity by place on the UAB campus using the latest snapshot in realData/ap/.
- No live API yet; focuses on polished UI and local snapshot selection.

Data sources
- AP snapshots: realData/ap/AP-info-v2-*.json (fields include name, client_count, radios[].utilization, group_name, etc.)
- AP geolocations: realData/geoloc/aps_geolocalizados_wgs84.geojson (properties.USER_NOM_A = AP name, geometry.coordinates = [lon, lat])

Conflictivity metric (static assumption)
- Combine client load and radio utilization per AP into a [0,1] score:
  score = 0.6 * norm01(client_count) + 0.4 * max_radio_utilization/100
- When min=max for client_count, use 0.5 for that component.
- For "place" aggregation, we use the AP name prefix (e.g., AP-VET71 -> VET) and compute the mean score per group.

UI features
- Snapshot selector (auto-selects latest on load)
- Aggregation: APs vs Buildings (group prefix)
- Map type: Heatmap vs Points
- Filters: group prefix, minimum conflictivity threshold, heatmap radius
- Top N most conflictive table

Run
  streamlit run dashboard/conflictivity_dashboard.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
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


def find_snapshot_files(ap_dir: Path) -> List[Path]:
    return sorted(ap_dir.glob("AP-info-v2-*.json"))


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


# -------- UI --------
st.set_page_config(page_title="UAB Wi‑Fi Conflictivity", page_icon="📶", layout="wide")
st.title("UAB Wi‑Fi Conflictivity Dashboard")
st.caption("Static prototype • Heatmap of conflictivity by place (AP or Building)")

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

# Sidebar: snapshot selector and controls
with st.sidebar:
    st.header("Controls")
    # Default to latest lexicographically (matches timestamp in filename pattern)
    default_idx = len(snapshots) - 1
    snap_labels = [p.name for p in snapshots]
    snap_choice = st.selectbox("Snapshot", options=range(len(snapshots)), format_func=lambda i: snap_labels[i], index=default_idx)
    selected_path = snapshots[snap_choice]

    agg_level = st.radio("Aggregation", options=["APs", "Buildings"], index=1, help="Show individual APs or building-level centroids")
    map_type = st.radio("Map type", options=["Heatmap", "Points"], index=0)
    min_conf = st.slider("Minimum conflictivity", 0.0, 1.0, 0.0, 0.01)
    radius = st.slider("Heatmap radius (px)", 5, 60, 25) if map_type == "Heatmap" else None
    top_n = st.slider("Top N listing", 5, 50, 15, step=5)

# Load data
geo_df = read_geoloc_points(GEOJSON_PATH)
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
    selected_groups = st.multiselect("Filter by building code", options=available_groups, default=available_groups)

if selected_groups:
    merged = merged[merged["name"].apply(extract_group).isin(selected_groups)]

if merged.empty:
    st.info("No APs after applying group filter.")
    st.stop()


# Prepare data for map
if agg_level == "Buildings":
    map_df = aggregate_by_group(merged, geo_df)
    lat_col, lon_col, z_col, hover_name = "lat", "lon", "conflictivity", "group_code"
else:
    map_df = merged.copy()
    lat_col, lon_col, z_col, hover_name = "lat", "lon", "conflictivity", "name"

if map_df.empty:
    st.info("No data to display on the map.")
    st.stop()


# Center map
center_lat = float(map_df[lat_col].mean())
center_lon = float(map_df[lon_col].mean())

# Color scale: red (high conflict) -> yellow -> green (low). We'll invert RdYlGn
colorscale = "RdYlGn_r"
zoom = 14 if agg_level == "Buildings" else 15

if map_type == "Heatmap":
    fig = px.density_mapbox(
        map_df,
        lat=lat_col,
        lon=lon_col,
        z=z_col,
        radius=radius or 25,
        center=dict(lat=center_lat, lon=center_lon),
        zoom=zoom,
        height=700,
        color_continuous_scale=colorscale,
    )
else:
    size_col = "client_count" if "client_count" in map_df.columns else z_col
    fig = px.scatter_mapbox(
        map_df,
        lat=lat_col,
        lon=lon_col,
        color=z_col,
        size=size_col,
        hover_name=hover_name,
        hover_data={
            "conflictivity": ":.2f",
            "client_count": True if "client_count" in map_df.columns else False,
            "max_radio_util": True if "max_radio_util" in map_df.columns else False,
        },
        color_continuous_scale=colorscale,
        height=700,
        zoom=zoom,
    )

fig.update_layout(mapbox_style="open-street-map", margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)


# Top conflictive listing
st.subheader("Top conflictive places")
if agg_level == "Buildings":
    top_df = map_df[["group_code", "conflictivity"]].sort_values("conflictivity", ascending=False).head(top_n)
    top_df = top_df.rename(columns={"group_code": "place", "conflictivity": "score"})
else:
    cols = [c for c in ["name", "group_code", "client_count", "max_radio_util", "conflictivity"] if c in map_df.columns]
    tmp = map_df[cols].copy()
    if "group_code" not in tmp.columns:
        tmp["group_code"] = tmp["name"].apply(extract_group)
    top_df = tmp.sort_values("conflictivity", ascending=False).head(top_n)
    top_df = top_df.rename(columns={"name": "place", "conflictivity": "score"})
top_df["score"] = top_df["score"].map(lambda x: f"{x:.2f}")
st.dataframe(top_df, use_container_width=True)


# Footer
st.caption(
    "Conflictivity = 0.6 · normalized clients + 0.4 · max radio utilization. "
    "When clients are constant across APs, that component becomes 0.5."
)
