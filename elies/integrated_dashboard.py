"""
Integrated Conflictivity Dashboard with AI Analysis and Interpolation

Purpose
- Combine heatmap visualization with clickable AINA AI analysis
- Show interpolated conflictivity surfaces for UAB campus
- Dual-mode visualization: heatmap points + interpolated tiles
- Time series navigation through Wi-Fi snapshots

Features
- Click any AP on the map to get AINA AI analysis
- Interpolated tile-based visualization for smooth coverage view
- Voronoi-weighted connectivity analysis
- Band mode selection (2.4GHz, 5GHz, worst, avg)
- Group filtering and time navigation

Run
  streamlit run elies/integrated_dashboard.py
"""

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
from shapely.geometry import Point, MultiPoint, LineString, Polygon
from shapely.ops import unary_union, linemerge, snap
from matplotlib.path import Path as MplPath

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

# --- Scoring utilities ----------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def airtime_score(util: float, band: str) -> float:
    """Map channel utilization % to [0,1] pain score."""
    u = clamp(util or 0.0, 0.0, 100.0)
    if band == "2g":
        if u <= 10:
            return 0.05 * (u / 10.0)
        if u <= 25:
            return 0.05 + 0.35 * ((u - 10) / 15.0)
        if u <= 50:
            return 0.40 + 0.35 * ((u - 25) / 25.0)
        return 0.75 + 0.25 * ((u - 50) / 50.0)
    else:
        if u <= 15:
            return 0.05 * (u / 15.0)
        if u <= 35:
            return 0.05 + 0.35 * ((u - 15) / 20.0)
        if u <= 65:
            return 0.40 + 0.35 * ((u - 35) / 30.0)
        return 0.75 + 0.25 * ((u - 65) / 35.0)

def client_pressure_score(n_clients: float, peers_p95: float) -> float:
    n = max(0.0, float(n_clients or 0.0))
    denom = max(1.0, float(peers_p95 or 1.0))
    x = math.log1p(n) / math.log1p(denom)
    return clamp(x, 0.0, 1.0)

def cpu_health_score(cpu_pct: float) -> float:
    c = clamp(cpu_pct or 0.0, 0.0, 100.0)
    if c <= 70:
        return 0.0
    if c <= 90:
        return 0.6 * ((c - 70) / 20.0)
    return 0.6 + 0.4 * ((c - 90) / 10.0)

def mem_health_score(mem_used_pct: float) -> float:
    m = clamp(mem_used_pct or 0.0, 0.0, 100.0)
    if m <= 80:
        return 0.0
    if m <= 95:
        return 0.6 * ((m - 80) / 15.0)
    return 0.6 + 0.4 * ((m - 95) / 5.0)

def read_ap_snapshot(path: Path, band_mode: str = "worst") -> pd.DataFrame:
    """Read AP snapshot and calculate advanced conflictivity."""
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

    w_2g = 0.6
    w_5g = 0.4

    # Airtime scores per band
    df["air_s_2g"] = df["util_2g"].apply(lambda u: airtime_score(u, "2g") if not np.isnan(u) else np.nan)
    df["air_s_5g"] = df["util_5g"].apply(lambda u: airtime_score(u, "5g") if not np.isnan(u) else np.nan)

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

    def relief(a_score: float, clients: float) -> float:
        if np.isnan(a_score):
            return np.nan
        if (clients or 0) > 0:
            return a_score
        return a_score * 0.8

    df["airtime_score_adj"] = [
        relief(a, c) for a, c in zip(df["airtime_score"], df["client_count"])
    ]

    # Final conflictivity weights
    W_AIR = 0.75
    W_CL  = 0.15
    W_CPU = 0.05
    W_MEM = 0.05

    df["airtime_score_filled"] = df["airtime_score_adj"].fillna(0.4)

    df["conflictivity"] = (
        df["airtime_score_filled"] * W_AIR
        + df["client_score"].fillna(0) * W_CL
        + df["cpu_score"].fillna(0) * W_CPU
        + df["mem_score"].fillna(0) * W_MEM
    ).clip(0, 1)

    df["max_radio_util"] = df["agg_util"].fillna(0)
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

# -------- Interpolation utilities --------
def _compute_convex_hull_polygon(lons: np.ndarray, lats: np.ndarray):
    """Return a shapely Polygon of the convex hull, or None if degenerate."""
    pts = [Point(xy) for xy in zip(lons, lats)]
    if len(pts) < 3:
        return None
    mp = MultiPoint(pts)
    hull = mp.convex_hull
    if hull.is_empty or hull.geom_type != "Polygon":
        return None
    return hull

def _mask_points_in_polygon(lon_grid: np.ndarray, lat_grid: np.ndarray, polygon) -> np.ndarray:
    """Boolean mask for grid points inside polygon using matplotlib.path.Path."""
    x, y = polygon.exterior.coords.xy
    poly_path = MplPath(np.vstack([x, y]).T)
    XX, YY = np.meshgrid(lon_grid, lat_grid)
    pts = np.vstack([XX.ravel(), YY.ravel()]).T
    inside = poly_path.contains_points(pts)
    return inside.reshape(XX.shape)

def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2.0) ** 2
    return 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))

def _interp_kernel(dist_m: np.ndarray, R_m: float, mode: str = "decay"):
    """Kernel by distance in meters within radius R_m."""
    x = np.clip(dist_m / max(R_m, 1e-6), 0.0, 1.0)
    if mode == "grow":
        w = x
    else:
        w = 1.0 - x
    w[dist_m >= R_m] = 0.0
    return w

def _uab_tiled_choropleth_layer(df_uab: pd.DataFrame, *, tile_meters: float = 7.0,
                                radius_m: float = 25.0, mode: str = "decay",
                                value_mode: str = "conflictivity",
                                max_tiles: int = 40000, colorscale=None):
    """Create a Choroplethmapbox layer of rectangular tiles."""
    if colorscale is None:
        colorscale = [[0.0, 'rgb(0, 255, 0)'], [0.5, 'rgb(255, 255, 0)'], [1.0, 'rgb(255, 0, 0)']]

    lons = df_uab["lon"].to_numpy(dtype=float)
    lats = df_uab["lat"].to_numpy(dtype=float)
    hull_poly = _compute_convex_hull_polygon(lons, lats)
    if hull_poly is None:
        return None, tile_meters, None

    lat0 = float(np.mean(lats)) if len(lats) else 41.5
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    dlat = tile_meters / meters_per_deg_lat
    dlon = tile_meters / meters_per_deg_lon if meters_per_deg_lon > 0 else tile_meters / 100_000.0

    minx, miny, maxx, maxy = hull_poly.bounds
    lon_centers = np.arange(minx + dlon/2, maxx, dlon)
    lat_centers = np.arange(miny + dlat/2, maxy, dlat)
    XX, YY = np.meshgrid(lon_centers, lat_centers)
    centers = np.column_stack([XX.ravel(), YY.ravel()])

    x_h, y_h = hull_poly.exterior.coords.xy
    poly_path = MplPath(np.vstack([x_h, y_h]).T)
    inside = poly_path.contains_points(centers)
    centers_in = centers[inside]

    effective_tile_meters = tile_meters
    n_tiles = centers_in.shape[0]
    if n_tiles > max_tiles and n_tiles > 0:
        factor = float(np.ceil(n_tiles / max_tiles))
        effective_tile_meters = tile_meters * factor
        dlat *= factor
        dlon *= factor
        lon_centers = np.arange(minx + dlon/2, maxx, dlon)
        lat_centers = np.arange(miny + dlat/2, maxy, dlat)
        XX, YY = np.meshgrid(lon_centers, lat_centers)
        centers = np.column_stack([XX.ravel(), YY.ravel()])
        inside = poly_path.contains_points(centers)
        centers_in = centers[inside]

    if centers_in.size == 0:
        return None, effective_tile_meters, hull_poly

    dists = _haversine_m(centers_in[:, 1][:, None], centers_in[:, 0][:, None], lats[None, :], lons[None, :])

    if value_mode == "connectivity":
        d_min = dists.min(axis=1)
        z_pred = np.clip(d_min / max(radius_m, 1e-6), 0.0, 1.0)
    else:
        d_min = dists.min(axis=1)
        boundary_conf = np.where(d_min >= radius_m, 1.0, d_min / max(radius_m, 1e-6))
        cvals = df_uab["conflictivity"].to_numpy(dtype=float)
        W = _interp_kernel(dists, radius_m, mode=mode)
        denom = W.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            num = (W * cvals[None, :]).sum(axis=1)
            weighted_conf = np.where(denom > 0, num / denom, np.nan)
        z_pred = np.maximum(weighted_conf, boundary_conf)
        z_pred = np.clip(z_pred, 0.0, 1.0)

    features = []
    ids = []
    for i, (lon_c, lat_c) in enumerate(centers_in):
        lon0, lon1 = lon_c - dlon/2, lon_c + dlon/2
        lat0, lat1 = lat_c - dlat/2, lat_c + dlat/2
        poly_coords = [[
            [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]
        ]]
        features.append({
            "type": "Feature",
            "id": str(i),
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": poly_coords}
        })
        ids.append(str(i))

    geojson = {"type": "FeatureCollection", "features": features}

    colorbar_title = "Connectivity" if value_mode == "connectivity" else "Conflictivity"
    ch = go.Choroplethmapbox(
        geojson=geojson,
        locations=ids,
        z=z_pred,
        colorscale=colorscale,
        zmin=0, zmax=1,
        marker_opacity=0.7,
        marker_line_width=0,
        showscale=True,
        colorbar=dict(title=colorbar_title, thickness=15, len=0.7, x=1.02),
        name="Interpolated surface",
        hovertemplate='Conflictivity: %{z:.2f}<extra></extra>',
    )

    return ch, effective_tile_meters, hull_poly

def create_optimized_heatmap(
    df: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    min_conflictivity: float = 0.0,
    radius: int = 15,
    zoom: int = 15,
) -> go.Figure:
    """Create heatmap with clickable AP points."""
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
    
    location_groups = location_groups[location_groups["max_conflictivity"] >= min_conflictivity]
    location_groups = location_groups.sort_values("max_conflictivity", ascending=True)

    hover_texts = []
    ap_names_list = []
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
            ap_names_list.append([n])
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
            ap_names_list.append(names_at_location)
        hover_texts.append(t)

    fig = go.Figure()
    
    # Add scatter points for APs
    fig.add_trace(go.Scattermapbox(
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
                title="AP Conflictivity",
                thickness=15,
                len=0.7,
                tickmode="linear",
                tick0=0,
                dtick=0.2,
                tickformat=".1f",
                x=0.98,
            ),
        ),
        text=hover_texts,
        hovertemplate="%{text}<extra></extra>",
        showlegend=False,
        customdata=ap_names_list,
        name="Access Points",
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
st.set_page_config(page_title="UAB Wi‑Fi Integrated Dashboard", page_icon="📶", layout="wide")
st.title("UAB Wi‑Fi Integrated Dashboard")
st.caption("Time series visualization • Interpolated surfaces + Clickable heatmap with AI analysis")

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
    
    viz_mode = st.radio(
        "Visualization Mode",
        options=["Heatmap (clickable)", "Interpolated Tiles", "Both"],
        index=2,
        help="Choose between AP heatmap points, interpolated surface, or both"
    )
    
    band_mode = st.radio(
        "Band Mode",
        options=["worst", "avg", "2.4GHz", "5GHz"],
        index=0,
        help="worst: max(max_2.4, max_5) • avg: weighted average of band maxima",
        horizontal=True,
    )
    
    if viz_mode in ["Interpolated Tiles", "Both"]:
        radius_m = st.slider("Interpolation radius (m)", 5, 60, 25, step=5,
                           help="Distance for interpolation kernel")
        tile_meters = st.slider("Tile size (m)", 3, 15, 7, step=1,
                              help="Size of each interpolation tile")
    else:
        radius_m = 25
        tile_meters = 7
    
    radius = 5  # Fixed heatmap point radius
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

# Initialize session state for chart key
if "chart_refresh_key" not in st.session_state:
    st.session_state.chart_refresh_key = 0

def on_dialog_close():
    st.session_state.chart_refresh_key += 1

# Dialog function for AINA AI analysis
@st.dialog("🤖 Anàlisi AINA AI", width="large")
def show_aina_analysis(ap_name: str, ap_row: pd.Series):
    """Show AINA AI analysis in a modal dialog."""
    st.subheader(f"Access Point: {ap_name}")
    
    util_2g = ap_row.get("util_2g", np.nan)
    util_5g = ap_row.get("util_5g", np.nan)
    client_count = ap_row.get("client_count", 0)
    cpu_util = ap_row.get("cpu_utilization", np.nan)
    mem_free = ap_row.get("mem_free", np.nan)
    mem_total = ap_row.get("mem_total", np.nan)
    mem_used_pct = ap_row.get("mem_used_pct", np.nan)
    conflictivity = ap_row.get("conflictivity", np.nan)
    
    def format_value(val, format_str="{:.1f}", default="no disponible"):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return format_str.format(val)
    
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

# Create figure based on visualization mode
fig = go.Figure()

# Add interpolated tiles if requested
if viz_mode in ["Interpolated Tiles", "Both"]:
    tmp = map_df.copy()
    if "group_code" not in tmp.columns:
        tmp["group_code"] = tmp["name"].apply(extract_group)
    
    # UAB interpolation (non-SAB)
    uab_df = tmp[tmp["group_code"] != "SAB"].copy()
    if not uab_df.empty:
        ch, eff_tile, hull = _uab_tiled_choropleth_layer(
            uab_df, tile_meters=tile_meters, radius_m=radius_m, mode="decay",
            value_mode="conflictivity", max_tiles=40000
        )
        if ch is not None:
            fig.add_trace(ch)

# Add heatmap points if requested
if viz_mode in ["Heatmap (clickable)", "Both"]:
    heatmap_fig = create_optimized_heatmap(
        df=map_df,
        center_lat=center_lat,
        center_lon=center_lon,
        min_conflictivity=min_conf,
        radius=radius,
        zoom=15,
    )
    # Add heatmap trace to main figure
    for trace in heatmap_fig.data:
        fig.add_trace(trace)

# Update layout
fig.update_layout(
    mapbox=dict(
        style="open-street-map",
        center=dict(lat=center_lat, lon=center_lon),
        zoom=15,
    ),
    margin=dict(l=10, r=10, t=30, b=10),
    height=700,
    legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02)
)

# Handle map selection for AI analysis
fig.update_layout(clickmode='event+select')
selected_points = st.plotly_chart(
    fig, 
    use_container_width=True, 
    on_select="rerun",
    key=f"ap_map_{st.session_state.chart_refresh_key}"
)

# Process selection and open dialog
if selected_points and "selection" in selected_points:
    selection = selected_points["selection"]
    if "points" in selection and len(selection["points"]) > 0:
        point = selection["points"][0]
        ap_name = None
        
        if "customdata" in point and point["customdata"]:
            ap_names = point["customdata"]
            if isinstance(ap_names, list) and len(ap_names) > 0:
                ap_name = ap_names[0] if isinstance(ap_names[0], str) else str(ap_names[0])
        
        if not ap_name and "text" in point:
            text = point["text"]
            name_match = re.search(r"<b>([^<]+)</b>", text)
            if name_match:
                ap_name = name_match.group(1)
        
        if ap_name:
            ap_data = merged[merged["name"] == ap_name]
            if not ap_data.empty:
                show_aina_analysis(ap_name, ap_data.iloc[0])

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

