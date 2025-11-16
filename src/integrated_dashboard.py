"""
Integrated Conflictivity Dashboard - AI Heatmap + Voronoi + Simulator

Purpose
- Unified dashboard combining three visualization modes:
  1. AI Heatmap: Clickable AP points with AINA AI analysis
  2. Voronoi: Interpolated surfaces with weighted Voronoi connectivity analysis
  3. Simulator: AP placement optimization with multi-scenario testing
- Time series navigation through Wi-Fi snapshots

Features
- Radio button to switch between AI Heatmap, Voronoi, and Simulator modes
- AI Heatmap: Click any AP to get AINA AI analysis of conflictivity
- Voronoi: Advanced interpolation with connectivity regions and hotspot detection
- Simulator: Interactive AP placement with stress profile analysis and Voronoi candidates
- Band mode selection, group filtering, time navigation

Run
  streamlit run elies/integrated_dashboard.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict, Counter

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
AP_DIR = REPO_ROOT / "reducedData" / "ap"
GEOJSON_PATH = REPO_ROOT / "reducedData" / "geoloc" / "aps_geolocalizados_wgs84.geojson"

# Add simulator to path
sys.path.insert(0, str(REPO_ROOT))

# Import simulator components (try-except for graceful degradation)
try:
    from experiments.polcorresa.simulator.config import SimulationConfig, StressLevel
    from experiments.polcorresa.simulator.stress_profiler import StressProfiler
    from experiments.polcorresa.simulator.scoring import CompositeScorer, NeighborhoodOptimizationMode
    from experiments.polcorresa.simulator.spatial import haversine_m as sim_haversine_m
    SIMULATOR_AVAILABLE = True
except ImportError:
    SIMULATOR_AVAILABLE = False

# Check for scipy Voronoi
try:
    from scipy.spatial import Voronoi
    _HAS_SCIPY_VORONOI = True
except Exception:
    _HAS_SCIPY_VORONOI = False

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


def resolve_stress_profiles(
    target: Optional["StressLevel"],
    stats: Dict["StressLevel", Dict[str, float]],
) -> Tuple[List["StressLevel"], Optional["StressLevel"], Optional[str]]:
    """Pick stress profiles to simulate, falling back gracefully when data is missing."""
    priority = [StressLevel.CRITICAL, StressLevel.HIGH, StressLevel.MEDIUM, StressLevel.LOW]
    counts = {lvl: stats.get(lvl, {}).get('count', 0) for lvl in priority}
    available = [lvl for lvl in priority if counts.get(lvl, 0) > 0]

    if target is None:
        if not available:
            return [], None, "No snapshots available in any stress profile."
        effective_target = None if len(available) > 1 else available[0]
        return available, effective_target, None

    if counts.get(target, 0) > 0:
        return [target], target, None

    if available:
        fallback = available[0]
        message = f"No snapshots found for stress profile {target.value}. Falling back to {fallback.value}."
        return [fallback], fallback, message

    return [], None, "No snapshots available to run the simulator."

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

# ======== AI HEATMAP MODE FUNCTIONS (from aina_dashboard.py) ========

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
            customdata=ap_names_list,
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

# ======== VORONOI MODE FUNCTIONS (from conflictivity_dashboard_interpolation.py) ========

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
        marker_opacity=0.9,
        marker_line_width=0,
        showscale=True,
        colorbar=dict(title=colorbar_title, thickness=15, len=0.7),
        name="UAB tiles",
    )

    return ch, effective_tile_meters, hull_poly

def _inverted_weighted_voronoi_edges(df: pd.DataFrame, *, weight_col: str = "conflictivity",
                                     radius_m: float = 25.0, clip_polygon=None,
                                     tolerance_m: float = 8.0):
    """Compute weighted Voronoi graph edges."""
    if df.empty or weight_col not in df.columns:
        return []
    pts_lon = df["lon"].to_numpy(dtype=float)
    pts_lat = df["lat"].to_numpy(dtype=float)
    base = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    mn, mx = float(base.min()), float(base.max())
    norm = (base - mn) / (mx - mn + 1e-12)
    inv_w = 1.0 - norm

    if len(pts_lon) < 3:
        return []
    try:
        from scipy.spatial import Voronoi
    except Exception:
        return []
    pts = np.column_stack([pts_lon, pts_lat])
    try:
        uniq, counts = np.unique(pts, axis=0, return_counts=True)
        if np.any(counts > 1):
            rng = np.random.RandomState(0)
            pts = pts + rng.randn(*pts.shape) * 1e-8
    except Exception:
        pass
    vor = Voronoi(pts)

    edges = []
    def _adiff(lon, lat, i1, i2):
        d1 = _haversine_m(lat, lon, pts_lat[i1], pts_lon[i1])
        d2 = _haversine_m(lat, lon, pts_lat[i2], pts_lon[i2])
        return abs((d1 - inv_w[i1] * max(radius_m, 1e-6)) - (d2 - inv_w[i2] * max(radius_m, 1e-6)))

    def _keep_and_clip_segment(a, b, i1, i2):
        seg = LineString([a, b])
        geom = seg if clip_polygon is None else seg.intersection(clip_polygon)
        if geom.is_empty:
            return []
        parts = []
        if geom.geom_type == 'LineString':
            parts = [geom]
        elif geom.geom_type == 'MultiLineString':
            parts = list(geom.geoms)
        kept = []
        for ls in parts:
            coords = list(ls.coords)
            if len(coords) < 2:
                continue
            K = 7
            min_diff = float('inf')
            for t in np.linspace(0.1, 0.9, K):
                x = coords[0][0] * (1 - t) + coords[-1][0] * t
                y = coords[0][1] * (1 - t) + coords[-1][1] * t
                min_diff = min(min_diff, _adiff(x, y, i1, i2))
            if min_diff < tolerance_m:
                (ax, ay), (bx, by) = coords[0], coords[-1]
                kept.append((ax, ay, bx, by))
        return kept

    for (p1, p2), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 not in rv:
            vcoords = vor.vertices[rv]
            lon1, lat1 = float(vcoords[0][0]), float(vcoords[0][1])
            lon2, lat2 = float(vcoords[1][0]), float(vcoords[1][1])
            kept = _keep_and_clip_segment((lon1, lat1), (lon2, lat2), p1, p2)
            edges.extend(kept)
        else:
            vs = [v for v in rv if v != -1]
            if not vs:
                continue
            v0 = vor.vertices[vs[0]]
            lon0, lat0 = float(v0[0]), float(v0[1])
            p1_xy = pts[[p1]][0]
            p2_xy = pts[[p2]][0]
            t = p2_xy - p1_xy
            dir_vec = np.array([t[1], -t[0]], dtype=float)
            nrm = np.linalg.norm(dir_vec)
            if nrm == 0:
                continue
            dir_vec /= nrm
            center = pts.mean(axis=0)
            mid = (p1_xy + p2_xy) / 2.0
            if np.dot(mid - center, dir_vec) < 0:
                dir_vec *= -1.0
            if clip_polygon is None:
                continue
            minx, miny, maxx, maxy = clip_polygon.bounds
            L = max(maxx - minx, maxy - miny) * 5.0 + 1e-6
            far = np.array([lon0, lat0]) + dir_vec * L
            ray = LineString([(lon0, lat0), (float(far[0]), float(far[1]))])
            inter = ray.intersection(clip_polygon)
            if inter.is_empty:
                continue
            def _add_segment_from_linestring(ls):
                coords = list(ls.coords)
                if len(coords) >= 2:
                    kept = _keep_and_clip_segment(coords[0], coords[-1], p1, p2)
                    edges.extend(kept)
            if inter.geom_type == 'LineString':
                _add_segment_from_linestring(inter)
            elif inter.geom_type == 'MultiLineString':
                for ls in inter.geoms:
                    _add_segment_from_linestring(ls)

    if clip_polygon is not None:
        clipped = []
        for (x1, y1, x2, y2) in edges:
            seg = LineString([(x1, y1), (x2, y2)])
            inter = seg.intersection(clip_polygon)
            if inter.is_empty:
                continue
            if inter.geom_type == 'LineString':
                coords = list(inter.coords)
                if len(coords) >= 2:
                    (ax, ay), (bx, by) = coords[0], coords[-1]
                    clipped.append((ax, ay, bx, by))
            elif inter.geom_type == 'MultiLineString':
                for seg2 in inter.geoms:
                    coords = list(seg2.coords)
                    if len(coords) >= 2:
                        (ax, ay), (bx, by) = coords[0], coords[-1]
                        clipped.append((ax, ay, bx, by))
        edges = clipped
    return edges

def _top_conflictive_voronoi_vertices(df: pd.DataFrame, *, radius_m: float, coverage_poly: Polygon, k: int = 3) -> list:
    """Return top-k Voronoi vertices with highest conflictivity score."""
    if df.empty:
        return []
    try:
        from scipy.spatial import Voronoi
    except Exception:
        return []
    pts_lon = df["lon"].to_numpy(float)
    pts_lat = df["lat"].to_numpy(float)
    cvals = df["conflictivity"].to_numpy(float)
    if len(pts_lon) < 3:
        return []
    pts = np.column_stack([pts_lon, pts_lat])
    try:
        uniq, counts = np.unique(pts, axis=0, return_counts=True)
        if np.any(counts > 1):
            rng = np.random.RandomState(7)
            pts = pts + rng.randn(*pts.shape) * 1e-8
    except Exception:
        pass
    vor = Voronoi(pts)
    cand = []
    for v in vor.vertices:
        lon, lat = float(v[0]), float(v[1])
        p = Point(lon, lat)
        if not coverage_poly.contains(p):
            continue
        dists = _haversine_m(lat, lon, pts_lat, pts_lon)
        dmin = dists.min()
        boundary_conf = 1.0 if dmin >= radius_m else (dmin / max(radius_m, 1e-6))
        W = _interp_kernel(dists, radius_m, mode="decay")
        denom = W.sum()
        if denom <= 0:
            continue
        weighted_conf = float((W * cvals).sum() / denom)
        combined = max(weighted_conf, boundary_conf)
        if boundary_conf >= 0.98 and weighted_conf < 0.5:
            continue
        cand.append((lon, lat, combined))
    cand.sort(key=lambda t: t[2], reverse=True)
    return cand[:max(0, int(k or 0))]

def _snap_and_connect_edges(segments: list, clip_polygon: Polygon,
                            *, lat0: float, snap_m: float = 2.0, join_m: float = 4.0):
    """Post-process Voronoi segments to enforce connectivity."""
    if not segments or clip_polygon is None:
        return None
    try:
        from shapely.geometry import MultiLineString, MultiPoint
    except Exception:
        return None

    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    tol_deg = max(snap_m / max(meters_per_deg_lat, 1e-6), snap_m / max(meters_per_deg_lon, 1e-6))
    join_deg = max(join_m / max(meters_per_deg_lat, 1e-6), join_m / max(meters_per_deg_lon, 1e-6))

    lines = [LineString([(x1, y1), (x2, y2)]) for (x1, y1, x2, y2) in segments]
    mls = MultiLineString(lines)

    endpoints = []
    for (x1, y1, x2, y2) in segments:
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))
    mp = MultiPoint(endpoints)
    target = unary_union([mp, clip_polygon.boundary])

    snapped = snap(mls, target, tol_deg)
    clipped = snapped.intersection(clip_polygon)
    if clipped.is_empty:
        return None
    if clipped.geom_type in ("Point", "MultiPoint"):
        return None

    def _collect_endpoints(geom):
        pts = []
        if geom.is_empty:
            return pts
        if geom.geom_type == 'LineString':
            coords = list(geom.coords)
            if len(coords) >= 2:
                pts.append(tuple(coords[0]))
                pts.append(tuple(coords[-1]))
        elif geom.geom_type == 'MultiLineString':
            for ls in geom.geoms:
                coords = list(ls.coords)
                if len(coords) >= 2:
                    pts.append(tuple(coords[0]))
                    pts.append(tuple(coords[-1]))
        return pts

    def _quantize(pt, q):
        return (round(pt[0] / q) * q, round(pt[1] / q) * q)

    pts = _collect_endpoints(clipped)
    if not pts:
        try:
            merged = linemerge(unary_union(clipped))
            return merged
        except Exception:
            return None

    deg = defaultdict(int)
    for p in pts:
        deg[_quantize(p, tol_deg)] += 1

    dangling = []
    for p in pts:
        qp = _quantize(p, tol_deg)
        if deg[qp] == 1:
            if Point(p).distance(clip_polygon.boundary) <= tol_deg * 1.5:
                continue
            dangling.append(p)

    added_connectors = []
    if dangling:
        for p in dangling:
            best = None
            best_d = 1e9
            for q in pts:
                if q == p:
                    continue
                d = ((p[0]-q[0])**2 + (p[1]-q[1])**2) ** 0.5
                if d < best_d:
                    best_d = d
                    best = q
            if best is not None and best_d <= join_deg:
                seg = LineString([p, best])
                inter = seg.intersection(clip_polygon)
                if not inter.is_empty:
                    if inter.geom_type == 'LineString':
                        added_connectors.append(inter)
                    elif inter.geom_type == 'MultiLineString':
                        for ls in inter.geoms:
                            added_connectors.append(ls)

    def _only_lines(g):
        lines = []
        if g is None or g.is_empty:
            return lines
        if g.geom_type == 'LineString':
            lines.append(g)
        elif g.geom_type == 'MultiLineString':
            for ls in g.geoms:
                if len(list(ls.coords)) >= 2:
                    lines.append(ls)
        return lines

    all_lines = _only_lines(clipped)
    if added_connectors:
        for ac in added_connectors:
            all_lines.extend(_only_lines(ac))
    if not all_lines:
        return None
    try:
        merged = linemerge(unary_union(all_lines))
        return merged
    except Exception:
        return None

def _coverage_regions_from_uab_tiles(uab_df: pd.DataFrame, tile_meters: float, radius_m: float,
                                     max_tiles: int = 40000) -> list:
    """Approximate coverage regions using tiling logic."""
    if uab_df.empty:
        return []
    lons = uab_df["lon"].to_numpy(float)
    lats = uab_df["lat"].to_numpy(float)
    hull = _compute_convex_hull_polygon(lons, lats)
    if hull is None:
        return []
    lat0 = float(np.mean(lats)) if len(lats) else 41.5
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    dlat = tile_meters / meters_per_deg_lat
    dlon = tile_meters / max(meters_per_deg_lon, 1e-6)
    minx, miny, maxx, maxy = hull.bounds
    lon_centers = np.arange(minx + dlon/2, maxx, dlon)
    lat_centers = np.arange(miny + dlat/2, maxy, dlat)
    XX, YY = np.meshgrid(lon_centers, lat_centers)
    centers = np.column_stack([XX.ravel(), YY.ravel()])
    poly_path = MplPath(np.vstack(hull.exterior.coords.xy).T)
    inside = poly_path.contains_points(centers)
    centers_in = centers[inside]
    if centers_in.size == 0:
        return []
    dists = _haversine_m(centers_in[:,1][:,None], centers_in[:,0][:,None], lats[None,:], lons[None,:])
    d_min = dists.min(axis=1)
    covered_mask = d_min < radius_m
    covered_centers = centers_in[covered_mask]
    if covered_centers.size == 0:
        return []
    tile_polys = []
    for (cx, cy) in covered_centers:
        lon0, lon1 = cx - dlon/2, cx + dlon/2
        lat0, lat1 = cy - dlat/2, cy + dlat/2
        tile_polys.append(Polygon([(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)]))
    merged = unary_union(tile_polys)
    polys = []
    if merged.geom_type == 'Polygon':
        polys = [merged]
    elif merged.geom_type == 'MultiPolygon':
        polys = list(merged.geoms)
    final_polys = [p for p in polys if p.area > 0]
    return final_polys

# ======== SIMULATOR MODE FUNCTIONS (from dashboard_voronoi_simulator.py) ========

def recalculate_conflictivity(df: pd.DataFrame) -> pd.DataFrame:
    """Full conflictivity recalculation after network changes."""
    df['air_s_2g'] = df['util_2g'].apply(lambda u: airtime_score(u, "2g") if not np.isnan(u) else np.nan)
    df['air_s_5g'] = df['util_5g'].apply(lambda u: airtime_score(u, "5g") if not np.isnan(u) else np.nan)
    
    df['airtime_score'] = np.nanmax(
        np.vstack([df['air_s_2g'].fillna(-1), df['air_s_5g'].fillna(-1)]),
        axis=0
    )
    df['airtime_score'] = df['airtime_score'].where(df['airtime_score'] >= 0, np.nan)
    
    p95 = float(np.nanpercentile(df['client_count'].fillna(0), 95)) if len(df) else 1.0
    df['client_score'] = df['client_count'].apply(lambda n: client_pressure_score(n, p95))
    
    if 'cpu_utilization' not in df.columns:
        df['cpu_utilization'] = 0.0
    if 'mem_used_pct' not in df.columns:
        df['mem_used_pct'] = 0.0
    
    df['cpu_score'] = df['cpu_utilization'].apply(cpu_health_score)
    df['mem_score'] = df['mem_used_pct'].apply(mem_health_score)
    
    df['airtime_score_adj'] = [
        (a * 0.8 if (c or 0) == 0 else a) if not np.isnan(a) else np.nan
        for a, c in zip(df['airtime_score'], df['client_count'])
    ]
    
    W_AIR, W_CL, W_CPU, W_MEM = 0.85, 0.10, 0.02, 0.03
    df['airtime_score_filled'] = df['airtime_score_adj'].fillna(0.4)
    df['conflictivity'] = (
        df['airtime_score_filled'] * W_AIR +
        df['client_score'].fillna(0) * W_CL +
        df['cpu_score'].fillna(0) * W_CPU +
        df['mem_score'].fillna(0) * W_MEM
    ).clip(0, 1)
    
    return df


def compute_rssi(distance_m: float, config) -> float:
    """Compute RSSI using log-distance path loss model."""
    if distance_m < config.reference_distance_m:
        distance_m = config.reference_distance_m
    
    path_loss = 10 * config.path_loss_exponent * np.log10(
        distance_m / config.reference_distance_m
    )
    
    return config.reference_rssi_dbm - path_loss


def estimate_client_distribution(
    df_aps: pd.DataFrame,
    new_ap_lat: float,
    new_ap_lon: float,
    config,
    mode: str = 'hybrid'
) -> Tuple[pd.DataFrame, Dict]:
    """Simulate client redistribution when a new AP is added."""
    df = df_aps.copy()
    
    df['dist_to_new'] = _haversine_m(
        new_ap_lat, new_ap_lon,
        df['lat'].values, df['lon'].values
    )
    
    df['rssi_new'] = df['dist_to_new'].apply(lambda d: compute_rssi(d, config))
    
    df['in_range'] = df['dist_to_new'] <= config.interference_radius_m
    
    total_transferred = 0
    
    candidates = df[df['in_range'] & (df['client_count'] > 0)].copy()
    
    if not candidates.empty:
        candidates = candidates.sort_values('dist_to_new', ascending=True)
        
        for idx, row in candidates.iterrows():
            signal_strength = max(0.0, (row['rssi_new'] - config.min_rssi_dbm) / 20.0)
            signal_strength = min(1.0, signal_strength)
            
            distance_factor = 1.0 - (row['dist_to_new'] / config.interference_radius_m)
            distance_factor = max(0.0, min(1.0, distance_factor))
            
            conflict_factor = float(row.get('conflictivity', 0.5))
            
            transfer_potential = (
                0.30 * signal_strength +
                0.30 * distance_factor +
                0.40 * conflict_factor
            )
            
            transfer_fraction = min(config.max_offload_fraction, transfer_potential * 0.8)
            
            transfer_fraction *= (1 - config.sticky_client_fraction)
            
            n_transfer = max(1, int(row['client_count'] * transfer_fraction))
            
            n_transfer = min(n_transfer, int(row['client_count']))
            
            if n_transfer > 0:
                df.at[idx, 'client_count'] = max(0, row['client_count'] - n_transfer)
                total_transferred += n_transfer
    
    if total_transferred == 0 and not candidates.empty:
        closest_with_clients = candidates.head(1)
        if not closest_with_clients.empty:
            idx = closest_with_clients.index[0]
            n_transfer = max(1, int(df.at[idx, 'client_count'] * 0.1))
            df.at[idx, 'client_count'] = max(0, df.at[idx, 'client_count'] - n_transfer)
            total_transferred = n_transfer
    
    new_ap_stats = {
        'lat': new_ap_lat,
        'lon': new_ap_lon,
        'client_count': total_transferred,
        'name': 'AP-NEW-SIM',
        'group_code': 'SIM',
    }
    
    client_fraction = min(1.0, total_transferred / config.max_clients_per_ap)
    new_ap_stats['util_2g'] = client_fraction * config.target_util_2g
    new_ap_stats['util_5g'] = client_fraction * config.target_util_5g
    
    return df, new_ap_stats


def apply_cca_interference(
    df_aps: pd.DataFrame,
    new_ap_stats: Dict,
    config,
) -> pd.DataFrame:
    """Apply co-channel interference (CCA busy increase) to neighbors."""
    df = df_aps.copy()
    
    distances = _haversine_m(
        new_ap_stats['lat'], new_ap_stats['lon'],
        df['lat'].values, df['lon'].values
    )
    
    in_interference_range = distances <= config.interference_radius_m
    
    increase_factor = np.where(
        in_interference_range,
        config.cca_increase_factor * (1 - distances / config.interference_radius_m),
        0.0
    )
    
    df['util_2g'] = np.clip(
        df['util_2g'] * (1 + increase_factor),
        0.0, 100.0
    )
    df['util_5g'] = np.clip(
        df['util_5g'] * (1 + increase_factor),
        0.0, 100.0
    )
    
    df['agg_util'] = np.maximum(df['util_2g'], df['util_5g'])
    
    return df


def simulate_ap_addition(
    df_baseline: pd.DataFrame,
    new_ap_lat: float,
    new_ap_lon: float,
    config,
    scorer,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Simulate adding a new AP at given location."""
    df_updated, new_ap_stats = estimate_client_distribution(
        df_baseline, new_ap_lat, new_ap_lon, config, mode='hybrid'
    )
    
    df_updated = apply_cca_interference(df_updated, new_ap_stats, config)
    
    df_updated = recalculate_conflictivity(df_updated)
    
    distances = _haversine_m(
        new_ap_lat, new_ap_lon,
        df_updated['lat'].values, df_updated['lon'].values
    )
    neighbor_mask = distances <= config.interference_radius_m
    
    baseline_conf = df_baseline['conflictivity'].values
    updated_conf = df_updated['conflictivity'].values
    
    component_scores = scorer.compute_component_scores(
        baseline_conf,
        updated_conf,
        neighbor_mask,
    )
    
    composite_score = scorer.compute_composite_score(component_scores)
    
    warnings = scorer.generate_warnings(component_scores)
    
    metrics = {
        **component_scores,
        'composite_score': composite_score,
        'warnings': warnings,
        
        'avg_conflictivity_before': float(baseline_conf.mean()),
        'avg_conflictivity_after': float(updated_conf.mean()),
        'avg_reduction': float(baseline_conf.mean() - updated_conf.mean()),
        'avg_reduction_pct': float((baseline_conf.mean() - updated_conf.mean()) / baseline_conf.mean() * 100) if baseline_conf.mean() > 0 else 0.0,
        
        'worst_ap_conflictivity_before': float(baseline_conf.max()),
        'worst_ap_conflictivity_after': float(updated_conf.max()),
        'worst_ap_improvement': float(baseline_conf.max() - updated_conf.max()),
        
        'num_high_conflict_before': int((baseline_conf > 0.7).sum()),
        'num_high_conflict_after': int((updated_conf > 0.7).sum()),
        
        'new_ap_client_count': new_ap_stats['client_count'],
        'new_ap_util_2g': new_ap_stats['util_2g'],
        'new_ap_util_5g': new_ap_stats['util_5g'],
    }
    
    return df_updated, new_ap_stats, metrics


def generate_candidate_locations(
    df_aps: pd.DataFrame,
    tile_meters: float,
    conflictivity_threshold: float,
    radius_m: float,
    indoor_only: bool = True,
    neighbor_radius_tiles: int = 1,
    inner_clearance_m: float = 0.0,
) -> pd.DataFrame:
    """Generate candidate locations for new AP placement."""
    lons = df_aps['lon'].values
    lats = df_aps['lat'].values
    pts = [Point(xy) for xy in zip(lons, lats)]
    mp = MultiPoint(pts)
    hull = mp.convex_hull
    
    lat0 = float(np.mean(lats))
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    dlat = tile_meters / meters_per_deg_lat
    dlon = tile_meters / meters_per_deg_lon
    
    minx, miny, maxx, maxy = hull.bounds
    lon_centers = np.arange(minx + dlon/2, maxx, dlon)
    lat_centers = np.arange(miny + dlat/2, maxy, dlat)
    XX, YY = np.meshgrid(lon_centers, lat_centers)
    
    centers = np.column_stack([XX.ravel(), YY.ravel()])
    
    x_h, y_h = hull.exterior.coords.xy
    poly_path = MplPath(np.vstack([x_h, y_h]).T)
    inside = poly_path.contains_points(centers)
    centers_in = centers[inside]
    
    if len(centers_in) == 0:
        return pd.DataFrame()
    
    dists = _haversine_m(
        centers_in[:, 1][:, None],
        centers_in[:, 0][:, None],
        lats[None, :],
        lons[None, :]
    )
    d_min = dists.min(axis=1)
    
    R_m = radius_m
    W = np.maximum(0, 1 - dists / R_m)
    W[dists >= R_m] = 0
    
    cvals = df_aps['conflictivity'].values
    denom = W.sum(axis=1)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        num = (W * cvals[None, :]).sum(axis=1)
    z_pred = np.where(denom > 0, num / denom, 0.0)
    
    painted_tiles = set()
    conf_grid = np.full((len(lat_centers), len(lon_centers)), np.nan)
    coord_to_idx = {}
    
    for i, (lon, lat) in enumerate(centers_in):
        lon_idx = min(range(len(lon_centers)), key=lambda x: abs(lon_centers[x] - lon))
        lat_idx = min(range(len(lat_centers)), key=lambda x: abs(lat_centers[x] - lat))
        
        painted_tiles.add((lat_idx, lon_idx))
        conf_grid[lat_idx, lon_idx] = z_pred[i]
        coord_to_idx[(lat_idx, lon_idx)] = i
    
    neighbor_offsets = [
        (dy, dx)
        for dy in range(-neighbor_radius_tiles, neighbor_radius_tiles + 1)
        for dx in range(-neighbor_radius_tiles, neighbor_radius_tiles + 1)
        if not (dy == 0 and dx == 0)
    ]
    
    valid_tiles_mask = np.zeros(len(centers_in), dtype=bool)
    boundary_mask = np.zeros(len(centers_in), dtype=bool)
    
    for i, (lon, lat) in enumerate(centers_in):
        lon_idx = min(range(len(lon_centers)), key=lambda x: abs(lon_centers[x] - lon))
        lat_idx = min(range(len(lat_centers)), key=lambda x: abs(lat_centers[x] - lat))
        
        immediate_offsets = [
            (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)
        ]
        is_boundary = False
        for dy, dx in immediate_offsets:
            n_lat = lat_idx + dy
            n_lon = lon_idx + dx
            if (n_lat, n_lon) not in painted_tiles:
                is_boundary = True
                break
        boundary_mask[i] = is_boundary

        all_neighbors_painted = True
        for dlat_idx, dlon_idx in neighbor_offsets:
            neighbor_lat_idx = lat_idx + dlat_idx
            neighbor_lon_idx = lon_idx + dlon_idx
            if (neighbor_lat_idx, neighbor_lon_idx) not in painted_tiles:
                all_neighbors_painted = False
                break
        if all_neighbors_painted and not is_boundary:
            valid_tiles_mask[i] = True
    
    inner_mask = d_min < max(0.0, radius_m - inner_clearance_m)
    final_mask = valid_tiles_mask & inner_mask
    
    centers_in = centers_in[final_mask]
    z_pred = z_pred[final_mask]
    boundary_mask = boundary_mask[final_mask]
    d_min = d_min[final_mask]
    
    if len(centers_in) == 0:
        return pd.DataFrame()
    
    candidates = pd.DataFrame({
        'lon': centers_in[:, 0],
        'lat': centers_in[:, 1],
        'conflictivity': z_pred,
        'is_boundary': boundary_mask,
        'min_dist_m': d_min,
    })
    
    candidates = candidates[(candidates['conflictivity'] >= conflictivity_threshold) & (~candidates['is_boundary'])].copy()
    
    candidates['reason'] = 'high_conflictivity_interior_non_boundary'
    
    return candidates.reset_index(drop=True)


def generate_voronoi_candidates(
    scenarios: list,
    geo_df: pd.DataFrame,
    radius_m: float,
    conflictivity_threshold: float,
    tile_radius_clearance_m: float,
    merge_radius_m: float = 8.0,
    max_vertices_per_scenario: int = 60,
) -> pd.DataFrame:
    """Generate AP candidate locations using Voronoi vertices across multiple scenarios."""
    if not _HAS_SCIPY_VORONOI:
        st.error("SciPy not available: install scipy to enable Voronoi candidate mode.")
        return pd.DataFrame()

    records = []
    effective_max_dist = max(0.0, radius_m - tile_radius_clearance_m)

    for profile, snap_path, snap_dt in scenarios:
        try:
            df_snap = read_ap_snapshot(snap_path, band_mode='worst')
            df_snap = df_snap.merge(geo_df, on='name', how='inner')
            df_snap = df_snap[df_snap['group_code'] != 'SAB'].copy()
            if df_snap.empty:
                continue
            pts_xy = df_snap[['lon', 'lat']].to_numpy()
            if len(pts_xy) < 3:
                continue
            vor = Voronoi(pts_xy)
            hull_poly = MultiPoint([Point(xy) for xy in pts_xy]).convex_hull
            lons = df_snap['lon'].to_numpy()
            lats = df_snap['lat'].to_numpy()
            cvals = df_snap['conflictivity'].to_numpy()
            for vidx, (vx, vy) in enumerate(vor.vertices):
                p = Point(vx, vy)
                if not hull_poly.contains(p):
                    continue
                dists = _haversine_m(vy, vx, lats, lons)
                d_min = float(dists.min())
                if d_min >= effective_max_dist:
                    continue
                w = np.maximum(0.0, 1 - dists / radius_m)
                w[dists >= radius_m] = 0.0
                if w.sum() == 0:
                    continue
                conf_pred = float((w * cvals).sum() / w.sum())
                if conf_pred < conflictivity_threshold:
                    continue
                records.append({
                    'lon': vx,
                    'lat': vy,
                    'conflictivity': conf_pred,
                    'min_dist_m': d_min,
                    'scenario_ts': snap_dt,
                    'stress_profile': profile.value,
                })
        except Exception as e:
            st.warning(f"Voronoi generation error for {snap_path.name}: {e}")
            continue

    if not records:
        return pd.DataFrame()

    df_all = pd.DataFrame(records)
    df_all['scenario_key'] = df_all['scenario_ts'].astype(str)
    df_all = df_all.sort_values(['scenario_key', 'conflictivity'], ascending=[True, False])
    df_all = df_all.groupby('scenario_key').head(max_vertices_per_scenario).reset_index(drop=True)

    clusters = []
    merge_r = merge_radius_m
    for row in df_all.itertuples():
        placed = False
        for cl in clusters:
            d = _haversine_m(row.lat, row.lon, cl['lat'], cl['lon'])
            if d <= merge_r:
                cl['points'].append(row)
                lats_cl = [p.lat for p in cl['points']]
                lons_cl = [p.lon for p in cl['points']]
                cl['lat'] = float(np.mean(lats_cl))
                cl['lon'] = float(np.mean(lons_cl))
                placed = True
                break
        if not placed:
            clusters.append({'lat': row.lat, 'lon': row.lon, 'points': [row]})

    out_rows = []
    for cl in clusters:
        pts = cl['points']
        confs = [p.conflictivity for p in pts]
        dmins = [p.min_dist_m for p in pts]
        scenarios_list = [str(p.scenario_ts) for p in pts]
        stress_profiles = [p.stress_profile for p in pts]
        out_rows.append({
            'lat': cl['lat'],
            'lon': cl['lon'],
            'avg_conflictivity': float(np.mean(confs)),
            'max_conflictivity': float(np.max(confs)),
            'freq': len(pts),
            'avg_min_dist_m': float(np.mean(dmins)),
            'scenarios': scenarios_list,
            'stress_profiles': stress_profiles,
        })

    df_clusters = pd.DataFrame(out_rows)
    df_clusters['conflictivity'] = df_clusters['avg_conflictivity']
    df_clusters = df_clusters.sort_values(['freq', 'avg_conflictivity'], ascending=[False, False]).reset_index(drop=True)
    return df_clusters


def aggregate_scenario_results(
    lat: float,
    lon: float,
    base_conflictivity: float,
    scenario_results: List[Dict],
) -> Dict:
    """Aggregate metrics across scenarios."""
    aggregated = {
        'lat': lat,
        'lon': lon,
        'base_conflictivity': base_conflictivity,
        'n_scenarios': len(scenario_results),
    }
    
    scores = [r['composite_score'] for r in scenario_results]
    aggregated['final_score'] = float(np.mean(scores))
    aggregated['score_std'] = float(np.std(scores))
    aggregated['score_min'] = float(np.min(scores))
    aggregated['score_max'] = float(np.max(scores))
    
    for key in ['worst_ap_improvement_raw', 'avg_reduction_raw', 'num_improved', 'new_ap_client_count']:
        values = [r.get(key, 0) for r in scenario_results]
        aggregated[f'{key}_mean'] = float(np.mean(values))
        aggregated[f'{key}_std'] = float(np.std(values))
    
    all_warnings = []
    for r in scenario_results:
        all_warnings.extend(r.get('warnings', []))
    
    warning_counts = Counter(all_warnings)
    aggregated['warnings'] = [
        f"{msg} (in {count}/{len(scenario_results)} scenarios)"
        for msg, count in warning_counts.most_common()
    ]
    
    by_profile = {}
    for r in scenario_results:
        profile = r.get('stress_profile', 'unknown')
        if profile not in by_profile:
            by_profile[profile] = []
        by_profile[profile].append(r['composite_score'])
    
    for profile, profile_scores in by_profile.items():
        aggregated[f'score_{profile}'] = float(np.mean(profile_scores))
    
    return aggregated


def simulate_multiple_ap_additions(
    df_baseline: pd.DataFrame,
    points: List[Dict],
    config,
) -> pd.DataFrame:
    """Approximate combined effect of adding multiple APs by applying them sequentially."""
    df_curr = df_baseline.copy()
    for i, p in enumerate(points, start=1):
        lat = float(p['lat'])
        lon = float(p['lon'])
        df_curr, new_ap_stats = estimate_client_distribution(df_curr, lat, lon, config, mode='hybrid')
        df_curr = apply_cca_interference(df_curr, new_ap_stats, config)
        new_row = {
            'name': f'AP-NEW-SIM-{i}',
            'group_code': 'SIM',
            'lat': lat,
            'lon': lon,
            'client_count': new_ap_stats.get('client_count', 0),
            'util_2g': new_ap_stats.get('util_2g', 0.0),
            'util_5g': new_ap_stats.get('util_5g', 0.0),
            'cpu_utilization': 0.0,
            'mem_used_pct': 0.0,
            'agg_util': max(new_ap_stats.get('util_2g', 0.0), new_ap_stats.get('util_5g', 0.0)),
        }
        df_curr = pd.concat([df_curr, pd.DataFrame([new_row])], ignore_index=True)
        df_curr = recalculate_conflictivity(df_curr)
    return df_curr

# -------- UI --------
st.set_page_config(page_title="UAB Wi‑Fi Integrated Dashboard", page_icon="📶", layout="wide")
st.title("UAB Wi‑Fi Integrated Dashboard")
st.caption("AI Heatmap + Voronoi + Simulator • Time series visualization")

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
    st.header("Visualization Mode")
    
    # Show Simulator option only if simulator module is available
    if SIMULATOR_AVAILABLE:
        viz_mode = st.radio(
            "Select Mode",
            options=["AI Heatmap", "Voronoi", "Simulator"],
            index=0,
            help="AI Heatmap: Click APs for AINA analysis | Voronoi: Interpolated surfaces | Simulator: AP placement optimization"
        )
    else:
        viz_mode = st.radio(
            "Select Mode",
            options=["AI Heatmap", "Voronoi"],
            index=0,
            help="AI Heatmap: Click APs for AINA analysis | Voronoi: Interpolated surfaces with connectivity"
        )
    
    st.divider()
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
    
    # Mode-specific controls
    if viz_mode == "AI Heatmap":
        radius = 5
        min_conf = st.slider("Minimum conflictivity", 0.0, 1.0, 0.0, 0.01)
        top_n = st.slider("Top N listing (table)", 5, 50, 15, step=5)
    elif viz_mode == "Voronoi":
        radius_m = st.slider("Radi de connectivitat (m)", 5, 60, 25, step=5,
                           help="Distància màxima perquè la connectivitat arribi a 1")
        value_mode = st.selectbox("Mode de valor", ["conflictivity", "connectivity"], index=0,
                                help="conflictivity: ponderació dels APs; connectivity: creix fins a 1 al radi")
        TILE_M_FIXED = 7.0
        MAX_TILES_NO_LIMIT = 1_000_000_000
        
        st.divider()
        st.header("Voronoi ponderat")
        show_awvd = True
        weight_source = st.selectbox(
            "Base connectivitat (per invertir)",
            ["conflictivity", "client_count", "max_radio_util", "airtime_score"],
            index=0,
            help="Es normalitza i s'inverteix: pes = 1 - norm(col)."
        )
        VOR_TOL_M_FIXED = 24.0
        SNAP_M_DEFAULT = float(max(1.5, TILE_M_FIXED * 0.2))
        JOIN_M_DEFAULT = float(max(3.0, TILE_M_FIXED * 0.6))
        show_hot_vertex = st.checkbox("Marcar punt més conflictiu (vertex Voronoi)", value=False,
                                    help="Evalua vertices del Voronoi i marca el de major conflictivitat.")
        min_conf = 0.0
        top_n = 15
    else:  # Simulator
        radius_m = 25
        value_mode = "conflictivity"
        TILE_M_FIXED = 7.0
        MAX_TILES_NO_LIMIT = 1_000_000_000
        min_conf = 0.0
        top_n = 15
        
        st.divider()
        st.header("🎯 AP Placement Simulator")
        
        run_simulation = True
        
        if run_simulation:
            st.subheader("Simulation Parameters")
            
            col_basic1, col_basic2 = st.columns(2)
            
            with col_basic1:
                sim_top_k = st.slider("Number of candidates to evaluate", 1, 10, 3, 
                                      help="How many top placement locations to test")
                
                sim_stress_profile = st.selectbox(
                    "Network condition to optimize for",
                    ["HIGH (Peak hours)", "CRITICAL (Overloaded)", "ALL (Robust)"],
                    index=0,
                    help="Which network stress level to prioritize"
                )
                
                sim_candidate_mode = st.selectbox(
                    "Candidate generation method",
                    ["Tile-based (uniform grid)", "Voronoi (network-aware)"],
                    index=0,
                    help="Tile: uniform grid | Voronoi: uses network topology vertices"
                )
            
            with col_basic2:
                sim_threshold = st.slider("Min conflictivity threshold", 0.4, 0.8, 0.6, 0.05,
                                          help="Only consider areas with high network stress")
                
                sim_snapshots_per_profile = st.slider(
                    "Test scenarios", 3, 10, 5,
                    help="More scenarios = more confidence, but slower"
                )
                
                if sim_candidate_mode == "Voronoi (network-aware)":
                    sim_merge_radius = st.slider(
                        "Voronoi merge radius (m)", 5, 15, 8, 1,
                        help="Nearby Voronoi vertices merged within this distance"
                    )
                else:
                    sim_merge_radius = 8
            
            with st.expander("⚙️ Advanced Settings (Optional)", expanded=False):
                st.caption("**Physics Parameters**")
                col_adv1, col_adv2 = st.columns(2)
                
                with col_adv1:
                    sim_interference_radius = st.slider(
                        "Interference radius (m)", 30, 80, 50, 5,
                        help="How far the new AP affects neighbors (typical: 50m)"
                    )
                    
                    sim_cca_increase = st.slider(
                        "Co-channel interference", 0.05, 0.30, 0.15, 0.05,
                        help="How much neighbors' utilization increases (typical: 15%)"
                    )
                
                with col_adv2:
                    st.caption("**Scoring Weights** (must sum to 1.0)")
                    w_worst = st.number_input("Worst AP", 0.0, 1.0, 0.30, 0.05, 
                                             help="Reduce worst-case overload")
                    w_avg = st.number_input("Average", 0.0, 1.0, 0.30, 0.05, 
                                           help="Overall network improvement")
                    w_cov = st.number_input("Coverage", 0.0, 1.0, 0.20, 0.05, 
                                           help="# of APs improved")
                    w_neigh = st.number_input("Neighborhood", 0.0, 1.0, 0.20, 0.05, 
                                             help="Protect nearby APs")
                
                total_weight = w_worst + w_avg + w_cov + w_neigh
                if abs(total_weight - 1.0) > 0.01:
                    st.warning(f"⚠️ Weights must sum to 1.0 (current: {total_weight:.2f})")
                
                st.caption("**Candidate Filters**")
                col_cf1, col_cf2 = st.columns(2)
                with col_cf1:
                    sim_interior_buffer_tiles = st.slider(
                        "Interior buffer (tiles)", 1, 4, 2,
                        help="How many tile rings inside the painted area a candidate must be (avoids outer transparent edge)"
                    )
                with col_cf2:
                    sim_inner_clearance_m = st.slider(
                        "Clearance from radius band (m)", 0, int(radius_m), 10,
                        help="Exclude tiles near the interpolation radius band (blue inner hull). Higher = farther from the red ring"
                    )
            
            if 'sim_interference_radius' not in locals():
                sim_interference_radius = 50
                sim_cca_increase = 0.15
                w_worst, w_avg, w_cov, w_neigh = 0.30, 0.30, 0.20, 0.20
                total_weight = 1.0
                sim_interior_buffer_tiles = 2
                sim_inner_clearance_m = 10
            
            stress_display_map = {
                "HIGH (Peak hours)": "HIGH",
                "CRITICAL (Overloaded)": "CRITICAL",
                "ALL (Robust)": "ALL"
            }
            sim_stress_profile_key = stress_display_map[sim_stress_profile]
            
            weights_ok = abs(total_weight - 1.0) <= 0.01
            sim_param_payload = {
                'top_k': sim_top_k,
                'threshold': sim_threshold,
                'stress_profile': sim_stress_profile_key,
                'snapshots': sim_snapshots_per_profile,
                'interference_radius': sim_interference_radius,
                'cca_increase': sim_cca_increase,
                'w_worst': w_worst,
                'w_avg': w_avg,
                'w_cov': w_cov,
                'w_neigh': w_neigh,
                'candidate_mode': sim_candidate_mode,
                'merge_radius': sim_merge_radius,
                'interior_buffer_tiles': sim_interior_buffer_tiles,
                'inner_clearance_m': sim_inner_clearance_m,
            }

            if weights_ok:
                st.session_state.run_sim = True
                st.session_state.sim_params = sim_param_payload
            else:
                st.session_state.run_sim = False
                st.info("Ajusta els pesos perquè sumin 1.0 per executar la simulació automàticament.")

            run_simulation = weights_ok
        
        # Voronoi Candidate Discovery (Step 1 of new workflow)
        st.divider()
        st.subheader("🧩 Voronoi Candidate Discovery")
        st.caption("Detect stable high-conflictivity Voronoi vertex clusters across representative scenarios before full simulation.")
        if not _HAS_SCIPY_VORONOI:
            st.warning("SciPy Voronoi is required to auto-detect candidate vertices.")
        else:
            voronoi_signature = (
                sim_stress_profile_key,
                sim_snapshots_per_profile,
                round(radius_m, 2),
                round(sim_threshold, 3),
                round(sim_inner_clearance_m, 2),
                round(sim_merge_radius, 2),
            )
            prev_signature = st.session_state.get("voronoi_signature")
            need_detection = (
                'voronoi_candidates' not in st.session_state
                or st.session_state.voronoi_candidates is None
                or st.session_state.voronoi_candidates.empty
                or prev_signature != voronoi_signature
            )

            if need_detection:
                with st.spinner("🔍 Detectant vertices Voronoi automàticament..."):
                    stress_map = {
                        "HIGH": StressLevel.HIGH,
                        "CRITICAL": StressLevel.CRITICAL,
                        "MEDIUM": StressLevel.MEDIUM,
                        "LOW": StressLevel.LOW,
                        "ALL": None
                    }
                    target_stress = stress_map.get(sim_stress_profile_key, StressLevel.HIGH)
                    profiler = StressProfiler(
                        snapshots,
                        utilization_threshold_critical=85,
                        utilization_threshold_high=70,
                    )
                    stress_profiles = profiler.classify_snapshots()
                    stats = profiler.get_profile_statistics()
                    profiles_to_test, effective_target, profile_message = resolve_stress_profiles(target_stress, stats)
                    if profile_message:
                        st.info(f"ℹ️ {profile_message}")
                    all_scenarios = []
                    for profile in profiles_to_test:
                        snaps_sel = profiler.get_representative_snapshots(profile, n_samples=sim_snapshots_per_profile)
                        for path, dt in snaps_sel:
                            all_scenarios.append((profile, path, dt))
                    st.session_state.voronoi_scenarios = all_scenarios
                    if not all_scenarios:
                        st.warning("No scenarios available for Voronoi detection.")
                    else:
                        st.info(f"Voronoi: Using {len(all_scenarios)} scenarios across {len(profiles_to_test)} profiles.")
                        vor_df = generate_voronoi_candidates(
                            all_scenarios,
                            geo_df=geo_df,
                            radius_m=radius_m,
                            conflictivity_threshold=sim_threshold,
                            tile_radius_clearance_m=sim_inner_clearance_m,
                            merge_radius_m=sim_merge_radius,
                            max_vertices_per_scenario=60,
                        )
                        st.session_state.voronoi_candidates = vor_df
                        st.session_state.voronoi_signature = voronoi_signature
                        if vor_df.empty:
                            st.warning("No Voronoi candidates detected. Try lowering conflictivity threshold or clearance.")
                        else:
                            st.success(f"Detected {len(vor_df)} Voronoi candidate clusters.")
            else:
                st.caption("Voronoi candidates already generated for the current parameters.")

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

# Dialog function for AINA AI analysis (used in AI Heatmap mode)
@st.dialog("🤖 Anàlisi AINA AI", width="large", on_dismiss=on_dialog_close)
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

# ========== VISUALIZATION LOGIC ==========

if viz_mode == "AI Heatmap":
    # ========== AI HEATMAP MODE ==========
    fig = create_optimized_heatmap(
        df=map_df,
        center_lat=center_lat,
        center_lon=center_lon,
        min_conflictivity=min_conf,
        radius=5,
        zoom=15,
    )

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

elif viz_mode == "Voronoi":
    # ========== VORONOI MODE ==========
    tmp = map_df.copy()
    if "group_code" not in tmp.columns:
        tmp["group_code"] = tmp["name"].apply(extract_group)
    
    sab_df = tmp[tmp["group_code"] == "SAB"].copy()
    uab_df = tmp[tmp["group_code"] != "SAB"].copy()

    fig = go.Figure()

    # UAB interpolation
    if not uab_df.empty:
        ch, eff_tile, hull = _uab_tiled_choropleth_layer(
            uab_df, tile_meters=TILE_M_FIXED, radius_m=radius_m, mode="decay",
            value_mode=value_mode, max_tiles=MAX_TILES_NO_LIMIT
        )
        if ch is not None:
            fig.add_trace(ch)
            fig.add_annotation(text=f"UAB tile ≈ {eff_tile:.1f} m",
                             showarrow=False, xref="paper", yref="paper",
                             x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="#888", font=dict(size=10))
        
        # AP points
        fig.add_trace(go.Scattermapbox(
            lat=uab_df['lat'], lon=uab_df['lon'], mode='markers',
            marker=dict(size=7, color='black', opacity=0.7),
            text=uab_df['name'], name="UAB APs",
            hovertemplate='<b>%{text}</b><br>Conflictivity src point<extra></extra>'
        ))

    # Voronoi weighted edges
    hot_vertices_info = None
    if show_awvd:
        aw_df = map_df[map_df["group_code"] != "SAB"].copy()
        base_col = weight_source if weight_source in aw_df.columns else "conflictivity"
        if not aw_df.empty and base_col in aw_df.columns:
            regions = _coverage_regions_from_uab_tiles(
                aw_df,
                tile_meters=float(TILE_M_FIXED),
                radius_m=radius_m,
                max_tiles=int(MAX_TILES_NO_LIMIT)
            )
            union_poly = None
            if regions:
                union_poly = unary_union(regions)
            else:
                union_poly = _compute_convex_hull_polygon(aw_df["lon"].to_numpy(float), aw_df["lat"].to_numpy(float))

            dedup = aw_df[["lon","lat", base_col]].copy()
            dedup = (dedup.groupby(["lon","lat"], as_index=False)
                           .agg({base_col: "max"}))

            if union_poly is not None:
                try:
                    lat0 = float(aw_df["lat"].mean()) if len(aw_df) else 41.5
                    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
                    eps_deg = (max(0.5, TILE_M_FIXED * 0.15)) / max(meters_per_deg_lon, 1e-6)
                    union_poly = union_poly.buffer(eps_deg)
                except Exception:
                    pass

            total_edges = _inverted_weighted_voronoi_edges(
                dedup.rename(columns={base_col: base_col}),
                weight_col=base_col,
                radius_m=radius_m,
                clip_polygon=union_poly,
                tolerance_m=VOR_TOL_M_FIXED
            ) if union_poly is not None and len(dedup) >= 3 else []

            if total_edges:
                lat0 = float(aw_df["lat"].mean()) if len(aw_df) else 41.5
                merged_lines = _snap_and_connect_edges(
                    total_edges,
                    union_poly,
                    lat0=lat0,
                    snap_m=SNAP_M_DEFAULT,
                    join_m=JOIN_M_DEFAULT,
                ) or linemerge(unary_union([LineString([(x1, y1), (x2, y2)]) for (x1, y1, x2, y2) in total_edges]))
                lons = []
                lats = []
                def add_lines(ls):
                    coords = list(ls.coords)
                    if len(coords) >= 2:
                        for (x, y) in coords:
                            lons.append(x)
                            lats.append(y)
                        lons.append(None)
                        lats.append(None)
                if merged_lines.geom_type == 'LineString':
                    add_lines(merged_lines)
                elif merged_lines.geom_type == 'MultiLineString':
                    for ls in merged_lines.geoms:
                        add_lines(ls)
                fig.add_trace(go.Scattermapbox(
                    lon=lons,
                    lat=lats,
                    mode='lines',
                    line=dict(color='#0b3d91', width=2),
                    name='Voronoi (ponderat, invertit)',
                    hoverinfo='skip'
                ))
                
                # Top hotspot vertices
                if show_hot_vertex and union_poly is not None:
                    topv = _top_conflictive_voronoi_vertices(aw_df, radius_m=radius_m, coverage_poly=union_poly, k=3)
                    if topv:
                        hot_vertices_info = [
                            {"rank": i+1, "lon": v[0], "lat": v[1], "score": float(v[2])}
                            for i, v in enumerate(topv)
                        ]
                        # Marker #1
                        hv_lon, hv_lat, hv_score = topv[0]
                        fig.add_trace(go.Scattermapbox(
                            lon=[hv_lon], lat=[hv_lat], mode='markers',
                            marker=dict(size=24, color='#ffffff', symbol='circle', opacity=0.95),
                            hoverinfo='skip', showlegend=False
                        ))
                        fig.add_trace(go.Scattermapbox(
                            lon=[hv_lon], lat=[hv_lat], mode='markers+text',
                            marker=dict(
                                size=18,
                                color='#ff00ff',
                                symbol='star',
                                opacity=0.95
                            ),
                            text=["#1"], textposition='top center',
                            textfont=dict(color='#ffffff', size=12, family='Arial Black'),
                            name='Hotspot #1',
                            hovertemplate='<b>#1 Hotspot</b><br>Score=%{customdata:.3f}<extra></extra>',
                            customdata=[hv_score]
                        ))
                        # Markers #2-#3
                        if len(topv) > 1:
                            lons = [t[0] for t in topv[1:]]
                            lats = [t[1] for t in topv[1:]]
                            scores = [float(t[2]) for t in topv[1:]]
                            labels = [f"#{i+2}" for i in range(len(lons))]
                            fig.add_trace(go.Scattermapbox(
                                lon=lons, lat=lats, mode='markers',
                                marker=dict(size=20, color='#ffffff', symbol='circle', opacity=0.95),
                                hoverinfo='skip', showlegend=False
                            ))
                            fig.add_trace(go.Scattermapbox(
                                lon=lons, lat=lats, mode='markers+text',
                                marker=dict(
                                    size=16,
                                    color='#00ffff',
                                    symbol='star',
                                    opacity=0.95
                                ),
                                text=labels, textposition='top center',
                                textfont=dict(color='#ffffff', size=11, family='Arial Black'),
                                name='Hotspot #2-#3',
                                hovertemplate='<b>%{text}</b><br>Score=%{customdata:.3f}<extra></extra>',
                                customdata=scores
                            ))
                
            # Draw coverage hull
            if union_poly is not None:
                hull_lons = []
                hull_lats = []
                polys = [union_poly] if union_poly.geom_type == 'Polygon' else list(union_poly.geoms)
                for reg in polys:
                    xh, yh = reg.exterior.coords.xy
                    hull_lons.extend(list(xh) + [None])
                    hull_lats.extend(list(yh) + [None])
                    for ring in reg.interiors:
                        xi, yi = zip(*list(ring.coords))
                        hull_lons.extend(list(xi) + [None])
                        hull_lats.extend(list(yi) + [None])
                if hull_lons:
                    fig.add_trace(go.Scattermapbox(
                        lon=hull_lons,
                        lat=hull_lats,
                        mode='lines',
                        line=dict(color='#0b3d91', width=1),
                        name='Coverage hulls',
                        hoverinfo='skip'
                    ))
                n_regs = 1 if union_poly.geom_type == 'Polygon' else len(list(union_poly.geoms))
                fig.add_annotation(text=f"Voronoi ponderat (edges) — {n_regs} regions", xref="paper", yref="paper", x=0.02, y=0.90,
                                 showarrow=False, bgcolor="rgba(0,0,0,0.4)", font=dict(color='white', size=10))

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

    st.plotly_chart(fig, use_container_width=True)

    # Table with top hotspot vertices
    try:
        if show_awvd and show_hot_vertex and hot_vertices_info:
            st.subheader("Top vertices Voronoi més conflictius")
            top_df = pd.DataFrame(hot_vertices_info)
            top_df["score"] = top_df["score"].map(lambda x: f"{x:.3f}")
            st.dataframe(top_df, use_container_width=True, hide_index=True)
    except Exception:
        pass

else:  # Simulator
    # ========== SIMULATOR MODE ==========
    tmp = map_df.copy()
    if "group_code" not in tmp.columns:
        tmp["group_code"] = tmp["name"].apply(extract_group)
    
    sab_df = tmp[tmp["group_code"] == "SAB"].copy()
    uab_df = tmp[tmp["group_code"] != "SAB"].copy()

    fig = go.Figure()

    # UAB interpolation
    if not uab_df.empty:
        ch, eff_tile, hull = _uab_tiled_choropleth_layer(
            uab_df, tile_meters=TILE_M_FIXED, radius_m=radius_m, mode="decay",
            value_mode=value_mode, max_tiles=MAX_TILES_NO_LIMIT
        )
        if ch is not None:
            fig.add_trace(ch)
            fig.add_annotation(text=f"UAB tile ≈ {eff_tile:.1f} m",
                             showarrow=False, xref="paper", yref="paper",
                             x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="#888", font=dict(size=10))
        
        # AP points
        fig.add_trace(go.Scattermapbox(
            lat=uab_df['lat'], lon=uab_df['lon'], mode='markers',
            marker=dict(size=7, color='black', opacity=0.7),
            text=uab_df['name'], name="UAB APs",
            hovertemplate='<b>%{text}</b><br>Conflictivity: %{customdata:.2f}<extra></extra>',
            customdata=uab_df['conflictivity']
        ))

    # Overlay Voronoi candidate markers if discovered
    if 'voronoi_candidates' in st.session_state and not st.session_state.voronoi_candidates.empty:
        vor_df = st.session_state.voronoi_candidates
        fig.add_trace(go.Scattermapbox(
            lat=vor_df['lat'],
            lon=vor_df['lon'],
            mode='markers+text',
            marker=dict(size=7, color='orange', opacity=0.85),
            text=[f"AP-VOR-{i+1}" for i in range(len(vor_df))],
            textposition="top center",
            name='Voronoi Candidates',
            hovertemplate='<b>%{text}</b><br>Avg Conflictivity: %{customdata[0]:.3f}<br>Freq: %{customdata[1]:.0f}<extra></extra>',
            customdata=np.column_stack([vor_df['avg_conflictivity'], vor_df['freq']])
        ))
    
    # Run simulation if enabled
    if run_simulation and st.session_state.get('run_sim', False) and SIMULATOR_AVAILABLE:
        params = st.session_state.get('sim_params', {})
        sim_top_k = params.get('top_k', 3)
        sim_threshold = params.get('threshold', 0.6)
        sim_stress_profile = params.get('stress_profile', 'HIGH')
        sim_snapshots_per_profile = params.get('snapshots', 5)
        sim_interference_radius = params.get('interference_radius', 50)
        sim_cca_increase = params.get('cca_increase', 0.15)
        w_worst = params.get('w_worst', 0.30)
        w_avg = params.get('w_avg', 0.30)
        w_cov = params.get('w_cov', 0.20)
        w_neigh = params.get('w_neigh', 0.20)
        sim_candidate_mode = params.get('candidate_mode', 'Tile-based (uniform grid)')
        sim_merge_radius = params.get('merge_radius', 8)
        sim_interior_buffer_tiles = params.get('interior_buffer_tiles', 2)
        sim_inner_clearance_m = params.get('inner_clearance_m', 10)

        with st.spinner("🔍 Running multi-scenario AP placement simulation..."):
            try:
                stress_map = {
                    "HIGH": StressLevel.HIGH,
                    "CRITICAL": StressLevel.CRITICAL,
                    "MEDIUM": StressLevel.MEDIUM,
                    "LOW": StressLevel.LOW,
                    "ALL": None
                }
                target_stress = stress_map.get(sim_stress_profile, StressLevel.HIGH)

                config = SimulationConfig(
                    interference_radius_m=sim_interference_radius,
                    cca_increase_factor=sim_cca_increase,
                    indoor_only=True,
                    conflictivity_threshold_placement=sim_threshold,
                    snapshots_per_profile=sim_snapshots_per_profile,
                    target_stress_profile=target_stress,
                    weight_worst_ap=w_worst,
                    weight_average=w_avg,
                    weight_coverage=w_cov,
                    weight_neighborhood=w_neigh,
                )

                profiler = StressProfiler(
                    snapshots,
                    utilization_threshold_critical=config.utilization_threshold_critical,
                    utilization_threshold_high=config.utilization_threshold_high,
                )

                st.info("📊 Classifying snapshots by stress level...")
                stress_profiles = profiler.classify_snapshots()

                stats = profiler.get_profile_statistics()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("LOW", f"{stats[StressLevel.LOW]['count']} snaps", 
                             f"{stats[StressLevel.LOW]['percentage']:.1f}%")
                with col2:
                    st.metric("MEDIUM", f"{stats[StressLevel.MEDIUM]['count']} snaps",
                             f"{stats[StressLevel.MEDIUM]['percentage']:.1f}%")
                with col3:
                    st.metric("HIGH", f"{stats[StressLevel.HIGH]['count']} snaps",
                             f"{stats[StressLevel.HIGH]['percentage']:.1f}%")
                with col4:
                    st.metric("CRITICAL", f"{stats[StressLevel.CRITICAL]['count']} snaps",
                             f"{stats[StressLevel.CRITICAL]['percentage']:.1f}%")

                profiles_to_test, effective_target, profile_message = resolve_stress_profiles(target_stress, stats)
                config.target_stress_profile = effective_target
                if profile_message:
                    st.info(f"ℹ️ {profile_message}")

                should_run = True
                if not profiles_to_test:
                    st.warning("No snapshots available for the simulator right now. Showing current map only.")
                    st.session_state.run_sim = False
                    st.session_state.pop('map_override_df', None)
                    st.session_state.pop('map_preview_metrics', None)
                    st.session_state.pop('new_node_markers', None)
                    should_run = False
                else:
                    all_scenarios = []
                    for profile in profiles_to_test:
                        snaps = profiler.get_representative_snapshots(profile, n_samples=sim_snapshots_per_profile)
                        for path, dt in snaps:
                            all_scenarios.append((profile, path, dt))

                    if not all_scenarios:
                        st.warning("⚠️ Snapshot pool empty for the selected filters. Keeping current view.")
                        st.session_state.run_sim = False
                        st.session_state.pop('map_override_df', None)
                        st.session_state.pop('map_preview_metrics', None)
                        st.session_state.pop('new_node_markers', None)
                        should_run = False

                if should_run:
                    st.success(f"✅ Testing {len(all_scenarios)} scenarios across {len(profiles_to_test)} stress profile(s)")
                    
                    # Generate candidates based on mode
                    if sim_candidate_mode == "Voronoi (network-aware)":
                        st.info(f"📍 Generating Voronoi candidate locations (merge_radius={sim_merge_radius}m, threshold={sim_threshold})...")
                        candidates = generate_voronoi_candidates(
                            all_scenarios,
                            geo_df,
                            radius_m,
                            sim_threshold,
                            tile_radius_clearance_m=5.0,
                            merge_radius_m=sim_merge_radius,
                            max_vertices_per_scenario=60,
                        )
                    else:
                        first_path = all_scenarios[0][1]
                        df_first = read_ap_snapshot(first_path, band_mode='worst')
                        df_first = df_first.merge(geo_df, on='name', how='inner')
                        df_first = df_first[df_first['group_code'] != 'SAB'].copy()
                        
                        if df_first.empty:
                            st.warning("⚠️ No UAB APs available for simulation")
                            st.session_state.run_sim = False
                            should_run = False
                        else:
                            st.info(f"📍 Generating tile-based candidate locations (tile_size={TILE_M_FIXED}m, threshold={sim_threshold})...")
                            
                            candidates = generate_candidate_locations(
                                df_first,
                                tile_meters=TILE_M_FIXED,
                                conflictivity_threshold=sim_threshold,
                                radius_m=radius_m,
                                indoor_only=config.indoor_only,
                                neighbor_radius_tiles=sim_interior_buffer_tiles,
                                inner_clearance_m=sim_inner_clearance_m,
                            )
                    
                    if not should_run:
                        pass
                    else:
                        if candidates.empty:
                            st.warning(f"⚠️ No candidates found with conflictivity > {sim_threshold}")
                            st.session_state.run_sim = False
                        else:
                            st.success(f"✅ Found {len(candidates)} candidate locations")
                            st.info(f"🧪 Evaluating top {min(sim_top_k, len(candidates))} candidates across scenarios...")
                            
                            scorer = CompositeScorer(
                                weight_worst_ap=w_worst,
                                weight_average=w_avg,
                                weight_coverage=w_cov,
                                weight_neighborhood=w_neigh,
                                neighborhood_mode=NeighborhoodOptimizationMode.BALANCED,
                                interference_radius_m=sim_interference_radius,
                            )
                            
                            results = []
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            total_sims = min(sim_top_k, len(candidates)) * len(all_scenarios)
                            sim_count = 0
                            
                            for cand_idx, cand_row in candidates.head(sim_top_k).iterrows():
                                scenario_results = []
                                
                                for profile, snap_path, snap_dt in all_scenarios:
                                    sim_count += 1
                                    progress = sim_count / total_sims
                                    progress_bar.progress(progress)
                                    status_text.text(f"Evaluating candidate {cand_idx+1}/{min(sim_top_k, len(candidates))} | "
                                                   f"Scenario {sim_count}/{total_sims} ({profile.value})")
                                    
                                    df_scenario = read_ap_snapshot(snap_path, band_mode='worst')
                                    df_scenario = df_scenario.merge(geo_df, on='name', how='inner')
                                    df_scenario = df_scenario[df_scenario['group_code'] != 'SAB'].copy()
                                    
                                    _, new_ap_stats, metrics = simulate_ap_addition(
                                        df_scenario,
                                        cand_row['lat'],
                                        cand_row['lon'],
                                        config,
                                        scorer,
                                    )
                                    
                                    metrics['stress_profile'] = profile.value
                                    metrics['timestamp'] = snap_dt
                                    scenario_results.append(metrics)
                                
                                aggregated = aggregate_scenario_results(
                                    cand_row['lat'],
                                    cand_row['lon'],
                                    cand_row.get('conflictivity', 0.0),
                                    scenario_results,
                                )
                                
                                results.append(aggregated)
                            
                            progress_bar.empty()
                            status_text.empty()
                            
                            results_df = pd.DataFrame(results)
                            results_df = results_df.sort_values('final_score', ascending=False)
                            
                            if len(results_df) > 0:
                                best = results_df.iloc[0]
                                fig.add_trace(go.Scattermapbox(
                                    lat=[best['lat']],
                                    lon=[best['lon']],
                                    mode='markers',
                                    marker=dict(
                                        size=5,
                                        color='blue',
                                        opacity=0.8
                                    ),
                                    name='🎯 Best Location',
                                    hovertemplate=(
                                        '<b>🎯 Best Placement Location</b><br>'
                                        'Final Score: %{customdata[0]:.3f} ± %{customdata[1]:.3f}<br>'
                                        'Avg Reduction: %{customdata[2]:.3f}<br>'
                                        'Worst AP Improvement: %{customdata[3]:.3f}<br>'
                                        'New AP Clients: %{customdata[4]:.0f}<br>'
                                        'Scenarios Tested: %{customdata[5]:.0f}<br>'
                                        '<extra></extra>'
                                    ),
                                    customdata=np.column_stack([
                                        [best['final_score']],
                                        [best['score_std']],
                                        [best['avg_reduction_raw_mean']],
                                        [best['worst_ap_improvement_raw_mean']],
                                        [best['new_ap_client_count_mean']],
                                        [best['n_scenarios']],
                                    ]),
                                ))
                            
                            for idx, row in results_df.iterrows():
                                rank = idx + 1
                                
                                fig.add_trace(go.Scattermapbox(
                                    lat=[row['lat']],
                                    lon=[row['lon']],
                                    mode='markers+text',
                                    marker=dict(
                                        size=5,
                                        color='purple',
                                        opacity=0.9
                                    ),
                                    text=f"#{rank}",
                                    textposition="top center",
                                    textfont=dict(size=10, color='white', family='Arial Black'),
                                    name=f'Proposed AP #{rank}',
                                    hovertemplate=(
                                        f'<b>Proposed AP #{rank}</b><br>'
                                        'Score: %{customdata[0]:.3f} ± %{customdata[1]:.3f}<br>'
                                        'Avg Reduction: %{customdata[2]:.3f}<br>'
                                        'Worst AP Improvement: %{customdata[3]:.3f}<br>'
                                        'New AP Clients: %{customdata[4]:.0f}<br>'
                                        '<extra></extra>'
                                    ),
                                    customdata=np.column_stack([
                                        [row['final_score']],
                                        [row['score_std']],
                                        [row['avg_reduction_raw_mean']],
                                        [row['worst_ap_improvement_raw_mean']],
                                        [row['new_ap_client_count_mean']],
                                    ]),
                                ))
                            
                            st.divider()
                            st.subheader("📊 Multi-Scenario Simulation Results")
                            
                            display_cols = ['lat', 'lon', 'final_score', 'score_std', 'avg_reduction_raw_mean', 
                                           'worst_ap_improvement_raw_mean', 'num_improved_mean', 'new_ap_client_count_mean', 'n_scenarios']
                            display_df = results_df[display_cols].copy()
                            display_df.columns = ['Latitude', 'Longitude', 'Final Score', 'Std Dev', 'Avg Reduction', 
                                                 'Worst AP Improv', '# Improved APs', 'New AP Clients', 'Scenarios']
                            
                            st.dataframe(
                                display_df.style.format({
                                    'Latitude': '{:.6f}',
                                    'Longitude': '{:.6f}',
                                    'Final Score': '{:.3f}',
                                    'Std Dev': '{:.3f}',
                                    'Avg Reduction': '{:.3f}',
                                    'Worst AP Improv': '{:.3f}',
                                    '# Improved APs': '{:.1f}',
                                    'New AP Clients': '{:.0f}',
                                    'Scenarios': '{:.0f}',
                                }).background_gradient(subset=['Final Score'], cmap='RdYlGn'),
                                use_container_width=True
                            )
                            
                            col1, col2, col3, col4 = st.columns(4)
                            best = results_df.iloc[0]
                            
                            with col1:
                                st.metric("Best Score", f"{best['final_score']:.3f}", 
                                         delta=f"±{best['score_std']:.3f}")
                            with col2:
                                st.metric("Avg Reduction", f"{best['avg_reduction_raw_mean']:.3f}")
                            with col3:
                                st.metric("Worst AP Improvement", f"{best['worst_ap_improvement_raw_mean']:.3f}")
                            with col4:
                                st.metric("New AP Clients", f"{int(best['new_ap_client_count_mean'])}")
                            
                            if best.get('warnings'):
                                st.subheader("⚠️ Placement Warnings")
                                for warning in best['warnings']:
                                    st.warning(warning)
                            else:
                                st.success("✅ No significant warnings for this placement")
                            
                            st.success(f"💡 **Recommendation**: Place new AP at ({best['lat']:.6f}, {best['lon']:.6f}) for maximum network improvement across {best['n_scenarios']:.0f} scenarios")

                            preview_metrics = None
                            try:
                                base_latest = read_ap_snapshot(selected_path, band_mode='worst').merge(geo_df, on='name', how='inner')
                                base_latest = base_latest[base_latest['group_code'] != 'SAB'].copy()
                                if not base_latest.empty:
                                    df_after_preview, _, preview_metrics = simulate_ap_addition(
                                        base_latest,
                                        float(best['lat']),
                                        float(best['lon']),
                                        config,
                                        scorer,
                                    )
                                    st.session_state['map_override_df'] = df_after_preview
                                    st.session_state['new_node_markers'] = [{
                                        'lat': float(best['lat']),
                                        'lon': float(best['lon']),
                                        'label': 'AP-BEST'
                                    }]
                                    st.session_state['map_preview_metrics'] = preview_metrics
                                else:
                                    st.warning("Cannot build simulated map: no UAB APs in the selected snapshot after filtering.")
                            except Exception as e:
                                st.warning(f"Preview map update failed: {e}")

                            if preview_metrics:
                                col_before, col_after, col_delta = st.columns(3)
                                with col_before:
                                    st.metric("Avg conflictivity (current)", f"{preview_metrics['avg_conflictivity_before']:.3f}")
                                with col_after:
                                    st.metric("Avg conflictivity (simulated)", f"{preview_metrics['avg_conflictivity_after']:.3f}")
                                with col_delta:
                                    st.metric("Avg improvement", f"{preview_metrics['avg_reduction']:.3f}", 
                                              delta=f"{preview_metrics['avg_reduction_pct']:.1f}%")
                                st.caption("Metrics computed on the currently selected snapshot to compare the before/after map views.")
                
            except Exception as e:
                st.error(f"❌ Simulation error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        
        st.session_state.run_sim = False

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

    st.plotly_chart(fig, use_container_width=True)
    
    # Render a second map with the simulated surface, if available
    if 'map_override_df' in st.session_state and st.session_state['map_override_df'] is not None:
        sim_df = st.session_state['map_override_df']
        
        # Create the simulated map (similar to the main map but with simulated data)
        tmp_sim = sim_df.copy()
        if "group_code" not in tmp_sim.columns:
            tmp_sim["group_code"] = tmp_sim["name"].apply(extract_group)
        
        sab_df_sim = tmp_sim[tmp_sim["group_code"] == "SAB"].copy()
        uab_df_sim = tmp_sim[tmp_sim["group_code"] != "SAB"].copy()

        fig_sim = go.Figure()

        # UAB interpolation for simulated data
        if not uab_df_sim.empty:
            ch_sim, eff_tile_sim, hull_sim = _uab_tiled_choropleth_layer(
                uab_df_sim, tile_meters=TILE_M_FIXED, radius_m=radius_m, mode="decay",
                value_mode=value_mode, max_tiles=MAX_TILES_NO_LIMIT
            )
            if ch_sim is not None:
                fig_sim.add_trace(ch_sim)
                fig_sim.add_annotation(text=f"UAB tile ≈ {eff_tile_sim:.1f} m (simulated)",
                                     showarrow=False, xref="paper", yref="paper",
                                     x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="#888", font=dict(size=10))
            
            # AP points (excluding new simulated APs)
            existing_aps = uab_df_sim[~uab_df_sim['name'].str.startswith('AP-NEW-SIM')]
            if not existing_aps.empty:
                fig_sim.add_trace(go.Scattermapbox(
                    lat=existing_aps['lat'], lon=existing_aps['lon'], mode='markers',
                    marker=dict(size=7, color='black', opacity=0.7),
                    text=existing_aps['name'], name="UAB APs",
                    hovertemplate='<b>%{text}</b><br>Conflictivity: %{customdata:.2f}<extra></extra>',
                    customdata=existing_aps['conflictivity']
                ))
        
        # Add new AP markers if available
        if 'new_node_markers' in st.session_state and st.session_state['new_node_markers']:
            nn = st.session_state['new_node_markers']
            fig_sim.add_trace(go.Scattermapbox(
                lat=[p['lat'] for p in nn],
                lon=[p['lon'] for p in nn],
                mode='markers+text',
                marker=dict(size=10, color='skyblue', opacity=0.95),
                text=[p.get('label', 'AP-NEW') for p in nn],
                textposition='top center',
                name='New APs (simulated)',
            ))
        
        fig_sim.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=15,
            ),
            margin=dict(l=10, r=10, t=30, b=10),
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02)
        )
        
        st.subheader("Simulated Map (with new APs)")
        st.plotly_chart(fig_sim, use_container_width=True)
    
    # Display Voronoi Candidates table if detected
    if 'voronoi_candidates' in st.session_state and not st.session_state.voronoi_candidates.empty:
        vor_df = st.session_state.voronoi_candidates
        st.divider()
        st.subheader("🧬 Voronoi Candidates")
        st.caption("Click Select checkbox and click Simulate selected to evaluate adding APs at those vertices across scenarios.")
        
        # Selectable editor with checkbox column
        display_vor = vor_df[['lat','lon','avg_conflictivity','max_conflictivity','freq','avg_min_dist_m']].copy()
        display_vor.columns = ['Latitude','Longitude','Avg Conflict','Max Conflict','Freq','Avg Dist (m)']
        display_vor['Idx'] = np.arange(len(display_vor))
        prev_sel = st.session_state.get('vor_selected_rows', set())
        display_vor['Select'] = display_vor['Idx'].apply(lambda i: i in prev_sel)
        
        edited = st.data_editor(
            display_vor,
            use_container_width=True,
            hide_index=True,
            disabled={
                'Latitude': True,
                'Longitude': True,
                'Avg Conflict': True,
                'Max Conflict': True,
                'Freq': True,
                'Avg Dist (m)': True,
                'Idx': True,
            },
            num_rows="fixed",
            key="voronoi_editor",
        )
        current_sel = set(edited.loc[edited['Select'] == True, 'Idx'].astype(int).tolist())
        st.session_state['vor_selected_rows'] = current_sel
        
        col_run, col_results = st.columns([1,4])
        with col_run:
            if st.button("Simulate selected"):
                if not current_sel:
                    st.warning("Select at least one candidate to simulate.")
                else:
                    params = st.session_state.get('sim_params', {})
                    w_worst = params.get('w_worst', 0.30)
                    w_avg = params.get('w_avg', 0.30)
                    w_cov = params.get('w_cov', 0.20)
                    w_neigh = params.get('w_neigh', 0.20)
                    interference_radius = params.get('interference_radius', 50)
                    cca_increase = params.get('cca_increase', 0.15)
                    
                    cfg = SimulationConfig(
                        interference_radius_m=interference_radius,
                        cca_increase_factor=cca_increase,
                        indoor_only=True,
                        conflictivity_threshold_placement=params.get('threshold', 0.6),
                        snapshots_per_profile=params.get('snapshots', 5),
                        target_stress_profile=None,
                        weight_worst_ap=w_worst,
                        weight_average=w_avg,
                        weight_coverage=w_cov,
                        weight_neighborhood=w_neigh,
                    )
                    scorer = CompositeScorer(
                        weight_worst_ap=w_worst,
                        weight_average=w_avg,
                        weight_coverage=w_cov,
                        weight_neighborhood=w_neigh,
                        neighborhood_mode=NeighborhoodOptimizationMode.BALANCED,
                        interference_radius_m=interference_radius,
                    )
                    
                    batch_results = []
                    combined_points = []
                    progress = st.progress(0.0)
                    
                    for b_i, idx in enumerate(sorted(current_sel)):
                        row = vor_df.iloc[idx]
                        single_results = []
                        combined_points.append({'lat': float(row['lat']), 'lon': float(row['lon']), 'label': f"AP-VOR-{idx+1}"})
                        
                        for (profile, snap_path, snap_dt) in st.session_state.get('voronoi_scenarios', []):
                            df_snap = read_ap_snapshot(snap_path, band_mode='worst').merge(geo_df, on='name', how='inner')
                            df_snap = df_snap[df_snap['group_code'] != 'SAB'].copy()
                            if df_snap.empty:
                                continue
                            _, new_ap_stats, metrics = simulate_ap_addition(
                                df_snap,
                                float(row['lat']),
                                float(row['lon']),
                                cfg,
                                scorer,
                            )
                            metrics['stress_profile'] = profile.value
                            metrics['timestamp'] = snap_dt
                            single_results.append(metrics)
                        
                        if single_results:
                            agg = aggregate_scenario_results(float(row['lat']), float(row['lon']), float(row.get('avg_conflictivity', 0.0)), single_results)
                            agg['label'] = f"AP-VOR-{idx+1}"
                            batch_results.append(agg)
                        
                        progress.progress((b_i+1)/max(1,len(current_sel)))
                    
                    progress.empty()
                    
                    if batch_results:
                        res_df = pd.DataFrame(batch_results)
                        res_df = res_df.sort_values('final_score', ascending=False)
                        st.session_state['batch_vor_results'] = res_df
                        
                        # Update main map with combined simulation of selected APs on latest snapshot
                        try:
                            base_latest = read_ap_snapshot(selected_path, band_mode='worst').merge(geo_df, on='name', how='inner')
                            base_latest = base_latest[base_latest['group_code'] != 'SAB'].copy()
                            if not base_latest.empty:
                                cfg_multi = SimulationConfig(
                                    interference_radius_m=interference_radius,
                                    cca_increase_factor=cca_increase,
                                    indoor_only=True,
                                    conflictivity_threshold_placement=params.get('threshold', 0.6),
                                    snapshots_per_profile=params.get('snapshots', 5),
                                    target_stress_profile=None,
                                    weight_worst_ap=w_worst,
                                    weight_average=w_avg,
                                    weight_coverage=w_cov,
                                    weight_neighborhood=w_neigh,
                                )
                                df_after_multi = simulate_multiple_ap_additions(base_latest, combined_points, cfg_multi)
                                st.session_state['map_override_df'] = df_after_multi
                                st.session_state['new_node_markers'] = combined_points
                                st.success("Simulated Map rendered below with sky blue NEW AP markers.")
                                # Trigger a rerun so the map section can render the Simulated Map
                                st.rerun()
                        except Exception as e:
                            st.warning(f"Combined map update failed: {e}")
        
        with col_results:
            if 'batch_vor_results' in st.session_state:
                res_df = st.session_state['batch_vor_results']
                show_cols = [c for c in ['label','lat','lon','final_score','score_std','avg_reduction_raw_mean','worst_ap_improvement_raw_mean','new_ap_client_count_mean','n_scenarios'] if c in res_df.columns]
                st.subheader("📊 Batch Simulation Results")
                st.dataframe(
                    res_df[show_cols].style.format({
                        'lat': '{:.6f}',
                        'lon': '{:.6f}',
                        'final_score': '{:.3f}',
                        'score_std': '{:.3f}',
                        'avg_reduction_raw_mean': '{:.3f}',
                        'worst_ap_improvement_raw_mean': '{:.3f}',
                        'new_ap_client_count_mean': '{:.1f}',
                        'n_scenarios': '{:.0f}',
                    }).background_gradient(subset=['final_score'], cmap='RdYlGn'),
                    use_container_width=True,
                    hide_index=True
                )
        
        st.caption(f"💡 **{len(vor_df)} candidates detected**. Freq = how many scenarios this vertex appeared in. Higher frequency = more stable location.")

# Top conflictive APs table (common to all three modes)
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

if viz_mode == "AI Heatmap":
    st.caption(
        f"📻 Band mode: {band_info[band_mode]}  |  "
        "💡 Conflictivity measures Wi-Fi stress by combining channel congestion (75%), number of connected devices (15%), and AP resource usage (10%)  |  "
        "🟢 Low ↔ 🔴 High (0–1)  |  "
        "👆 Selecciona un AP al mapa per analitzar-lo amb AINA AI"
    )
elif viz_mode == "Voronoi":
    st.caption(
        f"📻 Band mode: {band_info[band_mode]}  |  "
        "💡 Conflictivity ≈ 0.75×airtime + 0.15×clients + 0.05×CPU + 0.05×Memòria  |  "
        "🎨 Escala: 🟢 Low → 🟡 Medium → 🔴 High (0–1)"
    )
else:  # Simulator
    st.caption(
        f"📻 Band mode: {band_info[band_mode]}  |  "
        "💡 Conflictivity ≈ 0.85×airtime + 0.10×clients + 0.02×CPU + 0.03×Memòria  |  "
        "🎨 Escala: 🟢 Low → 🟡 Medium → 🔴 High (0–1)  |  "
        "🎯 Multi-scenario AP placement optimization"
    )
