"""
Conflictivity Dashboard (Interpolated Surfaces) with AP Placement Simulator

Purpose
- Render an interpolated surface of Wi‑Fi conflictivity with time series navigation.
- Perform TWO independent interpolations:
  • UAB campus (all APs except AP-SAB-*), masked to its convex hull.
  • Sabadell site (AP-SAB-*) masked to its convex hull.
- SIMULATE optimal AP placement locations interactively.

Notes
- Interpolation method: KNN (distance-weighted) over a lon/lat grid, masked to the convex hull.
- Fallback: if insufficient points for a hull, show points only for that layer.

Run
  streamlit run dashboard/conflictivity_dashboard_interpolation.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from math import log1p

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import Point, MultiPoint, LineString, Polygon
from shapely.ops import unary_union, linemerge, snap
from matplotlib.path import Path as MplPath
try:
    from scipy.spatial import Voronoi
    _HAS_SCIPY_VORONOI = True
except Exception:  # pragma: no cover
    _HAS_SCIPY_VORONOI = False


# -------- Paths --------
REPO_ROOT = Path(__file__).resolve().parents[1]
AP_DIR = REPO_ROOT / "realData" / "ap"
GEOJSON_PATH = REPO_ROOT / "realData" / "geoloc" / "aps_geolocalizados_wgs84.geojson"

# Add simulator to path
sys.path.insert(0, str(REPO_ROOT))

# Import simulator components
from simulator.config import SimulationConfig, StressLevel
from simulator.stress_profiler import StressProfiler
from simulator.scoring import CompositeScorer, NeighborhoodOptimizationMode
from simulator.spatial import haversine_m, compute_convex_hull_polygon, mask_points_in_polygon


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
    """Find all snapshot files and parse their timestamps. Sorted by time."""
    files = list(ap_dir.glob("AP-info-v2-*.json"))
    files_with_time = []
    for f in files:
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})", f.name)
        if match:
            year, month, day, hour, minute, second = map(int, match.groups())
            dt = datetime(year, month, day, hour, minute, second)
            files_with_time.append((f, dt))
    files_with_time.sort(key=lambda x: x[1])
    return files_with_time


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def airtime_score(util: float, band: str) -> float:
    """Misma funció que l'altre dashboard: mapatge de utilització a [0,1] diferenciat per banda."""
    u = clamp(util or 0.0, 0.0, 100.0)
    if band == "2g":
        if u <= 10: return 0.05 * (u / 10.0)
        if u <= 25: return 0.05 + 0.35 * ((u - 10) / 15.0)
        if u <= 50: return 0.40 + 0.35 * ((u - 25) / 25.0)
        return 0.75 + 0.25 * ((u - 50) / 50.0)
    else:  # 5g
        if u <= 15: return 0.05 * (u / 15.0)
        if u <= 35: return 0.05 + 0.35 * ((u - 15) / 20.0)
        if u <= 65: return 0.40 + 0.35 * ((u - 35) / 30.0)
        return 0.75 + 0.25 * ((u - 65) / 35.0)

def client_pressure_score(n_clients: float, peers_p95: float) -> float:
    n = max(0.0, float(n_clients or 0.0))
    denom = max(1.0, float(peers_p95 or 1.0))
    x = log1p(n) / log1p(denom)
    return clamp(x, 0.0, 1.0)

def cpu_health_score(cpu_pct: float) -> float:
    c = clamp(cpu_pct or 0.0, 0.0, 100.0)
    if c <= 70: return 0.0
    if c <= 90: return 0.6 * ((c - 70) / 20.0)
    return 0.6 + 0.4 * ((c - 90) / 10.0)

def mem_health_score(mem_used_pct: float) -> float:
    m = clamp(mem_used_pct or 0.0, 0.0, 100.0)
    if m <= 80: return 0.0
    if m <= 95: return 0.6 * ((m - 80) / 15.0)
    return 0.6 + 0.4 * ((m - 95) / 5.0)

def read_ap_snapshot(path: Path, band_mode: str = "worst") -> pd.DataFrame:
    """Llegir snapshot i calcular conflictivity avançada coherent amb l'altre dashboard."""
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
            b = r.get("band")
            if u is None: continue
            if b == 0: util_2g.append(float(u))
            elif b == 1: util_5g.append(float(u))
        max_2g = max(util_2g) if util_2g else np.nan
        max_5g = max(util_5g) if util_5g else np.nan
        if band_mode == "2.4GHz":
            agg_util = max_2g
        elif band_mode == "5GHz":
            agg_util = max_5g
        elif band_mode == "avg":
            vals = [x for x in [max_2g, max_5g] if not np.isnan(x)]
            agg_util = float(np.mean(vals)) if vals else np.nan
        else:  # worst
            agg_util = np.nanmax([max_2g, max_5g])
        rows.append({
            "name": name,
            "group_name": group_name,
            "site": site,
            "client_count": client_count,
            "cpu_utilization": cpu_util,
            "mem_free": mem_free,
            "mem_total": mem_total,
            "util_2g": max_2g,
            "util_5g": max_5g,
            "agg_util": agg_util,
        })
    df = pd.DataFrame(rows)
    # Sanitize
    num_cols = ["client_count","cpu_utilization","mem_free","mem_total","util_2g","util_5g","agg_util"]
    for c in num_cols:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce")
    df["mem_used_pct"] = (1 - (df["mem_free"] / df["mem_total"])).clip(0,1) * 100
    df["mem_used_pct"] = df["mem_used_pct"].fillna(0)
    # Airtime scores per band
    df["air_s_2g"] = df["util_2g"].apply(lambda u: airtime_score(u, "2g") if not np.isnan(u) else np.nan)
    df["air_s_5g"] = df["util_5g"].apply(lambda u: airtime_score(u, "5g") if not np.isnan(u) else np.nan)
    w_2g, w_5g = 0.6, 0.4
    if band_mode in ("2.4GHz","5GHz"):
        df["airtime_score"] = np.where(band_mode=="2.4GHz", df["air_s_2g"], df["air_s_5g"])
    elif band_mode == "avg":
        df["airtime_score"] = (
            (df["air_s_2g"].fillna(0)*w_2g + df["air_s_5g"].fillna(0)*w_5g) /
            ((~df["air_s_2g"].isna())*w_2g + (~df["air_s_5g"].isna())*w_5g).replace(0,np.nan)
        )
    else:
        df["airtime_score"] = np.nanmax(np.vstack([df["air_s_2g"].fillna(-1), df["air_s_5g"].fillna(-1)]), axis=0)
        df["airtime_score"] = df["airtime_score"].where(df["airtime_score"]>=0, np.nan)
    # Clients percentile
    p95 = float(np.nanpercentile(df["client_count"].fillna(0),95)) if len(df) else 1.0
    df["client_score"] = df["client_count"].apply(lambda n: client_pressure_score(n,p95))
    df["cpu_score"] = df["cpu_utilization"].apply(cpu_health_score)
    df["mem_score"] = df["mem_used_pct"].apply(mem_health_score)
    # Relief si 0 clients
    def relief(a,c):
        if np.isnan(a): return np.nan
        if (c or 0)>0: return a
        return a*0.8
    df["airtime_score_adj"] = [relief(a,c) for a,c in zip(df["airtime_score"], df["client_count"])]
    # Ponderacions ajustades
    W_AIR, W_CL, W_CPU, W_MEM = 0.85, 0.10, 0.02, 0.03
    df["airtime_score_filled"] = df["airtime_score_adj"].fillna(0.4)
    df["conflictivity"] = (
        df["airtime_score_filled"]*W_AIR + df["client_score"].fillna(0)*W_CL +
        df["cpu_score"].fillna(0)*W_CPU + df["mem_score"].fillna(0)*W_MEM
    ).clip(0,1)
    df["max_radio_util"] = df["agg_util"].fillna(0)
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
        if not props or not geom or geom.get("type") != "Point":
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


# -------- Interpolation utilities --------
def _compute_convex_hull_polygon(lons: np.ndarray, lats: np.ndarray):
    """Return a shapely Polygon of the convex hull, or None if degenerate."""
    return compute_convex_hull_polygon(lons, lats)


def _mask_points_in_polygon(lon_grid: np.ndarray, lat_grid: np.ndarray, polygon) -> np.ndarray:
    """Boolean mask for grid points inside polygon using matplotlib.path.Path."""
    return mask_points_in_polygon(lon_grid, lat_grid, polygon)


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters. Works with numpy arrays via broadcasting."""
    return haversine_m(lat1, lon1, lat2, lon2)


def _interp_kernel(dist_m: np.ndarray, R_m: float, mode: str = "decay"):
    """Kernel by distance in meters within radius R_m."""
    x = np.clip(dist_m / max(R_m, 1e-6), 0.0, 1.0)
    if mode == "grow":
        w = x
    else:
        w = 1.0 - x
    w[dist_m >= R_m] = 0.0
    return w


def _interpolate_conflictivity_kernel(df: pd.DataFrame, grid_size: int = 140, radius_m: float = 25.0,
                                      mode: str = "decay"):
    """Interpolate conflictivity onto a lon/lat grid using a custom distance kernel within radius_m."""
    lons = df["lon"].to_numpy(dtype=float)
    lats = df["lat"].to_numpy(dtype=float)
    cvals = df["conflictivity"].to_numpy(dtype=float)

    hull_poly = _compute_convex_hull_polygon(lons, lats)
    if hull_poly is None:
        return None, None, None

    minx, miny, maxx, maxy = hull_poly.bounds
    lon_grid = np.linspace(minx, maxx, grid_size)
    lat_grid = np.linspace(miny, maxy, grid_size)
    XX, YY = np.meshgrid(lon_grid, lat_grid)

    mask_inside = _mask_points_in_polygon(lon_grid, lat_grid, hull_poly)

    Z = np.full(XX.shape, np.nan, dtype=float)
    if np.any(mask_inside):
        pts_lat = YY[mask_inside]
        pts_lon = XX[mask_inside]
        dists = _haversine_m(pts_lat[:, None], pts_lon[:, None], lats[None, :], lons[None, :])
        d_min = dists.min(axis=1)
        boundary_conf = np.where(d_min >= radius_m, 1.0, d_min / max(radius_m, 1e-6))
        W = _interp_kernel(dists, radius_m, mode=mode)
        denom = W.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            num = (W * cvals[None, :]).sum(axis=1)
            weighted_conf = np.where(denom > 0, num / denom, np.nan)
        vals = np.maximum(weighted_conf, boundary_conf)
        Z[mask_inside] = np.clip(vals, 0.0, 1.0)

    return lon_grid, lat_grid, Z


def _add_density_layer(fig: go.Figure, lon_grid, lat_grid, Z, *, name: str,
                       showscale: bool, colorbar_title: str | None = None,
                       radius: int = 6, opacity: float = 0.9):
    """Add a Densitymapbox layer from a gridded surface."""
    if lon_grid is None or lat_grid is None or Z is None:
        return
    XX, YY = np.meshgrid(lon_grid, lat_grid)
    mask = np.isfinite(Z)
    if not np.any(mask):
        return
    lons = XX[mask].ravel()
    lats = YY[mask].ravel()
    vals = Z[mask].ravel()

    fig.add_trace(go.Densitymapbox(
        lon=lons,
        lat=lats,
        z=vals,
        radius=radius,
        colorscale=[[0.0, 'rgb(0, 255, 0)'], [0.5, 'rgb(255, 255, 0)'], [1.0, 'rgb(255, 0, 0)']],
        zmin=0, zmax=1,
        showscale=showscale,
        opacity=opacity,
        colorbar=dict(title=colorbar_title or name, thickness=15, len=0.7) if showscale else None,
        hoverinfo='skip',
        name=name,
    ))


def _uab_tiled_choropleth_layer(df_uab: pd.DataFrame, *, tile_meters: float = 3.0,
                                radius_m: float = 25.0, mode: str = "decay",
                                max_tiles: int = 40000, colorscale=None):
    """Create a Choroplethmapbox layer of rectangular tiles (Conflictivity only)."""
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
    colorbar_title = "Conflictivity (UAB tiles)"
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


def create_dual_interpolated_map(df: pd.DataFrame, center_lat: float, center_lon: float,
                                 show_uab: bool = True, show_sab: bool = True,
                                 zoom: int = 15,
                                 tile_meters: float = 3.0,
                                 max_tiles: int = 40000,
                                 radius_m: float = 25.0, kernel_mode: str = "decay",
                                 ) -> go.Figure:
    """Create two separate interpolations: one for non-SAB (UAB) and one for SAB APs."""
    tmp = df.copy()
    if "group_code" not in tmp.columns:
        tmp["group_code"] = tmp["name"].apply(extract_group)

    sab_df = tmp[tmp["group_code"] == "SAB"].copy()
    uab_df = tmp[tmp["group_code"] != "SAB"].copy()

    fig = go.Figure()

    if show_uab and not uab_df.empty:
        ch, eff_tile, hull = _uab_tiled_choropleth_layer(
            uab_df, tile_meters=tile_meters, radius_m=radius_m, mode=kernel_mode,
            max_tiles=max_tiles
        )
        if ch is not None:
            fig.add_trace(ch)
            fig.add_annotation(text=f"UAB tile ≈ {eff_tile:.1f} m",
                               showarrow=False, xref="paper", yref="paper",
                               x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="#888", font=dict(size=10))
        else:
            g1_lon, g1_lat, g1_Z = _interpolate_conflictivity_kernel(uab_df, grid_size=160, radius_m=radius_m, mode=kernel_mode)
            cb_title = "Conflictivity (UAB)"
            _add_density_layer(fig, g1_lon, g1_lat, g1_Z, name="UAB surface", showscale=True, colorbar_title=cb_title)
        fig.add_trace(go.Scattermapbox(
            lat=uab_df['lat'], lon=uab_df['lon'], mode='markers',
            marker=dict(size=7, color='black', opacity=0.7),
            text=uab_df['name'], name="UAB APs",
            hovertemplate='<b>%{text}</b><br>Conflictivity: %{customdata:.2f}<extra></extra>',
            customdata=uab_df['conflictivity']
        ))

    if show_sab and not sab_df.empty:
        g2_lon, g2_lat, g2_Z = _interpolate_conflictivity_kernel(sab_df, grid_size=120, radius_m=radius_m, mode=kernel_mode)
        _add_density_layer(fig, g2_lon, g2_lat, g2_Z, name="Sabadell surface", showscale=False)
        fig.add_trace(go.Scattermapbox(
            lat=sab_df['lat'], lon=sab_df['lon'], mode='markers',
            marker=dict(size=7, color='white', opacity=0.9),
            text=sab_df['name'], name="AP-SAB",
            hovertemplate='<b>%{text}</b><br>Conflictivity: %{customdata:.2f}<extra></extra>',
            customdata=sab_df['conflictivity']
        ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02)
    )

    return fig


# -------- SIMULATOR CODE --------

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


def compute_rssi(distance_m: float, config: SimulationConfig) -> float:
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
    config: SimulationConfig,
    mode: str = 'hybrid'
) -> Tuple[pd.DataFrame, Dict]:
    """Simulate client redistribution when a new AP is added."""
    df = df_aps.copy()
    
    # Compute distances and RSSI from new AP to all existing APs
    df['dist_to_new'] = _haversine_m(
        new_ap_lat, new_ap_lon,
        df['lat'].values, df['lon'].values
    )
    
    df['rssi_new'] = df['dist_to_new'].apply(lambda d: compute_rssi(d, config))
    
    # APs within interference radius are candidates for client redistribution
    df['in_range'] = df['dist_to_new'] <= config.interference_radius_m
    
    total_transferred = 0
    
    # Hybrid mode: combine signal strength + load balancing
    # Key fix: Don't filter by conflictivity > 0.5, use distance and signal instead
    candidates = df[df['in_range'] & (df['client_count'] > 0)].copy()
    
    if not candidates.empty:
        # Sort by distance (closer APs are more likely to offload clients)
        candidates = candidates.sort_values('dist_to_new', ascending=True)
        
        for idx, row in candidates.iterrows():
            # Signal strength factor (0-1): how strong is the signal from new AP?
            signal_strength = max(0.0, (row['rssi_new'] - config.min_rssi_dbm) / 20.0)
            signal_strength = min(1.0, signal_strength)
            
            # Distance factor (0-1): closer APs offload more
            distance_factor = 1.0 - (row['dist_to_new'] / config.interference_radius_m)
            distance_factor = max(0.0, min(1.0, distance_factor))
            
            # Conflictivity factor (0-1): high-conflictivity APs offload more
            conflict_factor = float(row.get('conflictivity', 0.5))
            
            # Combined transfer probability
            # Weight: 30% signal + 30% distance + 40% conflictivity
            transfer_potential = (
                0.30 * signal_strength +
                0.30 * distance_factor +
                0.40 * conflict_factor
            )
            
            # Apply maximum offload constraint
            transfer_fraction = min(config.max_offload_fraction, transfer_potential * 0.8)
            
            # Apply sticky client constraint (some clients won't roam)
            transfer_fraction *= (1 - config.sticky_client_fraction)
            
            # Calculate number of clients to transfer
            n_transfer = max(1, int(row['client_count'] * transfer_fraction))
            
            # Don't transfer more clients than the AP has
            n_transfer = min(n_transfer, int(row['client_count']))
            
            # Apply the transfer
            if n_transfer > 0:
                df.at[idx, 'client_count'] = max(0, row['client_count'] - n_transfer)
                total_transferred += n_transfer
    
    # Ensure we have at least some clients if there are nearby APs with clients
    if total_transferred == 0 and not candidates.empty:
        # Force minimum transfer from closest high-load AP
        closest_with_clients = candidates.head(1)
        if not closest_with_clients.empty:
            idx = closest_with_clients.index[0]
            n_transfer = max(1, int(df.at[idx, 'client_count'] * 0.1))  # Take 10% minimum
            df.at[idx, 'client_count'] = max(0, df.at[idx, 'client_count'] - n_transfer)
            total_transferred = n_transfer
    
    new_ap_stats = {
        'lat': new_ap_lat,
        'lon': new_ap_lon,
        'client_count': total_transferred,
        'name': 'AP-NEW-SIM',
        'group_code': 'SIM',
    }
    
    # Estimate new AP utilization based on client load
    client_fraction = min(1.0, total_transferred / config.max_clients_per_ap)
    new_ap_stats['util_2g'] = client_fraction * config.target_util_2g
    new_ap_stats['util_5g'] = client_fraction * config.target_util_5g
    
    return df, new_ap_stats


def apply_cca_interference(
    df_aps: pd.DataFrame,
    new_ap_stats: Dict,
    config: SimulationConfig,
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
    config: SimulationConfig,
    redistribution_mode: str = 'hybrid',
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Simulate adding a new AP at given location."""
    df_updated, new_ap_stats = estimate_client_distribution(
        df_baseline, new_ap_lat, new_ap_lon, config, mode=redistribution_mode
    )
    
    df_updated = apply_cca_interference(df_updated, new_ap_stats, config)
    
    df_updated = recalculate_conflictivity(df_updated)
    
    baseline_conf = df_baseline['conflictivity']
    updated_conf = df_updated['conflictivity']
    
    metrics = {
        'avg_conflictivity_before': float(baseline_conf.mean()),
        'avg_conflictivity_after': float(updated_conf.mean()),
        'avg_reduction': float(baseline_conf.mean() - updated_conf.mean()),
        'avg_reduction_pct': float((baseline_conf.mean() - updated_conf.mean()) / baseline_conf.mean() * 100),
        
        'worst_ap_conflictivity_before': float(baseline_conf.max()),
        'worst_ap_conflictivity_after': float(updated_conf.max()),
        'worst_ap_improvement': float(baseline_conf.max() - updated_conf.max()),
        
        'num_high_conflict_before': int((baseline_conf > 0.7).sum()),
        'num_high_conflict_after': int((updated_conf > 0.7).sum()),
        
        'new_ap_client_count': new_ap_stats['client_count'],
        'new_ap_util_2g': new_ap_stats['util_2g'],
        'new_ap_util_5g': new_ap_stats['util_5g'],
    }
    
    improved_aps = (baseline_conf - updated_conf > 0.05).sum()
    metrics['num_improved_aps'] = int(improved_aps)
    metrics['coverage_improvement'] = float(improved_aps / len(df_baseline))
    
    worst_score = min(1.0, metrics['worst_ap_improvement'] / 0.3)
    avg_score = min(1.0, metrics['avg_reduction'] / 0.1)
    cov_score = metrics['coverage_improvement']
    
    metrics['composite_score'] = (
        0.40 * worst_score +
        0.40 * avg_score +
        0.20 * cov_score
    )
    
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
    """
    Generate candidate locations for new AP placement.
    Only selects tiles that are surrounded by other PAINTED tiles.
    A painted tile is one that appears in the visualization (inside hull with valid conflictivity).
    Additional constraint: require a solid buffer of painted tiles of size `neighbor_radius_tiles`
    around each candidate (i.e., morphological erosion with Chebyshev radius).
    """
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
    
    # Determine which tiles are PAINTED (inside hull)
    x_h, y_h = hull.exterior.coords.xy
    poly_path = MplPath(np.vstack([x_h, y_h]).T)
    inside = poly_path.contains_points(centers)
    centers_in = centers[inside]
    
    if len(centers_in) == 0:
        return pd.DataFrame()
    
    # Compute conflictivity for all tiles inside hull
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
    
    # Create a SET of painted tile coordinates (these are the tiles that appear in the viz)
    # Also create a grid for easy neighbor checking
    painted_tiles = set()
    conf_grid = np.full((len(lat_centers), len(lon_centers)), np.nan)
    coord_to_idx = {}
    
    for i, (lon, lat) in enumerate(centers_in):
        # Find grid indices
        lon_idx = min(range(len(lon_centers)), key=lambda x: abs(lon_centers[x] - lon))
        lat_idx = min(range(len(lat_centers)), key=lambda x: abs(lat_centers[x] - lat))
        
        # Mark as painted tile
        painted_tiles.add((lat_idx, lon_idx))
        conf_grid[lat_idx, lon_idx] = z_pred[i]
        coord_to_idx[(lat_idx, lon_idx)] = i
    
    print(f"DEBUG: Total painted tiles: {len(painted_tiles)}")
    print(f"DEBUG: Grid shape: {conf_grid.shape}")
    
    # Build neighbor offsets for a Chebyshev radius (square ring) of size neighbor_radius_tiles
    # radius=1 -> 8-neighborhood; radius=2 -> 24 neighbors; etc.
    neighbor_offsets = [
        (dy, dx)
        for dy in range(-neighbor_radius_tiles, neighbor_radius_tiles + 1)
        for dx in range(-neighbor_radius_tiles, neighbor_radius_tiles + 1)
        if not (dy == 0 and dx == 0)
    ]
    
    valid_tiles_mask = np.zeros(len(centers_in), dtype=bool)
    boundary_mask = np.zeros(len(centers_in), dtype=bool)
    
    for i, (lon, lat) in enumerate(centers_in):
        # Find grid position
        lon_idx = min(range(len(lon_centers)), key=lambda x: abs(lon_centers[x] - lon))
        lat_idx = min(range(len(lat_centers)), key=lambda x: abs(lat_centers[x] - lat))
        
        # Determine neighbor presence set (8-neighborhood radius=1 always used for boundary test)
        # Boundary if any of the immediate 8 neighbors is missing.
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

        # Interior validity: ALL neighbors within chosen radius must be painted
        all_neighbors_painted = True
        for dlat_idx, dlon_idx in neighbor_offsets:
            neighbor_lat_idx = lat_idx + dlat_idx
            neighbor_lon_idx = lon_idx + dlon_idx
            if (neighbor_lat_idx, neighbor_lon_idx) not in painted_tiles:
                all_neighbors_painted = False
                break
        if all_neighbors_painted and not is_boundary:
            valid_tiles_mask[i] = True
    
    print(f"DEBUG: Interior non-boundary tiles passing mask: {valid_tiles_mask.sum()}")
    
    # Use only the strict interior mask (no separate boundary mask needed)
    # Exclude tiles on/after the inner hull: points with distance to nearest AP >= (radius_m - inner_clearance_m)
    inner_mask = d_min < max(0.0, radius_m - inner_clearance_m)
    final_mask = valid_tiles_mask & inner_mask
    
    # Filter to only truly interior tiles
    centers_in = centers_in[final_mask]
    z_pred = z_pred[final_mask]
    boundary_mask = boundary_mask[final_mask]
    d_min = d_min[final_mask]
    
    print(f"DEBUG: After all filtering: {len(centers_in)} tiles remain (from {valid_tiles_mask.size} total)")
    
    if len(centers_in) == 0:
        return pd.DataFrame()
    
    candidates = pd.DataFrame({
        'lon': centers_in[:, 0],
        'lat': centers_in[:, 1],
        'conflictivity': z_pred,
        'is_boundary': boundary_mask,
        'min_dist_m': d_min,
    })
    
    print(f"DEBUG: Before conflictivity filter: {len(candidates)} candidates")
    
    # Filter by conflictivity threshold
    candidates = candidates[(candidates['conflictivity'] >= conflictivity_threshold) & (~candidates['is_boundary'])].copy()
    
    print(f"DEBUG: After conflictivity filter (>={conflictivity_threshold}): {len(candidates)} candidates")
    
    candidates['reason'] = 'high_conflictivity_interior_non_boundary'
    
    print(f"DEBUG: Final candidates returned: {len(candidates)}")
    
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
    """Generate AP candidate locations using Voronoi vertices across multiple scenarios.

    Workflow per scenario:
      1. Read snapshot, merge geoloc, exclude SAB.
      2. Build Voronoi diagram of AP coordinates (lon/lat as planar approx OK at campus scale).
      3. Keep vertices inside convex hull of AP set.
      4. Interpolate conflictivity at vertex using same kernel (1 - d/R).
      5. Filter by: distance to nearest AP < (radius_m - tile_radius_clearance_m) AND predicted conflictivity >= threshold.
      6. Keep up to max_vertices_per_scenario highest conflictivity vertices.

    Aggregation:
      - Merge close vertices across scenarios (within merge_radius_m) using simple clustering.
      - For each cluster accumulate: frequency (count of scenarios appeared), average & max conflictivity, average gap distance.

    Returns DataFrame with columns:
      lat, lon, avg_conflictivity, max_conflictivity, freq, avg_min_dist_m, scenarios (list)
    """
    if not _HAS_SCIPY_VORONOI:
        st.error("SciPy not available: install scipy to enable Voronoi candidate mode.")
        return pd.DataFrame()

    records = []
    # Precompute radius bound for inner clearance
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
            # Voronoi computation
            vor = Voronoi(pts_xy)
            # Convex hull polygon (Shapely) for inside test
            hull_poly = MultiPoint([Point(xy) for xy in pts_xy]).convex_hull
            # Iterate vertices
            lons = df_snap['lon'].to_numpy()
            lats = df_snap['lat'].to_numpy()
            cvals = df_snap['conflictivity'].to_numpy()
            for vidx, (vx, vy) in enumerate(vor.vertices):
                p = Point(vx, vy)
                if not hull_poly.contains(p):
                    continue
                # Distances to APs
                dists = _haversine_m(vy, vx, lats, lons)
                d_min = float(dists.min())
                if d_min >= effective_max_dist:  # outside allowed interior ring
                    continue
                # Kernel weights
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
        except Exception as e:  # pragma: no cover
            st.warning(f"Voronoi generation error for {snap_path.name}: {e}")
            continue

    if not records:
        return pd.DataFrame()

    df_all = pd.DataFrame(records)
    # Keep top vertices per scenario timestamp if too many
    df_all['scenario_key'] = df_all['scenario_ts'].astype(str)
    df_all = df_all.sort_values(['scenario_key', 'conflictivity'], ascending=[True, False])
    df_all = df_all.groupby('scenario_key').head(max_vertices_per_scenario).reset_index(drop=True)

    # Simple clustering (greedy) by merge_radius_m using haversine distance
    clusters = []  # each: dict with lat, lon, points list
    merge_r = merge_radius_m
    for row in df_all.itertuples():
        placed = False
        for cl in clusters:
            d = _haversine_m(row.lat, row.lon, cl['lat'], cl['lon'])
            if d <= merge_r:
                cl['points'].append(row)
                # Update centroid (mean) for stability
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
    # Rename for compatibility with later code expecting 'conflictivity'
    df_clusters['conflictivity'] = df_clusters['avg_conflictivity']
    # Sort: most frequent, then highest avg conflictivity
    df_clusters = df_clusters.sort_values(['freq', 'avg_conflictivity'], ascending=[False, False]).reset_index(drop=True)
    return df_clusters


def simulate_ap_addition(
    df_baseline: pd.DataFrame,
    new_ap_lat: float,
    new_ap_lon: float,
    config: SimulationConfig,
    scorer: CompositeScorer,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Simulate adding a new AP at given location."""
    df_updated, new_ap_stats = estimate_client_distribution(
        df_baseline, new_ap_lat, new_ap_lon, config, mode='hybrid'
    )
    
    df_updated = apply_cca_interference(df_updated, new_ap_stats, config)
    
    df_updated = recalculate_conflictivity(df_updated)
    
    # Compute neighbor mask
    distances = _haversine_m(
        new_ap_lat, new_ap_lon,
        df_updated['lat'].values, df_updated['lon'].values
    )
    neighbor_mask = distances <= config.interference_radius_m
    
    # Compute scores
    baseline_conf = df_baseline['conflictivity'].values
    updated_conf = df_updated['conflictivity'].values
    
    component_scores = scorer.compute_component_scores(
        baseline_conf,
        updated_conf,
        neighbor_mask,
    )
    
    composite_score = scorer.compute_composite_score(component_scores)
    
    # Generate warnings
    warnings = scorer.generate_warnings(component_scores)
    
    # Build metrics
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
    
    # Extract composite scores
    scores = [r['composite_score'] for r in scenario_results]
    aggregated['final_score'] = float(np.mean(scores))
    aggregated['score_std'] = float(np.std(scores))
    aggregated['score_min'] = float(np.min(scores))
    aggregated['score_max'] = float(np.max(scores))
    
    # Aggregate other metrics
    for key in ['worst_ap_improvement_raw', 'avg_reduction_raw', 'num_improved', 'new_ap_client_count']:
        values = [r.get(key, 0) for r in scenario_results]
        aggregated[f'{key}_mean'] = float(np.mean(values))
        aggregated[f'{key}_std'] = float(np.std(values))
    
    # Collect warnings
    all_warnings = []
    for r in scenario_results:
        all_warnings.extend(r.get('warnings', []))
    
    # Deduplicate and count
    from collections import Counter
    warning_counts = Counter(all_warnings)
    aggregated['warnings'] = [
        f"{msg} (in {count}/{len(scenario_results)} scenarios)"
        for msg, count in warning_counts.most_common()
    ]
    
    # Per-profile breakdown
    by_profile = {}
    for r in scenario_results:
        profile = r['stress_profile']
        if profile not in by_profile:
            by_profile[profile] = []
        by_profile[profile].append(r['composite_score'])
    
    for profile, profile_scores in by_profile.items():
        aggregated[f'score_{profile}'] = float(np.mean(profile_scores))
    
    return aggregated


def simulate_multiple_ap_additions(
    df_baseline: pd.DataFrame,
    points: List[Dict],
    config: SimulationConfig,
) -> pd.DataFrame:
    """Approximate combined effect of adding multiple APs by applying them sequentially.

    For each point {lat, lon}, we redistribute clients, apply CCA interference, append a synthetic
    new AP with estimated utilization, and recalculate conflictivity.
    Returns the updated DataFrame including synthetic AP rows (group_code='SIM').
    """
    df_curr = df_baseline.copy()
    for i, p in enumerate(points, start=1):
        lat = float(p['lat'])
        lon = float(p['lon'])
        # Client redistribution
        df_curr, new_ap_stats = estimate_client_distribution(df_curr, lat, lon, config, mode='hybrid')
        # Apply interference to neighbors
        df_curr = apply_cca_interference(df_curr, new_ap_stats, config)
        # Append synthetic AP
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
        # Recompute conflictivity after each addition
        df_curr = recalculate_conflictivity(df_curr)
    return df_curr


# -------- UI --------
st.set_page_config(page_title="UAB Wi‑Fi Conflictivity & Simulator", page_icon="📶", layout="wide")
st.title("UAB Wi‑Fi Conflictivity — Interpolated Surfaces & AP Placement Simulator")
st.caption("Time series visualization • Interpolated conflictivity surfaces • Interactive AP placement simulation")

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

# Sidebar: visualization and simulator controls
with st.sidebar:
    st.header("📊 Dataset Information")
    
    if len(snapshots) > 0:
        first_dt = snapshots[0][1]
        last_dt = snapshots[-1][1]
        st.info(f"**{len(snapshots)} snapshots**\n\n"
                f"📅 {first_dt.strftime('%Y-%m-%d')} to {last_dt.strftime('%Y-%m-%d')}\n\n"
                f"⏰ {first_dt.strftime('%H:%M')} to {last_dt.strftime('%H:%M')}")
        
        # Use the latest snapshot for visualization
        selected_path, selected_dt = snapshots[-1]
        st.caption(f"Showing latest: {selected_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    st.divider()
    st.header("Visualization Settings")

    # Connectivity mode removed; dashboard now always shows conflictivity interpolation.
    radius_m = st.slider("Radi de connectivitat (m)", 5, 60, 25, step=5,
                         help="Distància màxima perquè la connectivitat arribi a 1 (o decaigui a 0 en mode conflictivitat).")
    band_mode = st.radio(
        "Band Mode",
        options=["worst", "avg", "2.4GHz", "5GHz"],
        index=0,
        help="worst: max(max_2.4, max_5) • avg: weighted average of band maxima",
        horizontal=True,
    )
    show_uab = st.checkbox("Mostrar UAB (no SAB)", value=True)
    show_sab = st.checkbox("Mostrar Sabadell (AP-SAB)", value=True)
    # Quick control to reset any simulated map override
    if st.session_state.get('map_override_df') is not None or st.session_state.get('new_node_markers'):
        if st.button("Reset simulated map", help="Clear simulated results and return to baseline map"):
            st.session_state.pop('map_override_df', None)
            st.session_state.pop('new_node_markers', None)
            st.rerun()
    
    TILE_M_FIXED = 7.0
    MAX_TILES_NO_LIMIT = 1_000_000_000
    
    st.divider()
    st.header("🎯 AP Placement Simulator")
    
    run_simulation = st.checkbox("Enable AP Placement Simulation", value=False,
                                 help="Find optimal locations for new APs using multi-scenario analysis")
    
    if run_simulation:
        st.subheader("Simulation Parameters")
        
        # Simplified UI with presets
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
        
        with col_basic2:
            sim_threshold = st.slider("Min conflictivity threshold", 0.4, 0.8, 0.6, 0.05,
                                      help="Only consider areas with high network stress")
            
            sim_snapshots_per_profile = st.slider(
                "Test scenarios", 3, 10, 5,
                help="More scenarios = more confidence, but slower"
            )
        
        # Candidate placement filters
        with st.expander("🎯 Candidate Filters", expanded=False):
            st.caption("Control how far candidates must be from edges")
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
            st.caption("**Experimental Mode**: Voronoi-based candidates")
            use_voronoi_candidates = st.toggle(
                "Use Voronoi vertex candidates (experimental)",
                value=False,
                help="Instead of interior tiles, aggregate Voronoi vertices across selected scenarios where conflictivity is high."
            )
        
        # Advanced options (collapsed by default)
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
                w_avg = st.number_input("Average", 0.0, 1.0, 0.05, 0.05, 
                                       help="Overall network improvement")
                w_cov = st.number_input("Coverage", 0.0, 1.0, 0.05, 0.05, 
                                       help="# of APs improved")
                w_neigh = st.number_input("Neighborhood", 0.0, 1.0, 0.60, 0.05, 
                                         help="Protect nearby APs")
            
            # Validate weights
            total_weight = w_worst + w_avg + w_cov + w_neigh
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"⚠️ Weights must sum to 1.0 (current: {total_weight:.2f})")
        
        # Use defaults if expander not opened
        if 'sim_interference_radius' not in locals():
            sim_interference_radius = 50
            sim_cca_increase = 0.15
            w_worst, w_avg, w_cov, w_neigh = 0.30, 0.05, 0.05, 0.60
            total_weight = w_worst + w_avg + w_cov + w_neigh
            # Defaults for candidate filters if expander not opened
            sim_interior_buffer_tiles = 2
            sim_inner_clearance_m = 10
        
        # Map stress profile to enum
        stress_display_map = {
            "HIGH (Peak hours)": "HIGH",
            "CRITICAL (Overloaded)": "CRITICAL",
            "ALL (Robust)": "ALL"
        }
        sim_stress_profile_key = stress_display_map[sim_stress_profile]
        
        if st.button("🚀 Run Multi-Scenario Simulation", type="primary", disabled=abs(total_weight - 1.0) > 0.01):
            st.session_state.run_sim = True
            st.session_state.sim_params = {
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
                'interior_buffer_tiles': sim_interior_buffer_tiles,
                'inner_clearance_m': sim_inner_clearance_m,
                    'use_voronoi_candidates': 'use_voronoi_candidates' in locals() and use_voronoi_candidates,
            }
        else:
            if 'run_sim' not in st.session_state:
                st.session_state.run_sim = False

        # Voronoi Candidate Discovery (Step 1 of new workflow)
        st.divider()
        st.subheader("🧩 Voronoi Candidate Discovery")
        st.caption("Detect stable high-conflictivity Voronoi vertex clusters across representative scenarios before full simulation.")
        detect_voronoi = st.button("🔍 Detect Voronoi Vertices", disabled=not _HAS_SCIPY_VORONOI)
        if detect_voronoi:
            # Build scenarios similar to simulation preparation
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
                utilization_threshold_critical=85,  # use defaults already in class but explicit for clarity
                utilization_threshold_high=70,
            )
            stress_profiles = profiler.classify_snapshots()
            if target_stress is None:
                profiles_to_test = [StressLevel.LOW, StressLevel.MEDIUM, StressLevel.HIGH, StressLevel.CRITICAL]
            else:
                profiles_to_test = [target_stress]
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
                    merge_radius_m=8.0,
                    max_vertices_per_scenario=60,
                )
                st.session_state.voronoi_candidates = vor_df
                if vor_df.empty:
                    st.warning("No Voronoi candidates detected. Try lowering conflictivity threshold or clearance.")
                else:
                    st.success(f"Detected {len(vor_df)} Voronoi candidate clusters.")

# Load data for selected timestamp
ap_df = read_ap_snapshot(selected_path, band_mode=band_mode)

# Merge AP + geoloc
merged = ap_df.merge(geo_df, on="name", how="inner")

if merged.empty:
    st.info("No APs have geolocation data.")
    st.stop()

# Optional group filter
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

map_df = merged.copy()

center_lat = float(map_df["lat"].mean())
center_lon = float(map_df["lon"].mean())

# Create dual interpolated map (use override if present)
fig = create_dual_interpolated_map(
    df=map_df,
    center_lat=center_lat,
    center_lon=center_lon,
    show_uab=show_uab,
    show_sab=show_sab,
    zoom=15,
    tile_meters=TILE_M_FIXED,
    max_tiles=MAX_TILES_NO_LIMIT,
    radius_m=radius_m,
)

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

    # Show the map above the candidates table
    st.plotly_chart(fig, use_container_width=True)
    # Render a second map with the simulated surface, if available
    if 'map_override_df' in st.session_state and st.session_state['map_override_df'] is not None:
        sim_df = st.session_state['map_override_df']
        fig_sim = create_dual_interpolated_map(
            df=sim_df,
            center_lat=center_lat,
            center_lon=center_lon,
            show_uab=show_uab,
            show_sab=show_sab,
            zoom=15,
            tile_meters=TILE_M_FIXED,
            max_tiles=MAX_TILES_NO_LIMIT,
            radius_m=radius_m,
        )
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
        st.subheader("Simulated Map")
        st.plotly_chart(fig_sim, use_container_width=True)
    map_rendered = True

    # Display candidate table with a central preview area above it
    st.divider()
    st.subheader("🧬 Voronoi Candidates")
    st.caption("Click Simulate to evaluate adding an AP at that vertex across scenarios and preview map impact.")
    sim_preview_container = st.container()
    with sim_preview_container:
        # If a simulation was requested, render the BEFORE/AFTER maps here
        req = st.session_state.get('sim_preview_request')
        if req:
            try:
                # Use current params or defaults
                params = st.session_state.get('sim_params', {})
                w_worst = params.get('w_worst', 0.30)
                w_avg = params.get('w_avg', 0.05)
                w_cov = params.get('w_cov', 0.05)
                w_neigh = params.get('w_neigh', 0.60)
                interference_radius = params.get('interference_radius', 50)
                cca_increase = params.get('cca_increase', 0.15)
                # Build scorer & config
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
                # Scenarios must exist
                if 'voronoi_scenarios' not in st.session_state:
                    st.error("No scenarios loaded. Detect Voronoi vertices first.")
                else:
                    single_results = []
                    progress = st.progress(0.0)
                    total = len(st.session_state.voronoi_scenarios)
                    for i_s, (profile, snap_path, snap_dt) in enumerate(st.session_state.voronoi_scenarios, start=1):
                        df_snap = read_ap_snapshot(snap_path, band_mode='worst').merge(geo_df, on='name', how='inner')
                        df_snap = df_snap[df_snap['group_code'] != 'SAB'].copy()
                        if df_snap.empty:
                            continue
                        _, new_ap_stats, metrics = simulate_ap_addition(
                            df_snap,
                            req['lat'],
                            req['lon'],
                            cfg,
                            scorer,
                        )
                        metrics['stress_profile'] = profile.value
                        metrics['timestamp'] = snap_dt
                        single_results.append(metrics)
                        progress.progress(i_s/total)
                    progress.empty()
                    if not single_results:
                        st.warning("No simulation results produced.")
                    else:
                        agg = aggregate_scenario_results(req['lat'], req['lon'], req.get('base_conf', 0.0), single_results)
                        st.success(f"Simulated AP-VOR-{req['idx']+1}: Final Score {agg['final_score']:.3f} ± {agg['score_std']:.3f}")
                        # BEFORE / AFTER maps for latest selected snapshot
                        df_prev_base = read_ap_snapshot(selected_path, band_mode='worst')
                        df_prev_base = df_prev_base.merge(geo_df, on='name', how='inner')
                        df_prev_base = df_prev_base[df_prev_base['group_code'] != 'SAB'].copy()
                        fig_before = create_dual_interpolated_map(
                            df=df_prev_base,
                            center_lat=center_lat,
                            center_lon=center_lon,
                            show_uab=True,
                            show_sab=False,
                            zoom=15,
                            tile_meters=TILE_M_FIXED,
                            max_tiles=MAX_TILES_NO_LIMIT,
                            radius_m=radius_m,
                        )
                        df_prev_after, new_ap_stats_tmp, _ = simulate_ap_addition(
                            df_prev_base,
                            req['lat'],
                            req['lon'],
                            cfg,
                            scorer,
                        )
                        fig_after = create_dual_interpolated_map(
                            df=df_prev_after,
                            center_lat=center_lat,
                            center_lon=center_lon,
                            show_uab=True,
                            show_sab=False,
                            zoom=15,
                            tile_meters=TILE_M_FIXED,
                            max_tiles=MAX_TILES_NO_LIMIT,
                            radius_m=radius_m,
                        )
                        fig_after.add_trace(go.Scattermapbox(
                            lat=[req['lat']],
                            lon=[req['lon']],
                            mode='markers',
                            marker=dict(size=11, color='cyan', opacity=0.9),
                            name=f"AP-VOR-{req['idx']+1} (Simulated)",
                            hovertemplate='<b>%{text}</b><extra></extra>',
                            text=[f"AP-VOR-{req['idx']+1}"]
                        ))
                        col_before, col_after = st.columns(2)
                        with col_before:
                            st.caption("Before (latest snapshot)")
                            st.plotly_chart(fig_before, use_container_width=True)
                        with col_after:
                            st.caption("After adding AP (latest snapshot)")
                            st.plotly_chart(fig_after, use_container_width=True)
                        # Clear/keep controls
                        colc1, colc2 = st.columns([1,1])
                        with colc1:
                            if st.button("Clear preview"):
                                st.session_state.pop('sim_preview_request', None)
                        with colc2:
                            st.caption("Tip: adjust weights or filters and re-run Simulate.")
            except Exception as e:
                st.warning(f"Preview rendering error: {e}")
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
                w_avg = params.get('w_avg', 0.05)
                w_cov = params.get('w_cov', 0.05)
                w_neigh = params.get('w_neigh', 0.60)
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
                            # Trigger a second rerun so the top section can render the Simulated Map
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
                    'lat':'{:.6f}','lon':'{:.6f}','final_score':'{:.3f}','score_std':'{:.3f}','avg_reduction_raw_mean':'{:.3f}','worst_ap_improvement_raw_mean':'{:.3f}','new_ap_client_count_mean':'{:.0f}','n_scenarios':'{:.0f}'
                }),
                use_container_width=True
            )

# Run simulation if enabled
if run_simulation and st.session_state.get('run_sim', False):
    # Get stored parameters
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
    sim_interior_buffer_tiles = params.get('interior_buffer_tiles', 2)
    sim_inner_clearance_m = params.get('inner_clearance_m', 10)
    use_voronoi_candidates = params.get('use_voronoi_candidates', False)
    
    with st.spinner("🔍 Running multi-scenario AP placement simulation..."):
        try:
            # Create configuration
            # Map stress profile string to enum
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
            
            # Initialize profiler
            profiler = StressProfiler(
                snapshots,
                utilization_threshold_critical=config.utilization_threshold_critical,
                utilization_threshold_high=config.utilization_threshold_high,
            )
            
            # Classify snapshots
            st.info("📊 Classifying snapshots by stress level...")
            stress_profiles = profiler.classify_snapshots()
            
            # Show stress profile distribution
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
            
            # Get scenarios to test
            if target_stress is None:
                # Test all profiles
                profiles_to_test = [StressLevel.LOW, StressLevel.MEDIUM, StressLevel.HIGH, StressLevel.CRITICAL]
            else:
                profiles_to_test = [target_stress]
            
            all_scenarios = []
            for profile in profiles_to_test:
                snaps = profiler.get_representative_snapshots(profile, n_samples=sim_snapshots_per_profile)
                for path, dt in snaps:
                    all_scenarios.append((profile, path, dt))
            
            if not all_scenarios:
                st.error(f"❌ No snapshots found for stress profile: {sim_stress_profile}")
                st.session_state.run_sim = False
                st.stop()
            
            st.success(f"✅ Testing {len(all_scenarios)} scenarios across {len(profiles_to_test)} stress profile(s)")
            
            # Generate candidates from first snapshot
            first_path = all_scenarios[0][1]
            df_first = read_ap_snapshot(first_path, band_mode='worst')
            df_first = df_first.merge(geo_df, on='name', how='inner')
            df_first = df_first[df_first['group_code'] != 'SAB'].copy()
            
            if df_first.empty:
                st.warning("⚠️ No UAB APs available for simulation")
                st.session_state.run_sim = False
                st.stop()
            
            st.info(f"📍 Generating candidate locations (tile_size={TILE_M_FIXED}m, threshold={sim_threshold})...")
            
            if use_voronoi_candidates:
                # Build scenario list from all_scenarios for Voronoi aggregation
                candidates = generate_voronoi_candidates(
                    all_scenarios,
                    geo_df=geo_df,
                    radius_m=radius_m,
                    conflictivity_threshold=sim_threshold,
                    tile_radius_clearance_m=sim_inner_clearance_m,
                    merge_radius_m=8.0,
                    max_vertices_per_scenario=60,
                )
                if not candidates.empty:
                    st.info(f"🧩 Voronoi mode: grouped {len(candidates)} vertex clusters (sorted by frequency,avg conflictivity)")
            else:
                candidates = generate_candidate_locations(
                    df_first,
                    tile_meters=TILE_M_FIXED,
                    conflictivity_threshold=sim_threshold,
                    radius_m=radius_m,
                    indoor_only=config.indoor_only,
                    neighbor_radius_tiles=sim_interior_buffer_tiles,
                    inner_clearance_m=sim_inner_clearance_m,
                )
            
            if candidates.empty:
                st.warning(f"⚠️ No candidates found with conflictivity > {sim_threshold}")
                st.session_state.run_sim = False
                st.stop()
            
            st.success(f"✅ Found {len(candidates)} candidate locations")
            st.info(f"🧪 Evaluating top {min(sim_top_k, len(candidates))} candidates across scenarios...")
            
            # Initialize scorer
            scorer = CompositeScorer(
                weight_worst_ap=w_worst,
                weight_average=w_avg,
                weight_coverage=w_cov,
                weight_neighborhood=w_neigh,
                neighborhood_mode=NeighborhoodOptimizationMode.BALANCED,
                interference_radius_m=sim_interference_radius,
            )
            
            # Evaluate each candidate across all scenarios
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
                    
                    # Load scenario snapshot
                    df_scenario = read_ap_snapshot(snap_path, band_mode='worst')
                    df_scenario = df_scenario.merge(geo_df, on='name', how='inner')
                    df_scenario = df_scenario[df_scenario['group_code'] != 'SAB'].copy()
                    
                    # Simulate
                    _, new_ap_stats, metrics = simulate_ap_addition(
                        df_scenario,
                        cand_row['lat'],
                        cand_row['lon'],
                        config,
                        scorer,
                    )
                    
                    # Debug: Log client transfer info for first scenario
                    if sim_count == 1:
                        st.info(f"🔍 Debug: First scenario analysis\n"
                               f"- New AP will serve **{new_ap_stats['client_count']} clients**\n"
                               f"- Avg conflictivity reduction: **{metrics['avg_reduction']:.3f}**\n"
                               f"- Worst AP improvement: **{metrics['worst_ap_improvement']:.3f}**\n"
                               f"- APs within {sim_interference_radius}m: "
                               f"**{int((df_scenario['dist_to_new'] <= sim_interference_radius).sum() if 'dist_to_new' in df_scenario.columns else 0)}**")
                    
                    metrics['stress_profile'] = profile.value
                    metrics['timestamp'] = snap_dt
                    scenario_results.append(metrics)
                
                # Aggregate across scenarios
                aggregated = aggregate_scenario_results(
                    cand_row['lat'],
                    cand_row['lon'],
                    cand_row['conflictivity'],
                    scenario_results,
                )
                
                results.append(aggregated)
            
            progress_bar.empty()
            status_text.empty()
            
            # Sort by final score
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('final_score', ascending=False)
            
            # Add blue dot for the best candidate (smaller 5px)
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
            
            # Add 5px purple points for all candidates (2nd, 3rd, nth)
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
            
            # Display results
            st.divider()
            st.subheader("📊 Multi-Scenario Simulation Results")
            
            # Display table
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
            
            # Summary metrics
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
            
            # Per-profile breakdown
            if 'score_high' in best or 'score_critical' in best:
                st.subheader("📈 Score by Stress Profile")
                profile_cols = st.columns(4)
                for i, level in enumerate([StressLevel.LOW, StressLevel.MEDIUM, StressLevel.HIGH, StressLevel.CRITICAL]):
                    key = f'score_{level.value}'
                    if key in best:
                        with profile_cols[i]:
                            st.metric(level.value.upper(), f"{best[key]:.3f}")
            
            # Warnings
            if best.get('warnings'):
                st.subheader("⚠️ Placement Warnings")
                for warning in best['warnings']:
                    st.warning(warning)
            else:
                st.success("✅ No significant warnings for this placement")
            
            st.success(f"💡 **Recommendation**: Place new AP at ({best['lat']:.6f}, {best['lon']:.6f}) for maximum network improvement across {best['n_scenarios']:.0f} scenarios")
            
        except Exception as e:
            st.error(f"❌ Simulation error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Reset simulation flag
    st.session_state.run_sim = False

if 'map_rendered' not in locals():
    st.plotly_chart(fig, use_container_width=True)
    # Render a second map with the simulated surface, if available
    if 'map_override_df' in st.session_state and st.session_state['map_override_df'] is not None:
        sim_df = st.session_state['map_override_df']
        fig_sim = create_dual_interpolated_map(
            df=sim_df,
            center_lat=center_lat,
            center_lon=center_lon,
            show_uab=show_uab,
            show_sab=show_sab,
            zoom=15,
            tile_meters=TILE_M_FIXED,
            max_tiles=MAX_TILES_NO_LIMIT,
            radius_m=radius_m,
        )
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
        st.subheader("Simulated Map")
        st.plotly_chart(fig_sim, use_container_width=True)

# Footer
st.caption(
    "📻 Band mode aplicat (worst/avg/2.4/5) • "
    "💡 Conflictivity ≈ 0.85×airtime + 0.10×clients + 0.02×CPU + 0.03×Memòria  |  "
    "🎨 Escala: 🟢 Low → 🟡 Medium → 🔴 High (0–1)"
)