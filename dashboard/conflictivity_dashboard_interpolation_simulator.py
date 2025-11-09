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


# -------- Paths --------
REPO_ROOT = Path(__file__).resolve().parents[1]
AP_DIR = REPO_ROOT / "realData" / "ap"
GEOJSON_PATH = REPO_ROOT / "realData" / "geoloc" / "aps_geolocalizados_wgs84.geojson"


# -------- Simulator Configuration --------
class StressLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SimulationConfig:
    """Simulation parameters"""
    path_loss_exponent: float = 3.5
    reference_distance_m: float = 1.0
    reference_rssi_dbm: float = -30.0
    min_rssi_dbm: float = -75.0
    handover_margin_db: float = 3.0
    interference_radius_m: float = 50.0
    cca_increase_factor: float = 0.15
    max_offload_fraction: float = 0.5
    sticky_client_fraction: float = 0.3
    max_clients_per_ap: int = 50
    target_util_2g: float = 40.0
    target_util_5g: float = 50.0
    indoor_only: bool = True
    min_improvement_threshold: float = 0.05


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
    """Great-circle distance in meters. Works with numpy arrays via broadcasting."""
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

    colorbar_title = "Connectivity (UAB tiles)" if value_mode == "connectivity" else "Conflictivity (UAB tiles)"
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
                                 value_mode: str = "conflictivity") -> go.Figure:
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
            value_mode=value_mode, max_tiles=max_tiles
        )
        if ch is not None:
            fig.add_trace(ch)
            fig.add_annotation(text=f"UAB tile ≈ {eff_tile:.1f} m",
                               showarrow=False, xref="paper", yref="paper",
                               x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="#888", font=dict(size=10))
        else:
            g1_lon, g1_lat, g1_Z = _interpolate_conflictivity_kernel(uab_df, grid_size=160, radius_m=radius_m, mode=kernel_mode)
            cb_title = "Connectivity (UAB)" if value_mode == "connectivity" else "Conflictivity (UAB)"
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
    
    df['dist_to_new'] = _haversine_m(
        new_ap_lat, new_ap_lon,
        df['lat'].values, df['lon'].values
    )
    
    df['rssi_new'] = df['dist_to_new'].apply(lambda d: compute_rssi(d, config))
    
    df['in_range'] = (df['rssi_new'] >= config.min_rssi_dbm) & \
                    (df['dist_to_new'] <= config.interference_radius_m)
    
    total_transferred = 0
    
    if mode == 'hybrid':
        candidates = df[df['in_range'] & (df['conflictivity'] > 0.5)].copy()
        candidates = candidates.sort_values('conflictivity', ascending=False)
        
        for idx, row in candidates.iterrows():
            rssi_factor = (row['rssi_new'] - config.min_rssi_dbm) / 20.0
            conflict_factor = row['conflictivity']
            
            transfer_fraction = min(
                config.max_offload_fraction,
                0.3 * rssi_factor + 0.4 * conflict_factor
            )
            transfer_fraction *= (1 - config.sticky_client_fraction)
            
            n_transfer = int(row['client_count'] * transfer_fraction)
            df.at[idx, 'client_count'] -= n_transfer
            total_transferred += n_transfer
    
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
    
    R_m = radius_m
    W = np.maximum(0, 1 - dists / R_m)
    W[dists >= R_m] = 0
    
    cvals = df_aps['conflictivity'].values
    denom = W.sum(axis=1)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        num = (W * cvals[None, :]).sum(axis=1)
        z_pred = np.where(denom > 0, num / denom, 0.0)
    
    d_min = dists.min(axis=1)
    boundary_conf = np.clip(d_min / R_m, 0.0, 1.0)
    z_pred = np.maximum(z_pred, boundary_conf)
    
    candidates = pd.DataFrame({
        'lon': centers_in[:, 0],
        'lat': centers_in[:, 1],
        'conflictivity': z_pred,
    })
    
    candidates = candidates[candidates['conflictivity'] >= conflictivity_threshold].copy()
    
    if indoor_only:
        def is_indoor(row):
            d = _haversine_m(
                row['lat'], row['lon'],
                lats, lons
            ).min()
            return d <= 20.0
        
        candidates['indoor'] = candidates.apply(is_indoor, axis=1)
        candidates = candidates[candidates['indoor']].copy()
        candidates.drop(columns=['indoor'], inplace=True)
    
    candidates['reason'] = 'high_conflictivity'
    
    return candidates.reset_index(drop=True)


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

# Sidebar: time slider and controls
with st.sidebar:
    st.header("Time Navigation")

    if len(snapshots) > 0:
        default_idx = len(snapshots) - 1
        selected_idx = st.slider(
            "Select Time",
            min_value=0,
            max_value=len(snapshots) - 1,
            value=default_idx,
            format="",
            help="Slide to navigate through time series data"
        )
        selected_path, selected_dt = snapshots[selected_idx]
        st.info(f"📅 **{selected_dt.strftime('%Y-%m-%d')}**\n\n⏰ **{selected_dt.strftime('%H:%M:%S')}**")
        first_dt = snapshots[0][1]
        last_dt = snapshots[-1][1]
        st.caption(f"Available data: {first_dt.strftime('%Y-%m-%d %H:%M')} to {last_dt.strftime('%Y-%m-%d %H:%M')}")
        st.caption(f"Total snapshots: {len(snapshots)}")

    st.divider()
    st.header("Visualization Settings")

    value_mode = st.selectbox("Mode de valor", ["conflictivity", "connectivity"], index=0,
                              help="conflictivity: ponderació dels APs; connectivity: creix fins a 1 al radi indicat")
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
    
    TILE_M_FIXED = 7.0
    MAX_TILES_NO_LIMIT = 1_000_000_000
    
    st.divider()
    st.header("🎯 AP Placement Simulator")
    
    run_simulation = st.checkbox("Enable AP Placement Simulation", value=False,
                                 help="Find optimal locations for new APs")
    
    if run_simulation:
        st.subheader("Simulation Parameters")
        
        sim_top_k = st.slider("Number of candidates", 1, 10, 3, 
                              help="How many top placement candidates to show")
        
        sim_threshold = st.slider("Min conflictivity threshold", 0.0, 1.0, 0.6, 0.05,
                                  help="Only consider locations with conflictivity above this value")
        
        sim_redistribution = st.selectbox("Client redistribution mode", 
                                          ["hybrid", "signal", "load_balance"],
                                          index=0,
                                          help="How clients move to the new AP")
        
        sim_interference_radius = st.slider("Interference radius (m)", 20, 100, 50, 5,
                                           help="Range of co-channel interference from new AP")
        
        sim_cca_increase = st.slider("CCA increase factor", 0.0, 0.5, 0.15, 0.05,
                                     help="How much new AP increases neighbor utilization")
        
        if st.button("🚀 Run Simulation", type="primary"):
            st.session_state.run_sim = True
        else:
            if 'run_sim' not in st.session_state:
                st.session_state.run_sim = False

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

# Create dual interpolated map
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
    value_mode=value_mode,
)

# Run simulation if enabled
if run_simulation and st.session_state.get('run_sim', False):
    with st.spinner("🔍 Running AP placement simulation..."):
        # Filter to UAB only
        sim_df = map_df[map_df["group_code"] != "SAB"].copy()
        
        if sim_df.empty:
            st.warning("No UAB APs available for simulation")
        else:
            # Create config
            config = SimulationConfig(
                interference_radius_m=sim_interference_radius,
                cca_increase_factor=sim_cca_increase,
                indoor_only=True,
            )
            
            # Generate candidates
            candidates = generate_candidate_locations(
                sim_df,
                tile_meters=TILE_M_FIXED,
                conflictivity_threshold=sim_threshold,
                radius_m=radius_m,
                indoor_only=config.indoor_only,
            )
            
            if candidates.empty:
                st.warning(f"No candidate locations found with conflictivity > {sim_threshold}")
            else:
                st.success(f"✅ Found {len(candidates)} candidate locations")
                
                # Evaluate each candidate
                results = []
                progress_bar = st.progress(0)
                for idx, row in candidates.head(sim_top_k).iterrows():
                    progress_bar.progress((idx + 1) / min(sim_top_k, len(candidates)))
                    
                    _, new_ap_stats, metrics = simulate_ap_addition(
                        sim_df,
                        row['lat'],
                        row['lon'],
                        config,
                        redistribution_mode=sim_redistribution,
                    )
                    
                    result = {
                        'lat': row['lat'],
                        'lon': row['lon'],
                        'base_conflictivity': row['conflictivity'],
                        **metrics
                    }
                    results.append(result)
                
                progress_bar.empty()
                
                # Sort by composite score
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('composite_score', ascending=False)
                
                # Add blue dot for the best candidate (rank #1)
                if len(results_df) > 0:
                    best = results_df.iloc[0]
                    fig.add_trace(go.Scattermapbox(
                        lat=[best['lat']],
                        lon=[best['lon']],
                        mode='markers',
                        marker=dict(
                            size=25,
                            color='blue',
                            opacity=0.8
                        ),
                        name='🎯 Best Location',
                        hovertemplate=(
                            '<b>🎯 Best Placement Location</b><br>'
                            'Score: %{customdata[0]:.3f}<br>'
                            'Avg Reduction: %{customdata[1]:.3f} (%{customdata[2]:.1f}%)<br>'
                            'Worst AP Improvement: %{customdata[3]:.3f}<br>'
                            'New AP Clients: %{customdata[4]:.0f}<br>'
                            '<extra></extra>'
                        ),
                        customdata=np.column_stack([
                            [best['composite_score']],
                            [best['avg_reduction']],
                            [best['avg_reduction_pct']],
                            [best['worst_ap_improvement']],
                            [best['new_ap_client_count']],
                        ]),
                    ))
                
                # Add numbered markers for all candidates
                for idx, row in results_df.iterrows():
                    rank = idx + 1
                    # Use different colors: blue for #1, cyan for others
                    marker_color = 'yellow' if rank == 1 else 'cyan'
                    marker_size = 20 if rank == 1 else 16
                    
                    fig.add_trace(go.Scattermapbox(
                        lat=[row['lat']],
                        lon=[row['lon']],
                        mode='markers+text',
                        marker=dict(
                            size=marker_size,
                            color=marker_color,
                            symbol='star'
                        ),
                        text=f"#{rank}",
                        textposition="top center",
                        textfont=dict(size=14, color='darkblue' if rank == 1 else 'white', family='Arial Black'),
                        name=f'Proposed AP #{rank}',
                        hovertemplate=(
                            f'<b>Proposed AP #{rank}</b><br>'
                            'Score: %{customdata[0]:.3f}<br>'
                            'Avg Reduction: %{customdata[1]:.3f} (%{customdata[2]:.1f}%)<br>'
                            'Worst AP Improvement: %{customdata[3]:.3f}<br>'
                            'New AP Clients: %{customdata[4]:.0f}<br>'
                            '<extra></extra>'
                        ),
                        customdata=np.column_stack([
                            [row['composite_score']],
                            [row['avg_reduction']],
                            [row['avg_reduction_pct']],
                            [row['worst_ap_improvement']],
                            [row['new_ap_client_count']],
                        ]),
                    ))
                
                # Display results table
                st.divider()
                st.subheader("📊 Simulation Results")
                
                display_cols = ['lat', 'lon', 'composite_score', 'avg_reduction', 'avg_reduction_pct', 
                               'worst_ap_improvement', 'num_improved_aps', 'new_ap_client_count']
                display_df = results_df[display_cols].copy()
                display_df.columns = ['Latitude', 'Longitude', 'Score', 'Avg Reduction', 'Avg Reduction %', 
                                     'Worst AP Improv', '# Improved APs', 'New AP Clients']
                
                st.dataframe(
                    display_df.style.format({
                        'Latitude': '{:.6f}',
                        'Longitude': '{:.6f}',
                        'Score': '{:.3f}',
                        'Avg Reduction': '{:.3f}',
                        'Avg Reduction %': '{:.1f}%',
                        'Worst AP Improv': '{:.3f}',
                        'New AP Clients': '{:.0f}',
                    }).background_gradient(subset=['Score'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                best = results_df.iloc[0]
                
                with col1:
                    st.metric("Best Score", f"{best['composite_score']:.3f}")
                with col2:
                    st.metric("Avg Reduction", f"{best['avg_reduction']:.3f}", 
                             delta=f"{best['avg_reduction_pct']:.1f}%")
                with col3:
                    st.metric("Worst AP Improvement", f"{best['worst_ap_improvement']:.3f}")
                with col4:
                    st.metric("New AP Clients", f"{int(best['new_ap_client_count'])}")
                
                st.caption(f"💡 **Recommendation**: Place new AP at ({best['lat']:.6f}, {best['lon']:.6f}) for maximum network improvement")
    
    # Reset simulation flag
    st.session_state.run_sim = False

st.plotly_chart(fig, use_container_width=True)

# Footer
st.caption(
    "📻 Band mode aplicat (worst/avg/2.4/5) • "
    "💡 Conflictivity ≈ 0.85×airtime + 0.10×clients + 0.02×CPU + 0.03×Memòria  |  "
    "🎨 Escala: 🟢 Low → 🟡 Medium → 🔴 High (0–1)"
)