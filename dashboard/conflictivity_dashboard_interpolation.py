"""
Conflictivity Dashboard (Interpolated Surfaces)

Purpose
- Render an interpolated surface of Wi‑Fi conflictivity with time series navigation.
- Perform TWO independent interpolations:
  • UAB campus (all APs except AP-SAB-*), masked to its convex hull.
  • Sabadell site (AP-SAB-*) masked to its convex hull.

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
from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import Point, MultiPoint
from matplotlib.path import Path as MplPath
# No sklearn dependency: custom kernel-based interpolation


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


def read_ap_snapshot(path: Path) -> pd.DataFrame:
    """Load one AP snapshot JSON into a DataFrame with selected fields and conflictivity."""
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
    if "client_count" in df:
        df["client_count"] = pd.to_numeric(df["client_count"], errors="coerce").fillna(0)
    if "max_radio_util" in df:
        df["max_radio_util"] = pd.to_numeric(df["max_radio_util"], errors="coerce").fillna(0)
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
    """Kernel by distance in meters within radius R_m.
    - decay: 1 at d=0 -> 0 at d>=R
    - grow:  0 at d=0 -> 1 at d=R; 0 outside R (ring-peak at R)
    """
    x = np.clip(dist_m / max(R_m, 1e-6), 0.0, 1.0)
    if mode == "grow":
        w = x  # 0..1 inside R
    else:
        w = 1.0 - x  # default decay
    w[dist_m >= R_m] = 0.0
    return w


def _interpolate_conflictivity_kernel(df: pd.DataFrame, grid_size: int = 140, radius_m: float = 25.0,
                                      mode: str = "decay"):
    """Interpolate conflictivity onto a lon/lat grid using a custom distance kernel within radius_m.
    Returns (lon_grid, lat_grid, Z) masked by convex hull, or (None, None, None).
    """
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

    # Compute distances from each grid point (inside hull) to every AP
    Z = np.full(XX.shape, np.nan, dtype=float)
    if np.any(mask_inside):
        pts_lat = YY[mask_inside]
        pts_lon = XX[mask_inside]
        # Distàncies a tots els APs (P,M)
        dists = _haversine_m(pts_lat[:, None], pts_lon[:, None], lats[None, :], lons[None, :])
        # Termini de frontera: fora del radi -> 1; dins creix linealment fins 1 al radi.
        d_min = dists.min(axis=1)
        boundary_conf = np.where(d_min >= radius_m, 1.0, d_min / max(radius_m, 1e-6))
        # Interpolació ponderada clàssica (conflictivitat dels APs)
        W = _interp_kernel(dists, radius_m, mode=mode)  # (P,M)
        denom = W.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            num = (W * cvals[None, :]).sum(axis=1)
            weighted_conf = np.where(denom > 0, num / denom, np.nan)
        # Combineu: assegurem que el valor no sigui inferior al terme de frontera
        vals = np.maximum(weighted_conf, boundary_conf)
        Z[mask_inside] = np.clip(vals, 0.0, 1.0)

    return lon_grid, lat_grid, Z


def _add_density_layer(fig: go.Figure, lon_grid, lat_grid, Z, *, name: str,
                       showscale: bool, colorbar_title: str | None = None,
                       radius: int = 6, opacity: float = 0.9):
    """Add a Densitymapbox layer from a gridded surface by sampling all inside-hull points."""
    if lon_grid is None or lat_grid is None or Z is None:
        return
    # Build flattened arrays for points inside hull
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


def _weighted_voronoi_tiled_layer(
    df: pd.DataFrame,
    *,
    tile_meters: float = 3.0,
    radius_m: float = 25.0,
    weight_col: str = "connectivity_norm",
    max_tiles: int = 40000,
    colorscale=None,
    name: str = "AWVD"
):
    """Approximate additively weighted Voronoi (Apollonius) via tile assignment inside hull.

    For each tile center x, assign owner i = argmin_j ( d(x, p_j) - w_j * radius_m ),
    where w_j in [0,1] comes from df[weight_col]. Returns a Choroplethmapbox trace with
    each tile colored by the owner's weight.

    Returns: (choropleth_trace, effective_tile_meters, hull_polygon)
    """
    if colorscale is None:
        colorscale = [[0.0, 'rgb(0, 0, 255)'], [0.5, 'rgb(100, 200, 255)'], [1.0, 'rgb(0, 255, 255)']]

    if df.empty or weight_col not in df.columns:
        return None, tile_meters, None

    lons = df["lon"].to_numpy(dtype=float)
    lats = df["lat"].to_numpy(dtype=float)
    weights = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    weights = np.clip(weights, 0.0, 1.0)

    hull_poly = _compute_convex_hull_polygon(lons, lats)
    if hull_poly is None:
        return None, tile_meters, None

    lat0 = float(np.mean(lats)) if len(lats) else 41.5
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    dlat = tile_meters / meters_per_deg_lat
    dlon = tile_meters / max(meters_per_deg_lon, 1e-6)

    minx, miny, maxx, maxy = hull_poly.bounds
    lon_centers = np.arange(minx + dlon/2, maxx, dlon)
    lat_centers = np.arange(miny + dlat/2, maxy, dlat)
    XX, YY = np.meshgrid(lon_centers, lat_centers)
    centers = np.column_stack([XX.ravel(), YY.ravel()])

    # Mask inside hull
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

    # Distàncies i assignació AWVD
    dists = _haversine_m(centers_in[:, 1][:, None], centers_in[:, 0][:, None], lats[None, :], lons[None, :])
    # Additively weighted distance: d - w * radius_m
    awd = dists - (weights[None, :] * max(radius_m, 1e-6))
    owners = np.argmin(awd, axis=1)  # (tiles,)
    z_vals = weights[owners]

    # Build GeoJSON per tile
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
            "properties": {"owner": int(owners[i])},
            "geometry": {"type": "Polygon", "coordinates": poly_coords}
        })
        ids.append(str(i))

    geojson = {"type": "FeatureCollection", "features": features}

    ch = go.Choroplethmapbox(
        geojson=geojson,
        locations=ids,
        z=z_vals,
        colorscale=colorscale,
        zmin=0, zmax=1,
        marker_opacity=0.25,
        marker_line_width=1,
        marker_line_color="#444",
        showscale=True,
        colorbar=dict(title=f"AWVD weight ({weight_col})", thickness=12, len=0.5),
        name=name,
    )

    return ch, effective_tile_meters, hull_poly


def _uab_tiled_choropleth_layer(df_uab: pd.DataFrame, *, tile_meters: float = 3.0,
                                radius_m: float = 25.0, mode: str = "decay",
                                value_mode: str = "conflictivity",  # or "connectivity"
                                max_tiles: int = 40000, colorscale=None):
    """Create a Choroplethmapbox layer of rectangular tiles (~tile_meters side) inside the UAB convex hull.

    To avoid browser overload, if the number of tiles exceeds max_tiles the effective tile size is increased
    proportionally so that total tiles ≲ max_tiles.

    Returns: (choropleth_trace, effective_tile_meters, hull_polygon)
    """
    if colorscale is None:
        colorscale = [[0.0, 'rgb(0, 255, 0)'], [0.5, 'rgb(255, 255, 0)'], [1.0, 'rgb(255, 0, 0)']]

    # Compute hull of UAB
    lons = df_uab["lon"].to_numpy(dtype=float)
    lats = df_uab["lat"].to_numpy(dtype=float)
    hull_poly = _compute_convex_hull_polygon(lons, lats)
    if hull_poly is None:
        return None, tile_meters, None

    # degree per meter around campus latitude
    lat0 = float(np.mean(lats)) if len(lats) else 41.5
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    dlat = tile_meters / meters_per_deg_lat
    dlon = tile_meters / meters_per_deg_lon if meters_per_deg_lon > 0 else tile_meters / 100_000.0

    minx, miny, maxx, maxy = hull_poly.bounds
    # Prepare grid centers
    lon_centers = np.arange(minx + dlon/2, maxx, dlon)
    lat_centers = np.arange(miny + dlat/2, maxy, dlat)
    XX, YY = np.meshgrid(lon_centers, lat_centers)
    centers = np.column_stack([XX.ravel(), YY.ravel()])

    # Mask centers inside hull
    x_h, y_h = hull_poly.exterior.coords.xy
    poly_path = MplPath(np.vstack([x_h, y_h]).T)
    inside = poly_path.contains_points(centers)
    centers_in = centers[inside]

    # If too many tiles, coarsen grid step
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

    # distances from each center to each AP
    dists = _haversine_m(centers_in[:, 1][:, None], centers_in[:, 0][:, None], lats[None, :], lons[None, :])

    if value_mode == "connectivity":
        # Connectivitat: valor creix amb la distància al AP més proper fins al radi (1 al radi)
        d_min = dists.min(axis=1)  # (tiles,)
        z_pred = np.clip(d_min / max(radius_m, 1e-6), 0.0, 1.0)
    else:
        # Conflictivitat: ponderació per kernel + terme de frontera
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

    # Auto-coarsen tile size (increase tile_meters) if tile count exceeds max_tiles.
    grid_shape = (lat_centers.shape[0], lon_centers.shape[0])
    mask_grid = inside.reshape(grid_shape)
    n_initial = centers_in.shape[0]
    if n_initial > max_tiles and n_initial > 0:
        # Factor to reduce count roughly by (factor^2) since both lat & lon steps increase.
        factor = (n_initial / max_tiles) ** 0.5
        effective_tile_meters = tile_meters * factor
        # Recompute degree steps with enlarged tile size
        dlat = effective_tile_meters / meters_per_deg_lat
        dlon = effective_tile_meters / max(meters_per_deg_lon, 1e-6)
        lon_centers = np.arange(minx + dlon/2, maxx, dlon)
        lat_centers = np.arange(miny + dlat/2, maxy, dlat)
        XX, YY = np.meshgrid(lon_centers, lat_centers)
        centers = np.column_stack([XX.ravel(), YY.ravel()])
        x_h, y_h = hull_poly.exterior.coords.xy
        poly_path = MplPath(np.vstack([x_h, y_h]).T)
        inside = poly_path.contains_points(centers)
        centers_in = centers[inside]
        # Recompute distances & z_pred with enlarged tiles
        dists = _haversine_m(centers_in[:, 1][:, None], centers_in[:, 0][:, None], lats[None, :], lons[None, :])
        if value_mode == "connectivity":
            z_pred = np.clip(dists.min(axis=1) / max(radius_m, 1e-6), 0.0, 1.0)
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
    else:
        effective_tile_meters = tile_meters

    # Build GeoJSON FeatureCollection of rectangles per center
    features = []
    ids = []
    for i, (lon_c, lat_c) in enumerate(centers_in):
        # rectangle corners
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


def _uab_continuous_density_layer(
    df_uab: pd.DataFrame,
    *,
    step_meters: float = 2.0,
    radius_m: float = 25.0,
    value_mode: str = "conflictivity",
    kernel_mode: str = "decay",
    max_points: int = 120_000,
    radius_px: int = 6,
    colorscale=None,
):
    """Create a smooth Densitymapbox layer by sampling inside the UAB hull with ~step_meters spacing.

    Returns: (density_trace, effective_step_meters, hull_polygon)
    """
    if colorscale is None:
        colorscale = [[0.0, 'rgb(0, 255, 0)'], [0.5, 'rgb(255, 255, 0)'], [1.0, 'rgb(255, 0, 0)']]

    lons = df_uab["lon"].to_numpy(dtype=float)
    lats = df_uab["lat"].to_numpy(dtype=float)
    hull_poly = _compute_convex_hull_polygon(lons, lats)
    if hull_poly is None:
        return None, step_meters, None

    lat0 = float(np.mean(lats)) if len(lats) else 41.5
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    dlat = step_meters / meters_per_deg_lat
    dlon = step_meters / max(meters_per_deg_lon, 1e-6)

    minx, miny, maxx, maxy = hull_poly.bounds
    lon_centers = np.arange(minx + dlon/2, maxx, dlon)
    lat_centers = np.arange(miny + dlat/2, maxy, dlat)
    XX, YY = np.meshgrid(lon_centers, lat_centers)
    centers = np.column_stack([XX.ravel(), YY.ravel()])

    # Mask inside hull
    x_h, y_h = hull_poly.exterior.coords.xy
    poly_path = MplPath(np.vstack([x_h, y_h]).T)
    inside = poly_path.contains_points(centers)
    centers_in = centers[inside]

    effective_step = step_meters
    n_pts = centers_in.shape[0]
    if n_pts > max_points and n_pts > 0:
        factor = float(np.ceil(n_pts / max_points))
        effective_step = step_meters * factor
        dlat *= factor
        dlon *= factor
        lon_centers = np.arange(minx + dlon/2, maxx, dlon)
        lat_centers = np.arange(miny + dlat/2, maxy, dlat)
        XX, YY = np.meshgrid(lon_centers, lat_centers)
        centers = np.column_stack([XX.ravel(), YY.ravel()])
        inside = poly_path.contains_points(centers)
        centers_in = centers[inside]

    if centers_in.size == 0:
        return None, effective_step, hull_poly

    # Compute values at sample points
    dists = _haversine_m(centers_in[:, 1][:, None], centers_in[:, 0][:, None], lats[None, :], lons[None, :])
    if value_mode == "connectivity":
        z = np.clip(dists.min(axis=1) / max(radius_m, 1e-6), 0.0, 1.0)
    else:
        d_min = dists.min(axis=1)
        boundary_conf = np.where(d_min >= radius_m, 1.0, d_min / max(radius_m, 1e-6))
        cvals = df_uab["conflictivity"].to_numpy(dtype=float)
        W = _interp_kernel(dists, radius_m, mode=kernel_mode)
        denom = W.sum(axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            num = (W * cvals[None, :]).sum(axis=1)
            weighted_conf = np.where(denom > 0, num / denom, np.nan)
        z = np.maximum(weighted_conf, boundary_conf)
        z = np.clip(z, 0.0, 1.0)

    cb_title = "Connectivity (UAB smooth)" if value_mode == "connectivity" else "Conflictivity (UAB smooth)"
    trace = go.Densitymapbox(
        lon=centers_in[:, 0],
        lat=centers_in[:, 1],
        z=z,
        radius=radius_px,
        colorscale=colorscale,
        zmin=0, zmax=1,
        showscale=True,
        colorbar=dict(title=cb_title, thickness=15, len=0.7),
        hoverinfo='skip',
        name="UAB smooth",
    )

    return trace, effective_step, hull_poly


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

    # UAB interpolation (non-SAB)
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
        # AP points
        fig.add_trace(go.Scattermapbox(
            lat=uab_df['lat'], lon=uab_df['lon'], mode='markers',
            marker=dict(size=7, color='black', opacity=0.7),
            text=uab_df['name'], name="UAB APs",
            hovertemplate='<b>%{text}</b><br>Conflictivity src point<extra></extra>'
        ))

    # SAB interpolation (kernel-based)
    if show_sab and not sab_df.empty:
        g2_lon, g2_lat, g2_Z = _interpolate_conflictivity_kernel(sab_df, grid_size=120, radius_m=radius_m, mode=kernel_mode)
        _add_density_layer(fig, g2_lon, g2_lat, g2_Z, name="Sabadell surface", showscale=False)
        fig.add_trace(go.Scattermapbox(
            lat=sab_df['lat'], lon=sab_df['lon'], mode='markers',
            marker=dict(size=7, color='white', opacity=0.9),
            text=sab_df['name'], name="AP-SAB",
            hovertemplate='<b>%{text}</b><br>Conflictivity src point<extra></extra>'
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


# -------- UI --------
st.set_page_config(page_title="UAB Wi‑Fi Conflictivity (Interpolated)", page_icon="📶", layout="wide")
st.title("UAB Wi‑Fi Conflictivity — Interpolated Surfaces")
st.caption("Time series visualization • Interpolated conflictivity surfaces (UAB hull + AP-SAB hull)")

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

    min_conf = st.slider("Minimum conflictivity (table only)", 0.0, 1.0, 0.0, 0.01)
    show_uab = st.checkbox("Mostrar UAB (no SAB)", value=True)
    show_sab = st.checkbox("Mostrar Sabadell (AP-SAB)", value=True)
    tile_m = st.number_input("Mida del tile UAB (m)", min_value=1.0, max_value=100.0, value=3.0, step=1.0)
    max_tiles = st.slider("Límit màxim de tiles", 5000, 80000, 40000, step=5000, help="Si se supera, es mostreja una submostra de tiles preservant la mida del tile.")
    top_n = st.slider("Top N listing", 5, 50, 15, step=5)
    st.divider()
    st.header("Voronoi ponderat")
    show_awvd = st.checkbox("Mostrar diagrama AWVD", value=False, help="Additively weighted Voronoi (d - w*R).")
    weight_source = st.selectbox("Pes dels APs", ["client_norm", "util_norm", "conflictivity"], index=0,
                                 help="Columna usada com a pes normalitzat (0..1) en el diagrama AWVD.")

# Load data for selected timestamp
ap_df = read_ap_snapshot(selected_path)

# Merge AP + geoloc; keep ALL data for interpolation consistency
merged = ap_df.merge(geo_df, on="name", how="inner")

if merged.empty:
    st.info("No APs have geolocation data.")
    st.stop()

# Optional group filter (by prefix code)
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

# Create dual interpolated map (UAB + SAB)
fig = create_dual_interpolated_map(
    df=map_df,
    center_lat=center_lat,
    center_lon=center_lon,
    show_uab=show_uab,
    show_sab=show_sab,
    zoom=15,
    tile_meters=float(tile_m),
    max_tiles=int(max_tiles),
    radius_m=radius_m,
    value_mode=value_mode,
)

# Afegir capa AWVD si es demana (sobre UAB només, excloent SAB per simplicitat)
if show_awvd:
    awvd_df = map_df[map_df["group_code"] != "SAB"].copy()
    if not awvd_df.empty and weight_source in awvd_df.columns:
        # Assegurar normalització 0..1 de la columna triada (si no ho està ja)
        col = weight_source
        vals = pd.to_numeric(awvd_df[col], errors="coerce").fillna(0.0)
        mn, mx = float(vals.min()), float(vals.max())
        if mx - mn > 1e-12:
            awvd_df[col + "_norm_tmp"] = (vals - mn) / (mx - mn)
            weight_col = col + "_norm_tmp"
        else:
            awvd_df[col + "_norm_tmp"] = 0.0
            weight_col = col + "_norm_tmp"
        awvd_trace, awvd_eff_tile, _ = _weighted_voronoi_tiled_layer(
            awvd_df, tile_meters=float(tile_m), radius_m=radius_m,
            weight_col=weight_col, max_tiles=int(max_tiles), name="AWVD"
        )
        if awvd_trace is not None:
            fig.add_trace(awvd_trace)
            fig.add_annotation(text=f"AWVD tile ≈ {awvd_eff_tile:.1f} m",
                               showarrow=False, xref="paper", yref="paper",
                               x=0.02, y=0.92, bgcolor="rgba(255,255,255,0.7)", bordercolor="#666", font=dict(size=10))

st.plotly_chart(fig, use_container_width=True)


# Top conflictive listing — filtered only for the table visualization
st.subheader("Top conflictive Access Points")
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

    if "Conflictivity Score" in top_df.columns:
        top_df["Conflictivity Score"] = top_df["Conflictivity Score"].map(lambda x: f"{x:.3f}")

    st.dataframe(top_df, use_container_width=True, hide_index=True)


# Footer
st.caption(
    "💡 Conflictivity = 0.6 × normalized_clients + 0.4 × max_radio_utilization  |  "
    "🎨 Color Scale: 🟢 Green (Low) → 🟡 Yellow (Medium) → 🔴 Red (High)"
)
