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
from shapely.geometry import Point, MultiPoint, LineString, Polygon
from shapely.ops import unary_union, linemerge, snap
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
    import math
    x = math.log1p(n) / math.log1p(denom)
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


def _inverted_weighted_voronoi_edges(df: pd.DataFrame, *, weight_col: str = "conflictivity",
                                     radius_m: float = 25.0, clip_polygon=None,
                                     tolerance_m: float = 8.0):
    """Compute an approximate additively weighted Voronoi graph (edges only) by:
    1. Invert weights: w' = 1 - norm(weight_col)
       (Baixa connectivitat => pes alt => cel·la gran en diagrama additiu.)
    2. Use SciPy Voronoi (unweighted) on AP coordinates as a geometric scaffold.
    3. Adjust edge inclusion heuristically by comparing additive distances of the two sites.

    Returns list of edge segments (lon1, lat1, lon2, lat2).
    """
    if df.empty or weight_col not in df.columns:
        return []
    pts_lon = df["lon"].to_numpy(dtype=float)
    pts_lat = df["lat"].to_numpy(dtype=float)
    base = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    # Normalització i inversió
    mn, mx = float(base.min()), float(base.max())
    norm = (base - mn) / (mx - mn + 1e-12)
    inv_w = 1.0 - norm  # baixa connectivitat -> valor gran

    if len(pts_lon) < 3:
        return []
    try:
        from scipy.spatial import Voronoi
    except Exception:
        return []
    pts = np.column_stack([pts_lon, pts_lat])
    # Ensure uniqueness to avoid Qhull errors: jitter duplicates slightly
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
            # sample K points along this clipped part and test min additive diff
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
            # Finite edge
            vcoords = vor.vertices[rv]  # shape (2,2)
            lon1, lat1 = float(vcoords[0][0]), float(vcoords[0][1])
            lon2, lat2 = float(vcoords[1][0]), float(vcoords[1][1])
            kept = _keep_and_clip_segment((lon1, lat1), (lon2, lat2), p1, p2)
            edges.extend(kept)
        else:
            # Infinite edge: extend ray to boundary and clip
            # Identify finite vertex
            vs = [v for v in rv if v != -1]
            if not vs:
                continue
            v0 = vor.vertices[vs[0]]
            lon0, lat0 = float(v0[0]), float(v0[1])
            # Direction: perpendicular to segment p2-p1
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
            # Orient away from center
            if np.dot(mid - center, dir_vec) < 0:
                dir_vec *= -1.0
            # Build a long segment and intersect with clip polygon
            if clip_polygon is None:
                # No boundary to clip against; skip to avoid unbounded lines
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

    # Optional clipping: trim segments to the clip polygon
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


def _snap_and_connect_edges(segments: list[tuple[float,float,float,float]], clip_polygon: Polygon,
                            *, lat0: float, snap_m: float = 2.0, join_m: float = 4.0):
    """Post-process Voronoi segments to enforce connectivity.
    - Snap endpoints to each other and to the clip polygon boundary within snap_m.
    - Connect dangling endpoints (degree=1) that are inside the polygon and within join_m of another endpoint.

    Returns a shapely LineString/MultiLineString geometry with merged lines.
    """
    if not segments or clip_polygon is None:
        return None
    try:
        from shapely.geometry import MultiLineString, MultiPoint
    except Exception:
        return None

    # Convert meters to degrees (approx)
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    tol_deg = max(snap_m / max(meters_per_deg_lat, 1e-6), snap_m / max(meters_per_deg_lon, 1e-6))
    join_deg = max(join_m / max(meters_per_deg_lat, 1e-6), join_m / max(meters_per_deg_lon, 1e-6))

    lines = [LineString([(x1, y1), (x2, y2)]) for (x1, y1, x2, y2) in segments]
    mls = MultiLineString(lines)

    # Build target: endpoints + polygon boundary
    endpoints = []
    for (x1, y1, x2, y2) in segments:
        endpoints.append((x1, y1))
        endpoints.append((x2, y2))
    mp = MultiPoint(endpoints)
    target = unary_union([mp, clip_polygon.boundary])

    # 1) Snap to endpoints and boundary, then clip to polygon
    snapped = snap(mls, target, tol_deg)
    clipped = snapped.intersection(clip_polygon)
    # If clip produced Points or empty, bail early
    if clipped.is_empty:
        return None
    if clipped.geom_type in ("Point", "MultiPoint"):
        return None

    # 2) Identify dangling endpoints (degree=1) far from boundary and connect to nearest endpoint
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

    # Count degrees by quantized endpoint
    from collections import defaultdict
    deg = defaultdict(int)
    for p in pts:
        deg[_quantize(p, tol_deg)] += 1

    # Build a set of candidate dangling points
    dangling = []
    for p in pts:
        qp = _quantize(p, tol_deg)
        # Degree counting counts each appearance; endpoints of middle nodes may appear twice
        if deg[qp] == 1:
            # if close to boundary, treat as connected to hull
            if Point(p).distance(clip_polygon.boundary) <= tol_deg * 1.5:
                continue
            dangling.append(p)

    added_connectors = []
    if dangling:
        # Prepare spatial index: simple brute-force nearest as counts are small
        for p in dangling:
            # find nearest other endpoint within join_deg
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
                # Clip and add only if inside polygon
                inter = seg.intersection(clip_polygon)
                if not inter.is_empty:
                    if inter.geom_type == 'LineString':
                        added_connectors.append(inter)
                    elif inter.geom_type == 'MultiLineString':
                        for ls in inter.geoms:
                            added_connectors.append(ls)

    # Ensure all added connectors are LineStrings
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


def _compute_connectivity_components(df: pd.DataFrame, radius_m: float) -> list[tuple[pd.DataFrame, object]]:
    """Split APs into connected components under a distance threshold radius_m.
    Returns a list of tuples: (sub_df, hull_polygon). Components with < 3 points are included
    but their hull may be None; callers can skip them for Voronoi edges.
    """
    if df.empty:
        return []
    lons = df["lon"].to_numpy(dtype=float)
    lats = df["lat"].to_numpy(dtype=float)
    n = len(lons)
    if n == 0:
        return []
    # Pairwise distances
    D = _haversine_m(lats[:, None], lons[:, None], lats[None, :], lons[None, :])
    adj = (D <= max(radius_m, 1e-6)) & (~np.eye(n, dtype=bool))
    visited = np.zeros(n, dtype=bool)
    comps = []
    for i in range(n):
        if visited[i]:
            continue
        # BFS/DFS
        stack = [i]
        visited[i] = True
        idxs = [i]
        while stack:
            u = stack.pop()
            neighbors = np.where(adj[u])[0]
            for v in neighbors:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
                    idxs.append(v)
        sub = df.iloc[idxs].copy()
        hull = None
        if len(idxs) >= 3:
            hull = _compute_convex_hull_polygon(sub["lon"].to_numpy(float), sub["lat"].to_numpy(float))
        comps.append((sub, hull))
    return comps


def _coverage_regions_from_uab_tiles(uab_df: pd.DataFrame, tile_meters: float, radius_m: float,
                                     max_tiles: int = 40000) -> list[Polygon]:
    """Approximate coverage regions using the same tiling logic as UAB tiles.
    A tile is considered 'covered' if its minimum distance to any AP < radius_m.
    Connected covered tiles are merged and their union polygon extracted.
    Returns list of polygons (one per connected coverage area)."""
    if uab_df.empty:
        return []
    lons = uab_df["lon"].to_numpy(float)
    lats = uab_df["lat"].to_numpy(float)
    hull = _compute_convex_hull_polygon(lons, lats)
    if hull is None:
        return []
    # Grid resolution (reuse logic from _uab_tiled_choropleth_layer minimal subset)
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
    # Distàncies als APs
    dists = _haversine_m(centers_in[:,1][:,None], centers_in[:,0][:,None], lats[None,:], lons[None,:])
    d_min = dists.min(axis=1)
    covered_mask = d_min < radius_m
    covered_centers = centers_in[covered_mask]
    if covered_centers.size == 0:
        return []
    # Construir polígons de cada tile cobert
    tile_polys = []
    for (cx, cy) in covered_centers:
        lon0, lon1 = cx - dlon/2, cx + dlon/2
        lat0, lat1 = cy - dlat/2, cy + dlat/2
        tile_polys.append(Polygon([(lon0, lat0), (lon1, lat0), (lon1, lat1), (lon0, lat1)]))
    # Unir i separar components
    merged = unary_union(tile_polys)
    polys = []
    if merged.geom_type == 'Polygon':
        polys = [merged]
    elif merged.geom_type == 'MultiPolygon':
        polys = list(merged.geoms)
    # Opcionalment filtrar polígons massa petits
    final_polys = [p for p in polys if p.area > 0]
    return final_polys


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
    band_mode = st.radio(
        "Band Mode",
        options=["worst", "avg", "2.4GHz", "5GHz"],
        index=0,
        help="worst: max(max_2.4, max_5) • avg: weighted average of band maxima",
        horizontal=True,
    )
    show_uab = st.checkbox("Mostrar UAB (no SAB)", value=True)
    show_sab = st.checkbox("Mostrar Sabadell (AP-SAB)", value=True)
    # Mida de tile fixa (7 m) i sense límit de tiles
    TILE_M_FIXED = 7.0
    MAX_TILES_NO_LIMIT = 1_000_000_000
    st.divider()
    st.header("Voronoi ponderat")
    show_awvd = st.checkbox("Mostrar arestes Voronoi ponderat", value=False,
                            help="Aproximació additivament ponderada (edges) – baixa connectivitat => pes alt.")
    weight_source = st.selectbox(
        "Base connectivitat (per invertir)",
        ["conflictivity", "client_count", "max_radio_util", "airtime_score"],
        index=0,
        help="Es normalitza i s'inverteix: pes = 1 - norm(col)."
    )
    # Paràmetres fixos per a AWVD (sense sliders)
    VOR_TOL_M_FIXED = 24.0
    SNAP_M_DEFAULT = float(max(1.5, TILE_M_FIXED * 0.2))
    JOIN_M_DEFAULT = float(max(3.0, TILE_M_FIXED * 0.6))

# Load data for selected timestamp
ap_df = read_ap_snapshot(selected_path, band_mode=band_mode)

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
    tile_meters=TILE_M_FIXED,
    max_tiles=MAX_TILES_NO_LIMIT,
    radius_m=radius_m,
    value_mode=value_mode,
)

# Afegir capa AWVD si es demana (sobre UAB només, excloent SAB per simplicitat)
if show_awvd:
    aw_df = map_df[map_df["group_code"] != "SAB"].copy()
    base_col = weight_source if weight_source in aw_df.columns else "conflictivity"
    if not aw_df.empty and base_col in aw_df.columns:
        # 1) Regions de cobertura (unió), amb fallback a hull complet
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

        # 2) Dedupe: agrupar APs solapats per (lon,lat) i prendre el pes "pitjor" (max)
        dedup = aw_df[["lon","lat", base_col]].copy()
        dedup = (dedup.groupby(["lon","lat"], as_index=False)
                       .agg({base_col: "max"}))

        # 3) Buffer lleu de la geometria per evitar talls numèrics
        if union_poly is not None:
            try:
                lat0 = float(aw_df["lat"].mean()) if len(aw_df) else 41.5
                meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
                eps_deg = (max(0.5, TILE_M_FIXED * 0.15)) / max(meters_per_deg_lon, 1e-6)  # ~0.5–1m
                union_poly = union_poly.buffer(eps_deg)
            except Exception:
                pass

        # 4) Voronoi ponderat sobre TOTS els punts, tallat per la unió
        total_edges = _inverted_weighted_voronoi_edges(
            dedup.rename(columns={base_col: base_col}),
            weight_col=base_col,
            radius_m=radius_m,
            clip_polygon=union_poly,
            tolerance_m=VOR_TOL_M_FIXED
        ) if union_poly is not None and len(dedup) >= 3 else []

        if total_edges:
            # Post-process: snap and connect to enforce end-to-end connectivity
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
        # Dibuixar la/les regions de cobertura (unió) per referència
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
            # Anotació: nombre de polígons
            n_regs = 1 if union_poly.geom_type == 'Polygon' else len(list(union_poly.geoms))
            fig.add_annotation(text=f"Voronoi ponderat (edges) — {n_regs} regions", xref="paper", yref="paper", x=0.02, y=0.90,
                               showarrow=False, bgcolor="rgba(0,0,0,0.4)", font=dict(color='white', size=10))

st.plotly_chart(fig, use_container_width=True)


# Footer
st.caption(
    "📻 Band mode aplicat (worst/avg/2.4/5) • "
    "💡 Conflictivity ≈ 0.85×airtime + 0.10×clients + 0.02×CPU + 0.03×Memòria  |  "
    "🎨 Escala: 🟢 Low → 🟡 Medium → 🔴 High (0–1)"
)
