"""
Voronoi Visualization Module

Purpose:
    Provides weighted Voronoi diagrams with interpolated conflictivity surfaces.
    Includes network topology analysis with connectivity regions and hotspot detection.

Features:
    - Haversine distance calculations for geospatial coordinates
    - Convex hull polygon computation for coverage area
    - Distance-based interpolation kernels (decay/grow modes)
    - Weighted Voronoi edge generation with inverted conflictivity
    - Top-k Voronoi vertex detection for hotspot identification
    - Tiled choropleth layer generation with interpolation

Usage:
    from dashboard.voronoi_viz import (
        haversine_m,
        compute_convex_hull_polygon,
        inverted_weighted_voronoi_edges,
        uab_tiled_choropleth_layer
    )
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
from matplotlib.path import Path as MplPath
from shapely.geometry import LineString, MultiLineString, Point, Polygon

from .types import BoolArray, FloatArray


def haversine_m(
    lat1: float | FloatArray,
    lon1: float | FloatArray,
    lat2: float | FloatArray,
    lon2: float | FloatArray,
) -> float | FloatArray:
    """
    Calculate great-circle distance in meters using Haversine formula.
    
    Args:
        lat1: Latitude of first point(s) in degrees
        lon1: Longitude of first point(s) in degrees
        lat2: Latitude of second point(s) in degrees
        lon2: Longitude of second point(s) in degrees
    
    Returns:
        Distance in meters (scalar or array matching input shapes)
    
    Example:
        >>> d = haversine_m(41.5, 2.1, 41.6, 2.2)
        >>> print(f"Distance: {d:.2f}m")
    """
    assert np.all(np.abs(lat1) <= 90.0), "lat1 must be in [-90, 90]"
    assert np.all(np.abs(lat2) <= 90.0), "lat2 must be in [-90, 90]"
    
    R = 6371000.0  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dl = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2.0) ** 2
    distance = 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    
    # Postcondition: distance must be non-negative
    assert np.all(distance >= 0.0), "Distance must be non-negative"
    return cast(float | FloatArray, distance)


def compute_convex_hull_polygon(lons: FloatArray, lats: FloatArray) -> Polygon | None:
    """
    Compute the convex hull polygon of point data.

    Args:
        lons: Array of longitude values
        lats: Array of latitude values

    Returns:
        Shapely Polygon representing the convex hull, or None if fewer than 3 points.

    Examples:
        >>> from shapely.geometry import Polygon
        >>> lons = np.array([0.0, 1.0, 0.5])
        >>> lats = np.array([0.0, 0.0, 1.0])
        >>> hull = compute_convex_hull_polygon(lons, lats)
        >>> isinstance(hull, Polygon)
        True
    """
    from shapely.geometry import MultiPoint  # type: ignore[import-untyped]

    # Shapely stubs incomplete - MultiPoint constructor and convex_hull not fully typed
    if len(lons) < 3:
        return None
    
    pts = [(float(lon), float(lat)) for lon, lat in zip(lons, lats)]
    mp = MultiPoint(pts)  # type: ignore[misc]
    return mp.convex_hull  # type: ignore[return-value]


def mask_points_in_polygon(
    lon_grid: FloatArray,
    lat_grid: FloatArray,
    polygon: Polygon
) -> BoolArray:
    """
    Create boolean mask for grid points inside polygon.
    
    Args:
        lon_grid: 1D array of longitude values
        lat_grid: 1D array of latitude values
        polygon: Shapely Polygon for masking
    
    Returns:
        2D boolean mask (True for points inside polygon)
    
    Uses matplotlib.path.Path for efficient point-in-polygon testing.
    """
    assert len(lon_grid.shape) == 1, "lon_grid must be 1D"
    assert len(lat_grid.shape) == 1, "lat_grid must be 1D"
    assert not polygon.is_empty, "polygon must not be empty"
    
    # Note: shapely coordinate access has partially unknown types due to stub limitations
    x, y = polygon.exterior.coords.xy  # type: ignore[misc]
    poly_path = MplPath(np.vstack([x, y]).T)
    
    XX, YY = np.meshgrid(lon_grid, lat_grid)
    pts = np.vstack([XX.ravel(), YY.ravel()]).T
    inside = poly_path.contains_points(pts)
    
    mask = inside.reshape(XX.shape)
    assert mask.shape == (len(lat_grid), len(lon_grid)), "Mask shape mismatch"
    return cast(BoolArray, mask)


def interp_kernel(
    dist_m: FloatArray,
    R_m: float,
    mode: str = "decay"
) -> FloatArray:
    """
    Compute distance-based interpolation kernel weights.
    
    Args:
        dist_m: Distance array in meters
        R_m: Kernel radius in meters
        mode: "decay" (1-x) or "grow" (x) weighting
    
    Returns:
        Weight array (0 outside radius, interpolated inside)
    
    Preconditions:
        - R_m must be positive
        - mode must be "decay" or "grow"
    """
    assert R_m > 0.0, "R_m must be positive"
    assert mode in ["decay", "grow"], "mode must be 'decay' or 'grow'"
    assert np.all(dist_m >= 0.0), "dist_m must be non-negative"
    
    x = np.clip(dist_m / max(R_m, 1e-6), 0.0, 1.0)
    
    if mode == "grow":
        w = x
    else:  # decay
        w = 1.0 - x
    
    w[dist_m >= R_m] = 0.0
    
    # Postcondition: weights in [0, 1]
    assert np.all((w >= 0.0) & (w <= 1.0)), "Weights must be in [0, 1]"
    return w


def uab_tiled_choropleth_layer(
    df_uab: pd.DataFrame,
    *,
    tile_meters: float = 7.0,
    radius_m: float = 25.0,
    mode: str = "decay",
    value_mode: str = "conflictivity",
    max_tiles: int = 40000,
    colorscale: list[list[Any]] | None = None,
) -> tuple[Any, float, Polygon | None]:
    """
    Create Plotly Choroplethmapbox layer with rectangular tiles.
    
    Args:
        df_uab: DataFrame with 'lat', 'lon', 'conflictivity' columns
        tile_meters: Tile size in meters (auto-scales if > max_tiles)
        radius_m: Interpolation radius in meters
        mode: Kernel mode ("decay" or "grow")
        value_mode: "conflictivity" or "connectivity"
        max_tiles: Maximum number of tiles (scales if exceeded)
        colorscale: Custom Plotly colorscale (default: green-yellow-red)
    
    Returns:
        Tuple of (Choroplethmapbox trace, effective_tile_meters, hull_polygon)
    
    Preconditions:
        - df_uab must contain 'lat', 'lon', 'conflictivity' columns
        - tile_meters must be positive
        - radius_m must be positive
    """
    assert not df_uab.empty, "df_uab must not be empty"
    assert "lat" in df_uab.columns, "df_uab must have 'lat' column"
    assert "lon" in df_uab.columns, "df_uab must have 'lon' column"
    assert "conflictivity" in df_uab.columns, "df_uab must have 'conflictivity' column"
    assert tile_meters > 0.0, "tile_meters must be positive"
    assert radius_m > 0.0, "radius_m must be positive"
    
    if colorscale is None:
        colorscale = [
            [0.0, 'rgb(0, 255, 0)'],
            [0.5, 'rgb(255, 255, 0)'],
            [1.0, 'rgb(255, 0, 0)']
        ]
    
    # Note: pandas to_numpy has partially unknown types due to stub limitations
    lons = df_uab["lon"].to_numpy(dtype=float)  # type: ignore[misc]
    lats = df_uab["lat"].to_numpy(dtype=float)  # type: ignore[misc]
    
    hull_poly = compute_convex_hull_polygon(lons, lats)
    if hull_poly is None:
        return None, tile_meters, None
    
    # Calculate grid parameters
    lat0 = float(np.mean(lats)) if len(lats) > 0 else 41.5
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
    
    dlat = tile_meters / meters_per_deg_lat
    dlon = tile_meters / meters_per_deg_lon if meters_per_deg_lon > 0 else tile_meters / 100_000.0
    
    # Create tile centers
    minx, miny, maxx, maxy = hull_poly.bounds
    lon_centers = np.arange(minx + dlon/2, maxx, dlon)
    lat_centers = np.arange(miny + dlat/2, maxy, dlat)
    xx, yy = np.meshgrid(lon_centers, lat_centers)
    centers = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Filter to hull interior
    x_h, y_h = hull_poly.exterior.coords.xy  # type: ignore[misc]
    poly_path = MplPath(np.vstack([x_h, y_h]).T)
    inside = poly_path.contains_points(centers)
    centers_in = centers[inside]
    
    # Auto-scale if too many tiles
    effective_tile_meters = tile_meters
    n_tiles = centers_in.shape[0]
    if n_tiles > max_tiles and n_tiles > 0:
        factor = float(np.ceil(n_tiles / max_tiles))
        effective_tile_meters = tile_meters * factor
        dlat *= factor
        dlon *= factor
        
        lon_centers = np.arange(minx + dlon/2, maxx, dlon)
        lat_centers = np.arange(miny + dlat/2, maxy, dlat)
        xx, yy = np.meshgrid(lon_centers, lat_centers)
        centers = np.column_stack([xx.ravel(), yy.ravel()])
        inside = poly_path.contains_points(centers)
        centers_in = centers[inside]
    
    if centers_in.size == 0:
        return None, effective_tile_meters, hull_poly
    
    # Calculate distances from centers to APs
    dists_result = haversine_m(
        centers_in[:, 1][:, None],
        centers_in[:, 0][:, None],
        lats[None, :],
        lons[None, :]
    )
    # Ensure dists is array type for subsequent operations
    dists = cast(FloatArray, dists_result)
    
    # Compute tile values based on mode
    if value_mode == "connectivity":
        d_min = cast(FloatArray, dists.min(axis=1))  # type: ignore[misc]
        z_pred = np.clip(d_min / max(radius_m, 1e-6), 0.0, 1.0)
    else:  # conflictivity
        d_min = cast(FloatArray, dists.min(axis=1))  # type: ignore[misc]
        boundary_conf = np.where(d_min >= radius_m, 1.0, d_min / max(radius_m, 1e-6))
        
        cvals = df_uab["conflictivity"].to_numpy(dtype=float)  # type: ignore[misc]
        W = interp_kernel(dists, radius_m, mode=mode)
        denom = W.sum(axis=1)
        
        with np.errstate(invalid='ignore', divide='ignore'):
            num = (W * cvals[None, :]).sum(axis=1)
            weighted_conf = np.where(denom > 0, num / denom, np.nan)
        
        z_pred = np.maximum(weighted_conf, boundary_conf)
        z_pred = np.clip(z_pred, 0.0, 1.0)
    
    # Generate GeoJSON features
    features: list[dict[str, Any]] = []
    ids: list[str] = []
    
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
    ch = go.Choroplethmapbox(  # type: ignore[misc]
        geojson=geojson,
        locations=ids,
        z=z_pred,
        colorscale=colorscale,
        zmin=0,
        zmax=1,
        marker_opacity=0.9,
        marker_line_width=0,
        showscale=True,
        colorbar=dict(title=colorbar_title, thickness=15, len=0.7),
        name="UAB tiles",
    )
    
    return ch, effective_tile_meters, hull_poly


def inverted_weighted_voronoi_edges(
    df: pd.DataFrame,
    *,
    weight_col: str = "conflictivity",
    radius_m: float = 25.0,
    clip_polygon: Polygon | None = None,
    tolerance_m: float = 8.0,
) -> list[tuple[float, float, float, float]]:
    """
    Compute weighted Voronoi graph edges with inverted conflictivity.
    
    Args:
        df: DataFrame with 'lat', 'lon', and weight_col columns
        weight_col: Column name for weighting (default: 'conflictivity')
        radius_m: Radius for weighted distance calculations
        clip_polygon: Optional polygon to clip edges
        tolerance_m: Tolerance for edge filtering (meters)
    
    Returns:
        List of edge tuples (lon1, lat1, lon2, lat2)
    
    Uses scipy.spatial.Voronoi to generate network topology.
    Edges represent boundaries where inverted conflictivity is balanced.
    """
    assert not df.empty, "df must not be empty"
    assert weight_col in df.columns, f"df must have '{weight_col}' column"
    assert "lat" in df.columns, "df must have 'lat' column"
    assert "lon" in df.columns, "df must have 'lon' column"
    assert radius_m > 0.0, "radius_m must be positive"
    assert tolerance_m > 0.0, "tolerance_m must be positive"
    
    if df.empty or weight_col not in df.columns:
        return []
    
    # Pandas stubs incomplete for to_numpy/to_numeric/fillna overloads
    pts_lon = df["lon"].to_numpy(dtype=float)  # type: ignore[call-overload]
    pts_lat = df["lat"].to_numpy(dtype=float)  # type: ignore[call-overload]
    base = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)  # type: ignore[call-overload,union-attr]
    
    # Normalize and invert weights
    mn, mx = float(base.min()), float(base.max())
    norm = (base - mn) / (mx - mn + 1e-12)
    inv_w = 1.0 - norm
    
    if len(pts_lon) < 3:
        return []
    
    try:
        from scipy.spatial import Voronoi  # type: ignore[import-untyped]
    except ImportError:
        return []
    
    pts = np.column_stack([pts_lon, pts_lat])
    
    # Add small jitter to avoid duplicate points
    try:
        _, counts = np.unique(pts, axis=0, return_counts=True)
        if np.any(counts > 1):
            rng = np.random.RandomState(42)
            pts = pts + rng.randn(*pts.shape) * 1e-8
            pts_lon, pts_lat = pts[:, 0], pts[:, 1]
    except Exception:
        pass
    
    vor = Voronoi(pts)
    
    edges: list[tuple[float, float, float, float]] = []
    
    def _adiff(lon: float, lat: float, i1: int, i2: int) -> float:
        """Absolute difference in weighted distances."""
        d1 = haversine_m(lat, lon, pts_lat[i1], pts_lon[i1])
        d2 = haversine_m(lat, lon, pts_lat[i2], pts_lon[i2])
        return abs((d1 - inv_w[i1] * max(radius_m, 1e-6)) - (d2 - inv_w[i2] * max(radius_m, 1e-6)))
    
    def _keep_and_clip_segment(
        a: tuple[float, float],
        b: tuple[float, float],
        i1: int,
        i2: int
    ) -> list[tuple[float, float, float, float]]:
        """Clip segment to polygon and filter by tolerance."""
        # Shapely stubs incomplete - LineString, intersection, is_empty, geom_type not fully typed
        seg = LineString([a, b])  # type: ignore[misc]
        geom = seg if clip_polygon is None else seg.intersection(clip_polygon)  # type: ignore[misc]
        
        if geom.is_empty:  # type: ignore[misc]
            return []
        
        parts: list[LineString] = []
        if geom.geom_type == 'LineString':  # type: ignore[misc]
            parts = [cast(LineString, geom)]
        elif geom.geom_type == 'MultiLineString':  # type: ignore[misc]
            parts = list(cast(MultiLineString, geom).geoms)
        
        kept: list[tuple[float, float, float, float]] = []
        for ls in parts:
            coords = list(ls.coords)
            if len(coords) < 2:
                continue
            
            # Sample segment to check tolerance
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
    
    # Process finite edges
    for (p1, p2), rv in zip(vor.ridge_points, vor.ridge_vertices):
        if -1 not in rv:
            vcoords = vor.vertices[rv]
            lon1, lat1 = float(vcoords[0][0]), float(vcoords[0][1])
            lon2, lat2 = float(vcoords[1][0]), float(vcoords[1][1])
            kept = _keep_and_clip_segment((lon1, lat1), (lon2, lat2), p1, p2)
            edges.extend(kept)
        else:
            # Handle infinite edges (rays)
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
            
            # Create ray and intersect with polygon
            minx, miny, maxx, maxy = clip_polygon.bounds
            L = max(maxx - minx, maxy - miny) * 5.0 + 1e-6
            far = np.array([lon0, lat0]) + dir_vec * L
            # Shapely stubs incomplete - LineString, intersection, is_empty, geom_type not fully typed
            ray = LineString([(lon0, lat0), (float(far[0]), float(far[1]))])  # type: ignore[misc]
            inter = ray.intersection(clip_polygon)  # type: ignore[misc]
            
            if inter.is_empty:  # type: ignore[misc]
                continue
            
            def _add_segment_from_linestring(ls: LineString) -> None:
                coords = list(ls.coords)
                if len(coords) >= 2:
                    kept = _keep_and_clip_segment(coords[0], coords[-1], p1, p2)
                    edges.extend(kept)
            
            if inter.geom_type == 'LineString':  # type: ignore[misc]
                _add_segment_from_linestring(cast(LineString, inter))
            elif inter.geom_type == 'MultiLineString':  # type: ignore[misc]
                for ls in cast(MultiLineString, inter).geoms:
                    _add_segment_from_linestring(ls)
    
    # Final clipping pass if polygon specified
    if clip_polygon is not None:
        clipped: list[tuple[float, float, float, float]] = []
        for (x1, y1, x2, y2) in edges:
            # Shapely stubs incomplete - LineString, intersection, is_empty, geom_type not fully typed
            seg = LineString([(x1, y1), (x2, y2)])  # type: ignore[misc]
            inter = seg.intersection(clip_polygon)  # type: ignore[misc]
            
            if inter.is_empty:  # type: ignore[misc]
                continue
            
            if inter.geom_type == 'LineString':  # type: ignore[misc]
                coords = list(cast(LineString, inter).coords)
                if len(coords) >= 2:
                    (ax, ay), (bx, by) = coords[0], coords[-1]
                    clipped.append((ax, ay, bx, by))
            elif inter.geom_type == 'MultiLineString':  # type: ignore[misc]
                for seg2 in cast(MultiLineString, inter).geoms:
                    coords = list(seg2.coords)
                    if len(coords) >= 2:
                        (ax, ay), (bx, by) = coords[0], coords[-1]
                        clipped.append((ax, ay, bx, by))
        edges = clipped
    
    return edges


def top_conflictive_voronoi_vertices(
    df: pd.DataFrame,
    *,
    radius_m: float,
    coverage_poly: Polygon,
    k: int = 3
) -> list[tuple[float, float, float]]:
    """
    Return top-k Voronoi vertices with highest conflictivity scores.
    
    Args:
        df: DataFrame with 'lat', 'lon', 'conflictivity' columns
        radius_m: Interpolation radius for scoring
        coverage_poly: Polygon defining valid area
        k: Number of top vertices to return
    
    Returns:
        List of (lon, lat, score) tuples sorted by score (descending)
    
    Voronoi vertices represent potential hotspots where multiple
    high-conflictivity APs converge.
    """
    assert not df.empty, "df must not be empty"
    assert radius_m > 0.0, "radius_m must be positive"
    assert k >= 0, "k must be non-negative"
    assert not coverage_poly.is_empty, "coverage_poly must not be empty"
    
    if df.empty:
        return []
    
    try:
        from scipy.spatial import Voronoi  # type: ignore[import-untyped]
    except ImportError:
        return []
    
    # Pandas stubs incomplete for to_numpy overloads
    pts_lon = df["lon"].to_numpy(float)  # type: ignore[call-overload]
    pts_lat = df["lat"].to_numpy(float)  # type: ignore[call-overload]
    cvals = df["conflictivity"].to_numpy(float)  # type: ignore[call-overload]
    
    if len(pts_lon) < 3:
        return []
    
    pts = np.column_stack([pts_lon, pts_lat])
    
    # Add jitter for duplicate points
    try:
        _, counts = np.unique(pts, axis=0, return_counts=True)
        if np.any(counts > 1):
            rng = np.random.RandomState(7)
            pts = pts + rng.randn(*pts.shape) * 1e-8
    except Exception:
        pass
    
    vor = Voronoi(pts)
    
    cand: list[tuple[float, float, float]] = []
    for v in vor.vertices:
        lon, lat = float(v[0]), float(v[1])
        # Shapely stubs incomplete - Point constructor not fully typed
        p = Point(lon, lat)  # type: ignore[misc]
        
        if not coverage_poly.contains(p):  # type: ignore[misc]
            continue
        
        dists = haversine_m(lat, lon, pts_lat, pts_lon)
        dmin = float(cast(FloatArray, dists).min())
        
        # Boundary score (proximity penalty)
        boundary_conf = 1.0 if dmin >= radius_m else (dmin / max(radius_m, 1e-6))
        
        # Weighted conflictivity score - ensure dists is array for interp_kernel
        W = interp_kernel(np.atleast_1d(dists), radius_m, mode="decay")
        denom = float(W.sum())
        
        if denom <= 0:
            continue
        
        weighted_conf = float((W * cvals).sum() / denom)
        combined = max(weighted_conf, boundary_conf)
        
        # Filter out false positives at boundary
        if boundary_conf >= 0.98 and weighted_conf < 0.5:
            continue
        
        cand.append((lon, lat, combined))
    
    cand.sort(key=lambda t: t[2], reverse=True)
    return cand[:max(0, int(k or 0))]
