"""
Shared spatial utility functions.
"""

import numpy as np
from shapely.geometry import Point, MultiPoint
from matplotlib.path import Path as MplPath


def haversine_m(lat1, lon1, lat2, lon2):
    """
    Great-circle distance in meters using Haversine formula.
    
    Works with numpy arrays via broadcasting.
    
    Args:
        lat1, lon1: First point(s) latitude/longitude (degrees)
        lat2, lon2: Second point(s) latitude/longitude (degrees)
    
    Returns:
        Distance(s) in meters
    """
    R = 6371000.0  # Earth radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    
    a = (
        np.sin(dphi / 2.0) ** 2 +
        np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    )
    
    return 2 * R * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def compute_convex_hull_polygon(lons: np.ndarray, lats: np.ndarray):
    """
    Compute convex hull polygon from points.
    
    Args:
        lons: Array of longitudes
        lats: Array of latitudes
    
    Returns:
        Shapely Polygon or None if insufficient points
    """
    pts = [Point(xy) for xy in zip(lons, lats)]
    if len(pts) < 3:
        return None
    
    mp = MultiPoint(pts)
    hull = mp.convex_hull
    
    if hull.is_empty or hull.geom_type != "Polygon":
        return None
    
    return hull


def mask_points_in_polygon(lon_grid: np.ndarray, lat_grid: np.ndarray, polygon) -> np.ndarray:
    """
    Create boolean mask for grid points inside polygon.
    
    Args:
        lon_grid: 1D array of longitude values
        lat_grid: 1D array of latitude values
        polygon: Shapely Polygon
    
    Returns:
        2D boolean mask (shape: len(lat_grid) × len(lon_grid))
    """
    x, y = polygon.exterior.coords.xy
    poly_path = MplPath(np.vstack([x, y]).T)
    
    XX, YY = np.meshgrid(lon_grid, lat_grid)
    pts = np.vstack([XX.ravel(), YY.ravel()]).T
    inside = poly_path.contains_points(pts)
    
    return inside.reshape(XX.shape)