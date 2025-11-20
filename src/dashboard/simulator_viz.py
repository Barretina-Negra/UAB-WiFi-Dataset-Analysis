"""
AP Placement Simulator Visualization Module

This module provides simulation capabilities for WiFi access point placement optimization,
including client distribution modeling, interference calculation, and candidate location generation.

Key Functions:
    - recalculate_conflictivity: Recompute conflictivity after network changes
    - compute_rssi: RSSI calculation using log-distance path loss model
    - estimate_client_distribution: Model client redistribution when adding new AP
    - apply_cca_interference: Apply co-channel interference effects
    - simulate_ap_addition: Full simulation of adding a new AP
    - generate_candidate_locations: Tile-based candidate generation
    - generate_voronoi_candidates: Network-aware Voronoi vertex candidates
    - aggregate_scenario_results: Multi-scenario performance aggregation
    - simulate_multiple_ap_additions: Sequential multi-AP placement simulation
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from .types import (
    BoolArray,
    FloatArray,
    W_AIR_SIM,
    W_CL_SIM,
    W_CPU_SIM,
    W_MEM_SIM,
)

# Import from other dashboard modules
from dashboard.voronoi_viz import (
    compute_convex_hull_polygon,
    haversine_m,
    mask_points_in_polygon,
)

if TYPE_CHECKING:
    from experiments.polcorresa.simulator.config import SimulationConfig as SimulationConfigType
    from experiments.polcorresa.simulator.config import StressLevel as StressLevelType
    from experiments.polcorresa.simulator.scoring import CompositeScorer as CompositeScorerType
else:
    SimulationConfigType = Any
    StressLevelType = Any
    CompositeScorerType = Any


# ======== SCORING UTILITIES ========

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp value to range [lo, hi].
    
    Args:
        x: Value to clamp
        lo: Minimum value
        hi: Maximum value
        
    Returns:
        Clamped value
    """
    return max(lo, min(hi, x))


def airtime_score(util: float, band: str) -> float:
    """Map channel utilization % to [0,1] pain score.
    
    Args:
        util: Channel utilization percentage (0-100)
        band: Band identifier ("2g" or "5g")
        
    Returns:
        Pain score from 0.0 (no congestion) to 1.0 (severe congestion)
    """
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
    """Calculate client pressure score relative to network percentile.
    
    Args:
        n_clients: Number of clients on AP
        peers_p95: 95th percentile of client counts across network
        
    Returns:
        Pressure score from 0.0 to 1.0
    """
    n = max(0.0, float(n_clients or 0.0))
    denom = max(1.0, float(peers_p95 or 1.0))
    x = math.log1p(n) / math.log1p(denom)
    return clamp(x, 0.0, 1.0)


def cpu_health_score(cpu_pct: float) -> float:
    """Calculate CPU health stress score.
    
    Args:
        cpu_pct: CPU utilization percentage (0-100)
        
    Returns:
        Health score from 0.0 (healthy) to 1.0 (critical)
    """
    c = clamp(cpu_pct or 0.0, 0.0, 100.0)
    if c <= 70:
        return 0.0
    if c <= 90:
        return 0.6 * ((c - 70) / 20.0)
    return 0.6 + 0.4 * ((c - 90) / 10.0)


def mem_health_score(mem_used_pct: float) -> float:
    """Calculate memory health stress score.
    
    Args:
        mem_used_pct: Memory utilization percentage (0-100)
        
    Returns:
        Health score from 0.0 (healthy) to 1.0 (critical)
    """
    m = clamp(mem_used_pct or 0.0, 0.0, 100.0)
    if m <= 80:
        return 0.0
    if m <= 95:
        return 0.6 * ((m - 80) / 15.0)
    return 0.6 + 0.4 * ((m - 95) / 5.0)


# ======== CORE SIMULATION FUNCTIONS ========

def recalculate_conflictivity(df: pd.DataFrame) -> pd.DataFrame:
    """Full conflictivity recalculation after network changes.
    
    Args:
        df: DataFrame with AP metrics (util_2g, util_5g, client_count, etc.)
        
    Returns:
        DataFrame with updated conflictivity scores
        
    Notes:
        Uses W_AIR_SIM, W_CL_SIM, W_CPU_SIM, W_MEM_SIM weights
    """
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
    
    df['airtime_score_filled'] = df['airtime_score_adj'].fillna(0.4)
    df['conflictivity'] = (
        df['airtime_score_filled'] * W_AIR_SIM +
        df['client_score'].fillna(0) * W_CL_SIM +
        df['cpu_score'].fillna(0) * W_CPU_SIM +
        df['mem_score'].fillna(0) * W_MEM_SIM
    ).clip(0, 1)
    
    return df


def compute_rssi(distance_m: float, config: SimulationConfigType) -> float:
    """Compute RSSI using log-distance path loss model.
    
    Args:
        distance_m: Distance from AP in meters
        config: Simulation configuration with path loss parameters
        
    Returns:
        RSSI value in dBm
    """
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
    config: SimulationConfigType,
    mode: str = 'hybrid'
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Simulate client redistribution when a new AP is added.
    
    Args:
        df_aps: DataFrame with existing AP information
        new_ap_lat: Latitude of new AP
        new_ap_lon: Longitude of new AP
        config: Simulation configuration
        mode: Distribution mode ('hybrid', 'signal', or 'distance')
        
    Returns:
        Tuple of (updated_dataframe, new_ap_stats_dict)
    """
    df = df_aps.copy()
    
    df['dist_to_new'] = haversine_m(
        new_ap_lat, new_ap_lon,
        df['lat'].values, df['lon'].values
    )
    
    df['rssi_new'] = df['dist_to_new'].apply(lambda d: compute_rssi(d, config))
    
    df['in_range'] = df['dist_to_new'] <= config.interference_radius_m
    
    total_transferred = 0
    
    # We consider all APs in range as candidates for offloading, regardless of their current conflictivity.
    # Clients roam based on signal strength, not the AP's stress level.
    candidates = df[df['in_range'] & (df['client_count'] > 0)].copy()
    
    if not candidates.empty:
        # Sort by conflictivity descending (prioritize helping stressed APs)
        candidates = candidates.sort_values('conflictivity', ascending=False)
        
        for idx, row in candidates.iterrows():
            signal_strength = max(0.0, (row['rssi_new'] - config.min_rssi_dbm) / 20.0)
            signal_strength = min(1.0, signal_strength)
            
            distance_factor = 1.0 - (row['dist_to_new'] / config.interference_radius_m)
            distance_factor = max(0.0, min(1.0, distance_factor))
            
            conflict_factor = float(row.get('conflictivity', 0.5))
            
            # Balanced approach: signal quality and current stress both drive migration.
            transfer_potential = (
                0.40 * signal_strength +
                0.20 * distance_factor +
                0.40 * conflict_factor
            )
            
            transfer_fraction = min(config.max_offload_fraction, transfer_potential)
            
            transfer_fraction *= (1 - config.sticky_client_fraction)
            
            # Use round() instead of int() to handle small client counts better
            n_transfer = int(round(row['client_count'] * transfer_fraction))
            
            n_transfer = min(n_transfer, int(row['client_count']))
            
            if n_transfer > 0:
                # Reduce utilization proportionally to client loss
                fraction_removed = n_transfer / row['client_count']
                
                # Update 2G
                current_2g = row['util_2g'] if not pd.isna(row['util_2g']) else 0.0
                new_2g = max(5.0, current_2g * (1 - fraction_removed))
                df.at[idx, 'util_2g'] = new_2g
                
                # Update 5G
                current_5g = row['util_5g'] if not pd.isna(row['util_5g']) else 0.0
                new_5g = max(5.0, current_5g * (1 - fraction_removed))
                df.at[idx, 'util_5g'] = new_5g

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
    new_ap_stats: Dict[str, float],
    config: SimulationConfigType,
) -> pd.DataFrame:
    """Apply co-channel interference (CCA busy increase) to neighbors.
    
    Args:
        df_aps: DataFrame with AP information
        new_ap_stats: Dictionary with new AP statistics
        config: Simulation configuration
        
    Returns:
        Updated DataFrame with interference effects applied
    """
    df = df_aps.copy()
    
    distances = haversine_m(
        new_ap_stats['lat'], new_ap_stats['lon'],
        df['lat'].values, df['lon'].values
    )
    
    in_interference_range = distances <= config.interference_radius_m
    
    increase_factor = np.where(
        in_interference_range,
        config.cca_increase_factor * (1 - distances / config.interference_radius_m),
        0.0
    )
    
    # Apply channel overlap probabilities
    prob_2g = getattr(config, 'channel_overlap_prob_2g', 0.33)
    prob_5g = getattr(config, 'channel_overlap_prob_5g', 0.10)
    
    df['util_2g'] = np.clip(
        df['util_2g'] * (1 + increase_factor * prob_2g),
        0.0, 100.0
    )
    df['util_5g'] = np.clip(
        df['util_5g'] * (1 + increase_factor * prob_5g),
        0.0, 100.0
    )
    
    df['agg_util'] = np.maximum(df['util_2g'], df['util_5g'])
    
    return df


def simulate_ap_addition(
    df_baseline: pd.DataFrame,
    new_ap_lat: float,
    new_ap_lon: float,
    config: SimulationConfigType,
    scorer: CompositeScorerType,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Simulate adding a new AP at given location.
    
    Args:
        df_baseline: Baseline network state
        new_ap_lat: Latitude of new AP
        new_ap_lon: Longitude of new AP
        config: Simulation configuration
        scorer: Composite scoring object
        
    Returns:
        Tuple of (updated_df, new_ap_stats, scenario_metrics)
    """
    # Ensure baseline conflictivity is consistent with our current scoring logic
    # This prevents artifacts where the simulator's scoring differs from the dataset's pre-calculated values
    df_baseline = recalculate_conflictivity(df_baseline.copy())

    df_updated, new_ap_stats = estimate_client_distribution(
        df_baseline, new_ap_lat, new_ap_lon, config, mode='hybrid'
    )
    
    df_updated = apply_cca_interference(df_updated, new_ap_stats, config)
    
    df_updated = recalculate_conflictivity(df_updated)
    
    distances = haversine_m(
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
    
    metrics: Dict[str, Any] = {
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


# ======== TILE MASK UTILITIES ========

def compute_tile_masks(tile_mask: BoolArray, neighbor_radius: int) -> tuple[BoolArray, BoolArray]:
    """Return (valid_tiles, boundary_tiles) masks for the provided painted grid.
    
    Args:
        tile_mask: Boolean mask indicating which tiles are painted
        neighbor_radius: Radius of neighbors to check (in tiles)
        
    Returns:
        Tuple of (valid_tiles_mask, boundary_tiles_mask)
    """
    assert neighbor_radius >= 0, "neighbor_radius must be non-negative"

    rows, cols = tile_mask.shape
    valid_tiles = np.zeros_like(tile_mask, dtype=bool)
    boundary_tiles = np.zeros_like(tile_mask, dtype=bool)

    immediate_offsets = [
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
    ]

    for row in range(rows):
        for col in range(cols):
            if not tile_mask[row, col]:
                continue
            is_boundary = False
            for d_row, d_col in immediate_offsets:
                n_row = row + d_row
                n_col = col + d_col
                if n_row < 0 or n_row >= rows or n_col < 0 or n_col >= cols:
                    is_boundary = True
                    break
                if not tile_mask[n_row, n_col]:
                    is_boundary = True
                    break
            boundary_tiles[row, col] = is_boundary
            if is_boundary:
                continue

            all_neighbors_present = True
            for d_row in range(-neighbor_radius, neighbor_radius + 1):
                for d_col in range(-neighbor_radius, neighbor_radius + 1):
                    if d_row == 0 and d_col == 0:
                        continue
                    n_row = row + d_row
                    n_col = col + d_col
                    if n_row < 0 or n_row >= rows or n_col < 0 or n_col >= cols:
                        all_neighbors_present = False
                        break
                    if not tile_mask[n_row, n_col]:
                        all_neighbors_present = False
                        break
                if not all_neighbors_present:
                    break

            valid_tiles[row, col] = all_neighbors_present

    return valid_tiles, boundary_tiles


# ======== CANDIDATE GENERATION ========

def generate_candidate_locations(
    df_aps: pd.DataFrame,
    tile_meters: float,
    conflictivity_threshold: float,
    radius_m: float,
    indoor_only: bool = True,
    neighbor_radius_tiles: int = 1,
    inner_clearance_m: float = 0.0,
) -> pd.DataFrame:
    """Generate candidate locations for new AP placement using tile-based approach.
    
    Args:
        df_aps: DataFrame with AP locations and conflictivity
        tile_meters: Tile size in meters
        conflictivity_threshold: Minimum conflictivity to consider
        radius_m: Interpolation radius in meters
        indoor_only: Whether to restrict to indoor locations
        neighbor_radius_tiles: Number of tile neighbors required
        inner_clearance_m: Inner clearance distance from existing APs
        
    Returns:
        DataFrame with candidate locations and scores
    """
    required_columns = {"lon", "lat", "conflictivity"}
    missing = required_columns - set(df_aps.columns)
    assert not missing, f"df_aps missing required columns: {missing}"

    if df_aps.empty:
        return pd.DataFrame(columns=["lon", "lat", "conflictivity", "min_dist_m", "reason"])

    coords = df_aps[["lon", "lat"]].to_numpy(dtype=float)  # type: ignore[call-overload]
    conflicts = df_aps["conflictivity"].to_numpy(dtype=float)  # type: ignore[call-overload]
    lons: FloatArray = coords[:, 0]
    lats: FloatArray = coords[:, 1]

    hull = compute_convex_hull_polygon(lons, lats)
    if hull is None:
        return pd.DataFrame(columns=["lon", "lat", "conflictivity", "min_dist_m", "reason"])

    lat0 = float(np.mean(lats))
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = max(1e-6, 111_320.0 * np.cos(np.deg2rad(lat0)))
    dlat = tile_meters / meters_per_deg_lat
    dlon = tile_meters / meters_per_deg_lon

    minx, miny, maxx, maxy = hull.bounds  # type: ignore[misc]
    lon_centers = np.arange(minx + dlon / 2.0, maxx, dlon)
    lat_centers = np.arange(miny + dlat / 2.0, maxy, dlat)

    if lon_centers.size == 0 or lat_centers.size == 0:
        return pd.DataFrame(columns=["lon", "lat", "conflictivity", "min_dist_m", "reason"])

    painted_mask = mask_points_in_polygon(lon_centers, lat_centers, hull)
    if not painted_mask.any():
        return pd.DataFrame(columns=["lon", "lat", "conflictivity", "min_dist_m", "reason"])

    lat_idx, lon_idx = np.nonzero(painted_mask)
    candidate_lons = lon_centers[lon_idx]
    candidate_lats = lat_centers[lat_idx]

    candidate_distances = haversine_m(
        candidate_lats[:, None],
        candidate_lons[:, None],
        lats[None, :],
        lons[None, :],
    )
    min_distances = candidate_distances.min(axis=1)  # type: ignore[union-attr]

    weights = np.maximum(0.0, 1.0 - candidate_distances / max(radius_m, 1e-6))
    weights[candidate_distances >= radius_m] = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        weighted_sum = (weights * conflicts[None, :]).sum(axis=1)
        denom = weights.sum(axis=1)
        predictions = np.divide(weighted_sum, denom, out=np.zeros_like(weighted_sum), where=denom > 0)

    valid_tiles_mask, boundary_tiles_mask = compute_tile_masks(painted_mask, neighbor_radius_tiles)
    candidate_valid = valid_tiles_mask[lat_idx, lon_idx]
    candidate_boundary = boundary_tiles_mask[lat_idx, lon_idx]

    inner_limit = max(0.0, radius_m - inner_clearance_m)
    inner_mask = min_distances < inner_limit

    final_mask = (
        candidate_valid
        & inner_mask
        & (predictions >= conflictivity_threshold)
        & (~candidate_boundary)
    )

    if not final_mask.any():
        return pd.DataFrame(columns=["lon", "lat", "conflictivity", "min_dist_m", "reason"])

    filtered_lon = candidate_lons[final_mask]
    filtered_lat = candidate_lats[final_mask]
    filtered_conf = predictions[final_mask]
    filtered_dist = min_distances[final_mask]

    candidates = pd.DataFrame(
        {
            "lon": filtered_lon,
            "lat": filtered_lat,
            "conflictivity": filtered_conf,
            "min_dist_m": filtered_dist,
            "reason": "high_conflictivity_interior_non_boundary",
        }
    )

    return candidates.reset_index(drop=True)


def generate_voronoi_candidates(
    scenarios: Sequence[tuple[Any, Path, datetime]],
    geo_df: pd.DataFrame,
    radius_m: float,
    conflictivity_threshold: float,
    tile_radius_clearance_m: float,
    merge_radius_m: float = 8.0,
    max_vertices_per_scenario: int = 60,
) -> pd.DataFrame:
    """Generate AP candidate locations using Voronoi vertices across multiple scenarios.
    
    Args:
        scenarios: List of (stress_level, snapshot_path, timestamp) tuples
        geo_df: DataFrame with geolocation data
        radius_m: Interpolation radius in meters
        conflictivity_threshold: Minimum conflictivity to consider
        tile_radius_clearance_m: Clearance from tile radius
        merge_radius_m: Distance to merge nearby vertices
        max_vertices_per_scenario: Maximum vertices to keep per scenario
        
    Returns:
        DataFrame with Voronoi-based candidate locations
    """
    try:
        from scipy.spatial import Voronoi  # type: ignore[import-untyped]
        from shapely.geometry import MultiPoint, Point  # type: ignore[import-untyped]
    except ImportError:
        st.error("SciPy or Shapely not available: install dependencies to enable Voronoi candidate mode.")
        return pd.DataFrame()

    # Import read_ap_snapshot from parent context - needs to be passed or imported
    from dashboard.data_io import read_ap_snapshot, extract_group
    
    records: list[Dict[str, Any]] = []
    effective_max_dist = max(0.0, radius_m - tile_radius_clearance_m)

    for profile, snap_path, snap_dt in scenarios:
        try:
            df_snap = read_ap_snapshot(snap_path, band_mode='worst')
            df_snap = df_snap.merge(geo_df, on='name', how='inner')
            
            if "group_code" not in df_snap.columns:
                df_snap["group_code"] = df_snap["name"].apply(extract_group)
            
            df_snap = df_snap[df_snap['group_code'] != 'SAB'].copy()

            # Ensure conflictivity is calculated
            if 'conflictivity' not in df_snap.columns:
                df_snap = recalculate_conflictivity(df_snap)

            if df_snap.empty:
                continue
            pts_xy = df_snap[['lon', 'lat']].to_numpy()  # type: ignore[call-overload]
            if len(pts_xy) < 3:
                continue
            vor = Voronoi(pts_xy)  # type: ignore[misc]
            hull_poly = MultiPoint([Point(xy) for xy in pts_xy]).convex_hull  # type: ignore[misc]
            lons = df_snap['lon'].to_numpy()  # type: ignore[call-overload]
            lats = df_snap['lat'].to_numpy()  # type: ignore[call-overload]
            cvals = df_snap['conflictivity'].to_numpy()  # type: ignore[call-overload]
            for vidx, (vx, vy) in enumerate(vor.vertices):
                p = Point(vx, vy)  # type: ignore[misc]
                if not hull_poly.contains(p):  # type: ignore[misc]
                    continue
                dists = haversine_m(vy, vx, lats, lons)
                d_min = float(np.atleast_1d(dists).min())
                if d_min >= effective_max_dist:
                    continue
                w = np.maximum(0.0, 1 - np.atleast_1d(dists) / radius_m)
                w[np.atleast_1d(dists) >= radius_m] = 0.0
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
                    'stress_profile': profile.value if hasattr(profile, 'value') else str(profile),
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
            d = haversine_m(row.lat, row.lon, cl['lat'], cl['lon'])
            if float(np.atleast_1d(d).item()) <= merge_r:
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
    scenario_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate metrics across scenarios.
    
    Args:
        lat: Latitude of candidate location
        lon: Longitude of candidate location
        base_conflictivity: Base conflictivity at location
        scenario_results: List of scenario result dictionaries
        
    Returns:
        Aggregated summary dictionary with mean/std/min/max scores
    """
    assert lat == lat and lon == lon, "Latitude and longitude must be finite"
    assert base_conflictivity >= 0.0, "Base conflictivity must be non-negative"
    assert scenario_results, "scenario_results must not be empty"

    scores: list[float] = []
    for result in scenario_results:
        composite = result.get('composite_score')
        assert composite is not None, "Each scenario result must include 'composite_score'"
        scores.append(float(composite))

    n_scenarios = len(scores)

    aggregated: Dict[str, Any] = {
        'lat': float(lat),
        'lon': float(lon),
        'base_conflictivity': float(base_conflictivity),
        'n_scenarios': n_scenarios,
        'final_score': float(np.mean(scores)),
        'score_std': float(np.std(scores)),
        'score_min': float(np.min(scores)),
        'score_max': float(np.max(scores)),
        'warnings': [],
    }

    metric_keys: Tuple[str, ...] = (
        'worst_ap_improvement_raw',
        'avg_reduction_raw',
        'num_improved',
        'new_ap_client_count',
    )
    for key in metric_keys:
        values = [float(r.get(key, 0.0)) for r in scenario_results]
        aggregated[f'{key}_mean'] = float(np.mean(values))

    all_warnings: list[str] = []
    for result in scenario_results:
        all_warnings.extend(result.get('warnings', []))

    warning_counts = Counter(all_warnings)
    aggregated['warnings'] = [
        f"{msg} (in {count}/{n_scenarios} scenarios)"
        for msg, count in warning_counts.most_common()
    ]

    by_profile: dict[str, list[float]] = {}
    for result in scenario_results:
        profile = result.get('stress_profile', 'unknown')
        profile_scores = by_profile.setdefault(profile, [])
        profile_scores.append(float(result.get('composite_score', 0.0)))

    for profile, profile_scores in by_profile.items():
        aggregated[f'score_{profile}'] = float(np.mean(profile_scores))

    return aggregated


def simulate_multiple_ap_additions(
    df_baseline: pd.DataFrame,
    points: List[Dict[str, float]],
    config: SimulationConfigType,
) -> pd.DataFrame:
    """Approximate combined effect of adding multiple APs by applying them sequentially.
    
    Args:
        df_baseline: Baseline network state
        points: List of placement point dictionaries with 'lat' and 'lon'
        config: Simulation configuration
        
    Returns:
        Updated DataFrame with all APs added
    """
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
