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
import time
import sys
_profile_start = time.perf_counter()
print(f"[PROFILE] Script Start: {_profile_start}", file=sys.stderr)

import math
import re
import sys
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import requests
from dotenv import load_dotenv
import os
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import unary_union, linemerge, snap
from matplotlib.path import Path as MplPath

# Import shared types and constants (only used ones)
from dashboard.types import (
    JOIN_M_DEFAULT,
    MAX_TILES_NO_LIMIT,
    SNAP_M_DEFAULT,
    TILE_M_FIXED,
    VOR_TOL_M_FIXED,
    W_AIR,
    W_CL,
    W_CPU,
    W_MEM,
)

def _log_profile(msg):
    if "_profile_start" in globals():
        print(f"[PROFILE] {time.perf_counter() - _profile_start:.4f}s: {msg}", file=sys.stderr)

_log_profile("Imports Complete")

SRC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SRC_ROOT.parent

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load environment variables
load_dotenv()
_log_profile("Env Loaded")

from dashboard.data_io import (
    AP_DIR,
    GEOJSON_PATH,
    extract_group,
    find_snapshot_files,
    read_ap_snapshot as load_ap_snapshot,
    read_geoloc_points,
)
_log_profile("Data IO Imports Complete")

# --- Caching Wrappers ---
@st.cache_data(show_spinner=False)
def get_cached_snapshots(ap_dir: Path) -> List[Tuple[Path, datetime]]:
    """Cached wrapper for finding snapshot files."""
    return find_snapshot_files(ap_dir)

@st.cache_data(show_spinner=False)
def get_cached_geoloc(geojson_path: Path) -> pd.DataFrame:
    """Cached wrapper for reading geolocation points."""
    return read_geoloc_points(geojson_path)
# ------------------------

if TYPE_CHECKING:
    pass
else:
    SimulationConfigType = Any
    StressLevelType = Enum
    CompositeScorerType = Any
    NeighborhoodOptimizationModeType = Enum
    StressProfilerType = Any

# Import simulator components (try-except for graceful degradation)
simulator_available: bool = False
try:
    from experiments.polcorresa.simulator.config import SimulationConfig
    from experiments.polcorresa.simulator.config import StressLevel
    from experiments.polcorresa.simulator.stress_profiler import StressProfiler
    from experiments.polcorresa.simulator.scoring import CompositeScorer, NeighborhoodOptimizationMode
    simulator_available = True
except ImportError:
    simulator_available = False

    class _FallbackStressLevel(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"

        def __str__(self) -> str:
            return self.value

    @dataclass
    class _FallbackSimulationConfig:
        reference_distance_m: float = 1.0
        path_loss_exponent: float = 3.5
        reference_rssi_dbm: float = -40.0
        min_rssi_dbm: float = -85.0
        max_offload_fraction: float = 0.25
        sticky_client_fraction: float = 0.15
        max_clients_per_ap: int = 35
        target_util_2g: float = 35.0
        target_util_5g: float = 25.0
        interference_radius_m: float = 50.0
        cca_increase_factor: float = 0.15
        indoor_only: bool = True
        conflictivity_threshold_placement: float = 0.6
        snapshots_per_profile: int = 5
        target_stress_profile: Optional[_FallbackStressLevel] = _FallbackStressLevel.HIGH
        weight_worst_ap: float = 0.30
        weight_average: float = 0.30
        weight_coverage: float = 0.20
        weight_neighborhood: float = 0.20
        utilization_threshold_critical: float = 70.0
        utilization_threshold_high: float = 50.0

        def get_weights_dict(self) -> dict[str, float]:
            return {
                "worst_ap": self.weight_worst_ap,
                "average": self.weight_average,
                "coverage": self.weight_coverage,
                "neighborhood": self.weight_neighborhood,
            }

    class _UnavailableComponent:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "Simulator components are unavailable. Install simulator dependencies to enable this feature."
            )

        def __call__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "Simulator components are unavailable. Install simulator dependencies to enable this feature."
            )

        def get_representative_snapshots(self, *args: object, **kwargs: object) -> list[Path]:
            raise RuntimeError(
                "Simulator components are unavailable. Install simulator dependencies to enable this feature."
            )

        def classify_snapshots(self, *args: object, **kwargs: object) -> Dict[str, Any]:
            raise RuntimeError(
                "Simulator components are unavailable. Install simulator dependencies to enable this feature."
            )

        def get_profile_statistics(self, *args: object, **kwargs: object) -> Dict[str, Any]:
            raise RuntimeError(
                "Simulator components are unavailable. Install simulator dependencies to enable this feature."
            )

    class _FallbackNeighborhoodMode(Enum):
        BALANCED = "balanced"

    def sim_haversine_m(*args: object, **kwargs: object) -> float:
        raise RuntimeError("Simulator components are unavailable. Install simulator dependencies to enable this feature.")

    StressLevel = _FallbackStressLevel
    SimulationConfig = _FallbackSimulationConfig
    StressProfiler = _UnavailableComponent  # type: ignore[assignment]
    CompositeScorer = _UnavailableComponent  # type: ignore[assignment]
    NeighborhoodOptimizationMode = _FallbackNeighborhoodMode

# Check for scipy Voronoi
has_scipy_voronoi: bool = False
try:
    from scipy.spatial import Voronoi  # type: ignore[import-not-found]
    has_scipy_voronoi = True
except Exception:
    has_scipy_voronoi = False


class DashboardMode(Enum):
    AI_HEATMAP = "ai-heatmap"
    VORONOI = "voronoi"
    SIMULATOR = "simulator"

    @classmethod
    def label_map(cls) -> dict[str, str]:
        return {
            cls.AI_HEATMAP.value: "AI Heatmap",
            cls.VORONOI.value: "Voronoi",
            cls.SIMULATOR.value: "Simulator",
        }


def _mode_to_slug(mode: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", mode.lower()).strip("-")
    labels = DashboardMode.label_map()
    for slug, label in labels.items():
        if normalized == slug or mode == label:
            return slug
    if normalized == "ai-heatmap":
        return DashboardMode.AI_HEATMAP.value
    return normalized or DashboardMode.AI_HEATMAP.value


def _slug_to_label(slug: str, available_modes: list[str]) -> str | None:
    if not slug:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
    labels = DashboardMode.label_map()
    label = labels.get(normalized)
    if label and label in available_modes:
        return label
    for candidate in available_modes:
        candidate_slug = re.sub(r"[^a-z0-9]+", "-", candidate.lower()).strip("-")
        if candidate_slug == normalized:
            return candidate
    return None


def _get_query_param_value(key: str) -> str | None:
    # Check environment variable first for profiling/headless mode
    env_val = os.environ.get(key)
    if env_val:
        return env_val
        
    raw_value = st.query_params.get(key)
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return raw_value[0] if raw_value else None
    return str(raw_value)


def _set_query_param_value(key: str, value: str) -> None:
    current_params = {k: v for k, v in st.query_params.items()}
    if current_params.get(key) == value:
        return
    current_params[key] = value
    st.query_params = current_params


def resolve_dashboard_mode(
    *,
    available_modes: list[str],
    simulator_enabled: bool,
) -> str:
    assert available_modes, "available_modes must not be empty"
    query_choice = _get_query_param_value("dashboard")
    env_choice = os.getenv("dashboard") or os.getenv("DASHBOARD")
    preferred_order = [query_choice, env_choice]
    for preferred in preferred_order:
        if preferred is None:
            continue
        label = _slug_to_label(preferred, available_modes)
        if label:
            if label == "Simulator" and not simulator_enabled:
                continue
            return label
    
    return available_modes[0]


def resolve_stress_profiles(
    target: Any,  # StressLevel or None
    stats: Dict[Any, Dict[str, float]],  # Dict[StressLevel, Dict[str, float]]
) -> Tuple[List[Any], Any, Optional[str]]:  # Returns (List[StressLevel], StressLevel | None, str | None)
    """Pick stress profiles to simulate, falling back gracefully when data is missing.
    
    Args:
        target: Target stress level (StressLevel enum member or None)
        stats: Statistics dictionary keyed by StressLevel
        
    Returns:
        Tuple of (profiles_list, effective_target, error_message)
    """
    priority: list[Any] = [
        StressLevel.CRITICAL,  # type: ignore[attr-defined]
        StressLevel.HIGH,  # type: ignore[attr-defined]
        StressLevel.MEDIUM,  # type: ignore[attr-defined]
        StressLevel.LOW  # type: ignore[attr-defined]
    ]
    counts: dict[Any, float] = {lvl: stats.get(lvl, {}).get('count', 0) for lvl in priority}
    available: list[Any] = [lvl for lvl in priority if counts.get(lvl, 0) > 0]

    if target is None:
        if not available:
            return [], None, "No snapshots available in any stress profile."
        effective_target: Any = None if len(available) > 1 else available[0]
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

def _enrich_conflictivity_metrics(df: pd.DataFrame, band_mode: str, p95_clients: float) -> pd.DataFrame:
    w_2g = 0.6
    w_5g = 0.4

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
    else:
        df["airtime_score"] = np.nanmax(
            np.vstack([df["air_s_2g"].fillna(-1), df["air_s_5g"].fillna(-1)]), axis=0
        )
        df["airtime_score"] = df["airtime_score"].where(df["airtime_score"] >= 0, np.nan)

    df["client_score"] = df["client_count"].apply(lambda n: client_pressure_score(n, p95_clients))
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


def read_ap_snapshot(path: Path, band_mode: str = "worst") -> pd.DataFrame:
    base_df = load_ap_snapshot(path, band_mode)
    p95_clients = float(np.nanpercentile(base_df["client_count"].fillna(0), 95)) if len(base_df) else 1.0
    return _enrich_conflictivity_metrics(base_df, band_mode, p95_clients)


# ======== DASHBOARD MODE MODULES ========
# Import modular dashboard components
from dashboard.ai_heatmap import (
    create_optimized_heatmap,
    HeatmapConfig,
)


# ======== VORONOI MODE FUNCTIONS (from voronoi_viz module) ========

# Import optimized Voronoi functions from dedicated module
from dashboard.voronoi_viz import (
    haversine_m as _haversine_m,
    compute_convex_hull_polygon as _compute_convex_hull_polygon,
    inverted_weighted_voronoi_edges as _inverted_weighted_voronoi_edges,
    top_conflictive_voronoi_vertices as _top_conflictive_voronoi_vertices,
    uab_tiled_choropleth_layer as _uab_tiled_choropleth_layer,
    compute_coverage_regions as _compute_coverage_regions,
)

# ======== SIMULATOR MODE FUNCTIONS (from simulator_viz module) ========

# Import simulator functions from dedicated module
from dashboard.simulator_viz import (
    simulate_ap_addition,
    generate_candidate_locations,
    generate_voronoi_candidates,
    aggregate_scenario_results,
    simulate_multiple_ap_additions,
)


# Note: _inverted_weighted_voronoi_edges and _top_conflictive_voronoi_vertices
# are now imported from dashboard.voronoi_viz module (see imports above)

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

snapshots = get_cached_snapshots(AP_DIR)
_log_profile("Snapshots Found")
if not snapshots:
    st.warning("No AP snapshots found in realData/ap. Please add AP-info-v2-*.json files.")
    st.stop()

geo_df = get_cached_geoloc(GEOJSON_PATH)
_log_profile("GeoJSON Loaded")

# Sidebar
mode_options = ["AI Heatmap", "Voronoi"]
if simulator_available:
    mode_options.append("Simulator")
default_mode = resolve_dashboard_mode(
    available_modes=mode_options,
    simulator_enabled=simulator_available,
)
if "selected_dashboard" not in st.session_state:
    st.session_state.selected_dashboard = default_mode
expected_slug = _mode_to_slug(st.session_state.selected_dashboard)
current_slug = _get_query_param_value("dashboard")
if current_slug != expected_slug:
    _set_query_param_value("dashboard", expected_slug)
with st.sidebar:
    st.header("Visualization Mode")
    
    current_index = mode_options.index(st.session_state.selected_dashboard)
    viz_mode = st.radio(
        "Select Mode",
        options=mode_options,
        index=current_index,
        help="AI Heatmap: Click APs for AINA analysis | Voronoi: Interpolated surfaces | Simulator: AP placement optimization",
        key="viz_mode_radio",
    )
    if viz_mode != st.session_state.selected_dashboard:
        st.session_state.selected_dashboard = viz_mode
        _set_query_param_value("dashboard", _mode_to_slug(viz_mode))
    
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
        
        st.divider()
        st.header("Voronoi ponderat")
        show_awvd = True
        weight_source = st.selectbox(
            "Base connectivitat (per invertir)",
            ["conflictivity", "client_count", "max_radio_util", "airtime_score"],
            index=0,
            help="Es normalitza i s'inverteix: pes = 1 - norm(col)."
        )
        snap_m = float(max(1.5, TILE_M_FIXED * 0.2))
        join_m = float(max(3.0, TILE_M_FIXED * 0.6))
        show_hot_vertex = st.checkbox("Marcar punt més conflictiu (vertex Voronoi)", value=False,
                                    help="Evalua vertices del Voronoi i marca el de major conflictivitat.")
        min_conf = 0.0
        top_n = 15
    else:  # Simulator
        radius_m = 25
        value_mode = "conflictivity"
        # Using module-level constants TILE_M_FIXED and MAX_TILES_NO_LIMIT
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
                        "Interference radius (m)", 10, 80, 25, 5,
                        help="How far the new AP affects neighbors (typical: 25m)"
                    )
                    
                    sim_cca_increase = st.slider(
                        "Co-channel interference", 0.01, 0.30, 0.02, 0.01,
                        help="How much neighbors' utilization increases (typical: 2%)"
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
                sim_interference_radius = 25
                sim_cca_increase = 0.02
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
        if not has_scipy_voronoi:
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

- band_mode="avg": mitjana ponderada (2,4 GHz 0,6, 5 GHz 0:40%)

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
    st.info(
        "ℹ️ **Guide:** The **colored circles** on the map represent Access Points (APs).\n\n"
        "Click on any **conflicting AP** (orange/red) to ask **AIna** for a diagnosis: "
        "she will analyze if the issue is due to **device saturation** (CPU/RAM) or **airtime congestion**."
    )

    config = HeatmapConfig(
        min_conflictivity=min_conf,
        marker_radius=5,
        default_zoom=15,
    )
    
    fig = create_optimized_heatmap(
        df=map_df,
        center_lat=center_lat,
        center_lon=center_lon,
        config=config,
    )

    fig.update_layout(clickmode='event+select')
    selected_points = st.plotly_chart(
        fig,
        width="stretch",
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
    st.info(
        "ℹ️ **Guide:** This map shows the **interpolated conflictivity** across the campus.\n\n"
        "The **blue lines** (Voronoi edges) indicate boundaries between AP coverage areas. "
        "Use this to identify **'dark zones'** or gaps where signal quality might degrade between access points."
    )

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
        fig.add_trace(go.Scattermap(
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
            regions = _compute_coverage_regions(
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
                fig.add_trace(go.Scattermap(
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
                        fig.add_trace(go.Scattermap(
                            lon=[hv_lon], lat=[hv_lat], mode='markers',
                            marker=dict(size=24, color='#ffffff', symbol='circle', opacity=0.95),
                            hoverinfo='skip', showlegend=False
                        ))
                        fig.add_trace(go.Scattermap(
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
                            fig.add_trace(go.Scattermap(
                                lon=lons, lat=lats, mode='markers',
                                marker=dict(size=20, color='#ffffff', symbol='circle', opacity=0.95),
                                hoverinfo='skip', showlegend=False
                            ))
                            fig.add_trace(go.Scattermap(
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
                    fig.add_trace(go.Scattermap(
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
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=15,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02)
    )

    st.plotly_chart(fig, width="stretch")

    # Table with top hotspot vertices
    try:
        if show_awvd and show_hot_vertex and hot_vertices_info:
            st.subheader("Top vertices Voronoi més conflictius")
            top_df = pd.DataFrame(hot_vertices_info)
            top_df["score"] = top_df["score"].map(lambda x: f"{x:.3f}")
            st.dataframe(top_df, width="stretch", hide_index=True)
    except Exception:
        pass

else:  # Simulator
    # ========== SIMULATOR MODE ==========
    st.info(
        "ℹ️ **Guide:** This tool **automatically simulates** the addition of a new AP to **optimize network coverage**.\n\n"
        "The simulation runs with default parameters. You can adjust the **Candidate Generation Mode** (Tile-based or Voronoi) and other settings in the sidebar to refine the search for the best location."
    )

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
        fig.add_trace(go.Scattermap(
            lat=uab_df['lat'], lon=uab_df['lon'], mode='markers',
            marker=dict(size=7, color='black', opacity=0.7),
            text=uab_df['name'], name="UAB APs",
            hovertemplate='<b>%{text}</b><br>Conflictivity: %{customdata:.2f}<extra></extra>',
            customdata=uab_df['conflictivity']
        ))

    # Overlay Voronoi candidate markers if discovered
    if 'voronoi_candidates' in st.session_state and not st.session_state.voronoi_candidates.empty:
        vor_df = st.session_state.voronoi_candidates
        fig.add_trace(go.Scattermap(
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
    if run_simulation and st.session_state.get('run_sim', False) and simulator_available:
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
                                fig.add_trace(go.Scattermap(
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
                                        'Impact Efficiency: %{customdata[6]:.3f}<br>'
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
                                        [best.get('impact_efficiency_mean', 0.0)],
                                    ]),
                                ))
                            
                            for idx, row in results_df.iterrows():
                                rank = idx + 1
                                
                                fig.add_trace(go.Scattermap(
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
                                width="stretch"
                            )
                            
                            best = results_df.iloc[0]
                            
                            # Sync Voronoi selection with Best Result if applicable
                            # This ensures the "Best Location" is pre-selected in the table below
                            if 'voronoi_candidates' in st.session_state and not st.session_state.voronoi_candidates.empty:
                                vor_df_sync = st.session_state.voronoi_candidates
                                # Find integer index of the match
                                match_idx = np.where(
                                    (np.abs(vor_df_sync['lat'].values - best['lat']) < 1e-5) & 
                                    (np.abs(vor_df_sync['lon'].values - best['lon']) < 1e-5)
                                )[0]
                                if len(match_idx) > 0:
                                    st.session_state['vor_selected_rows'] = {int(match_idx[0])}

                            # Main Score - Prominently displayed
                            impact_eff = best.get('impact_efficiency_mean', 0.0)
                            st.metric(
                                "Impact Efficiency", 
                                f"{impact_eff * 100:.1f}%", 
                                help="Average improvement on APs that actually improved significantly (>0.05). This is the primary measure of placement quality."
                            )
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric(
                                    "Best Score", 
                                    f"{best['final_score']:.3f}", 
                                    delta=f"±{best['score_std']:.3f}",
                                    help="Composite score (0-1) combining worst-AP improvement, average reduction, coverage, and neighborhood health."
                                )
                            with col2:
                                st.metric(
                                    "Avg Reduction", 
                                    f"{best['avg_reduction_raw_mean']:.3f}",
                                    help="Average reduction in conflictivity across the entire network."
                                )
                            with col3:
                                st.metric(
                                    "Worst AP Improvement", 
                                    f"{best['worst_ap_improvement_raw_mean']:.3f}",
                                    help="Reduction in conflictivity for the most stressed AP in the network."
                                )
                            with col4:
                                st.metric(
                                    "Clients on New AP", 
                                    f"{int(best['new_ap_client_count_mean'])}",
                                    help="Projected number of clients that will connect to this new AP."
                                )
                            
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
        map=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=15,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=700,
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02)
    )

    st.plotly_chart(fig, width="stretch")
    
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
                fig_sim.add_trace(go.Scattermap(
                    lat=existing_aps['lat'], lon=existing_aps['lon'], mode='markers',
                    marker=dict(size=7, color='black', opacity=0.7),
                    text=existing_aps['name'], name="UAB APs",
                    hovertemplate='<b>%{text}</b><br>Conflictivity: %{customdata:.2f}<extra></extra>',
                    customdata=existing_aps['conflictivity']
                ))
        
        # Add new AP markers if available
        if 'new_node_markers' in st.session_state and st.session_state['new_node_markers']:
            nn = st.session_state['new_node_markers']
            fig_sim.add_trace(go.Scattermap(
                lat=[p['lat'] for p in nn],
                lon=[p['lon'] for p in nn],
                mode='markers+text',
                marker=dict(size=10, color='skyblue', opacity=0.95),
                text=[p.get('label', 'AP-NEW') for p in nn],
                textposition='top center',
                name='New APs (simulated)',
                hovertemplate="<b>%{text}</b><extra>(%{lat:.5f}, %{lon:.5f})</extra>"
            ))
        
        fig_sim.update_layout(
            map=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lon),
                zoom=15,
            ),
            margin=dict(l=10, r=10, t=30, b=10),
            height=700,
            legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02)
        )
        
        st.subheader("Simulated Map (with new APs)")
        st.plotly_chart(fig_sim, width="stretch")
    
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
            width="stretch",
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
                    width="stretch",
                    hide_index=True
                )
        
        st.caption(f"💡 **{len(vor_df)} candidates detected**. Freq = how many scenarios this vertex appeared in. Higher frequency = more stable location.")
