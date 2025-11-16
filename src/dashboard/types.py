"""Shared type definitions for dashboard modules.

This module provides common TypedDicts, type aliases, and constants used across
all dashboard visualization components (ai_heatmap, voronoi_viz, simulator_viz,
integrated_dashboard).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, NotRequired, TypedDict

import numpy as np
from numpy.typing import NDArray

# ======== NUMPY TYPE ALIASES ========

FloatArray = NDArray[np.floating[Any]]
"""Type alias for NumPy arrays of floating point numbers."""

BoolArray = NDArray[np.bool_]
"""Type alias for NumPy arrays of boolean values."""


# ======== VORONOI & SIMULATOR TYPEDDICTS ========


class VoronoiCandidateRecord(TypedDict):
    """Record describing a candidate AP placement location from Voronoi analysis.
    
    Attributes:
        lon: Longitude coordinate of candidate location.
        lat: Latitude coordinate of candidate location.
        conflictivity: Computed conflictivity score at this location.
        min_dist_m: Minimum distance in meters to nearest existing AP.
        scenario_ts: Timestamp of the scenario data used for analysis.
        stress_profile: Name of the stress profile used for weighting.
    """

    lon: float
    lat: float
    conflictivity: float
    min_dist_m: float
    scenario_ts: datetime
    stress_profile: str


class ScenarioScoreMetrics(TypedDict):
    """Comprehensive metrics for evaluating AP placement scenario outcomes.
    
    Required metrics:
        composite_score: Overall quality score for the placement scenario.
        warnings: List of warning messages about scenario issues.
        avg_conflictivity_before: Mean conflictivity across all APs before change.
        avg_conflictivity_after: Mean conflictivity across all APs after change.
        avg_reduction: Average absolute reduction in conflictivity.
        avg_reduction_pct: Average percentage reduction in conflictivity.
        worst_ap_conflictivity_before: Maximum AP conflictivity before change.
        worst_ap_conflictivity_after: Maximum AP conflictivity after change.
        worst_ap_improvement: Improvement in worst-case AP conflictivity.
        num_high_conflict_before: Count of high-conflict APs before change.
        num_high_conflict_after: Count of high-conflict APs after change.
        new_ap_client_count: Number of clients connected to new AP.
        new_ap_util_2g: Utilization percentage of new AP's 2.4GHz band.
        new_ap_util_5g: Utilization percentage of new AP's 5GHz band.
    
    Optional metrics:
        worst_ap_improvement_raw: Raw improvement value for worst AP.
        avg_reduction_raw: Raw average reduction value.
        num_improved: Number of APs showing improvement.
        new_ap_client_count_mean: Mean client count for new AP.
        stress_profile: Name of stress profile used.
        timestamp: Timestamp of scenario evaluation.
    """

    composite_score: float
    warnings: List[str]
    avg_conflictivity_before: float
    avg_conflictivity_after: float
    avg_reduction: float
    avg_reduction_pct: float
    worst_ap_conflictivity_before: float
    worst_ap_conflictivity_after: float
    worst_ap_improvement: float
    num_high_conflict_before: int
    num_high_conflict_after: int
    new_ap_client_count: float
    new_ap_util_2g: float
    new_ap_util_5g: float
    worst_ap_improvement_raw: NotRequired[float]
    avg_reduction_raw: NotRequired[float]
    num_improved: NotRequired[float]
    new_ap_client_count_mean: NotRequired[float]
    stress_profile: NotRequired[str]
    timestamp: NotRequired[datetime]


class PlacementPoint(TypedDict, total=False):
    """Geographic point for AP placement visualization.
    
    All fields are optional to support partial data scenarios.
    
    Attributes:
        lat: Latitude coordinate.
        lon: Longitude coordinate.
        label: Optional text label for the point.
    """

    lat: float
    lon: float
    label: NotRequired[str]


AggregatedScenarioSummary = Dict[str, float | int | List[str]]
"""Summary statistics aggregated across multiple placement scenarios.

Dictionary keys are metric names, values can be numeric or lists of strings
(e.g., warnings, stress profiles used).
"""


# ======== CONFLICTIVITY WEIGHTING CONSTANTS ========

# Default conflictivity weighting profile
W_AIR: float = 0.75
"""Weight for air utilization in conflictivity calculation."""

W_CL: float = 0.15
"""Weight for client count in conflictivity calculation."""

W_CPU: float = 0.05
"""Weight for CPU usage in conflictivity calculation."""

W_MEM: float = 0.05
"""Weight for memory usage in conflictivity calculation."""

# Simulator conflictivity weights (alternative profile)
W_AIR_SIM: float = 0.85
"""Weight for air utilization in simulator conflictivity calculation."""

W_CL_SIM: float = 0.10
"""Weight for client count in simulator conflictivity calculation."""

W_CPU_SIM: float = 0.02
"""Weight for CPU usage in simulator conflictivity calculation."""

W_MEM_SIM: float = 0.03
"""Weight for memory usage in simulator conflictivity calculation."""


# ======== VORONOI & SPATIAL CONSTANTS ========

TILE_M_FIXED: float = 7.0
"""Fixed tile size in meters for spatial discretization."""

MAX_TILES_NO_LIMIT: int = 1_000_000_000
"""Maximum number of tiles when no limit is desired."""

VOR_TOL_M_FIXED: float = 24.0
"""Voronoi tolerance in meters for edge snapping."""

SNAP_M_DEFAULT: float = 1.5
"""Default snap distance in meters for point matching."""

JOIN_M_DEFAULT: float = 4.2
"""Default join distance in meters for merging nearby points."""
