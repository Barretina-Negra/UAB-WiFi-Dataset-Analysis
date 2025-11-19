"""
Simulation configuration and constants.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StressLevel(Enum):
    """Temporal stress classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SimulationConfig:
    """
    Configuration for multi-scenario AP placement simulation.
    
    Physics Parameters:
    -------------------
    path_loss_exponent : float
        Indoor RF propagation path loss exponent (default: 3.5)
        - 2.0 = free space
        - 3.5 = typical indoor (recommended)
        - 4.0 = heavy walls/obstacles
    
    reference_distance_m : float
        Reference distance for RSSI calculation (default: 1.0m)
    
    reference_rssi_dbm : float
        RSSI at reference distance (default: -30 dBm)
    
    min_rssi_dbm : float
        Minimum signal strength for client association (default: -75 dBm)
    
    Interference Parameters:
    ------------------------
    interference_radius_m : float
        Range of co-channel interference (CCA busy) (default: 30m)
        # Reduced from 50m to 30m to match typical CCA threshold (~-82dBm)
        # with path_loss_exponent=3.5
    
    cca_increase_factor : float
        How much new AP increases neighbor utilization (default: 0.05 = 5%)
        # Reduced from 0.15 to 0.05 to account for auto-channel selection
        # and avoid penalizing neighbors too harshly.
    
    channel_overlap_prob_2g : float
        Probability that a neighbor is on the same 2.4GHz channel (default: 0.33)
        # 1/3 chance (Channels 1, 6, 11)
    
    channel_overlap_prob_5g : float
        Probability that a neighbor is on the same 5GHz channel (default: 0.10)
        # ~1/10 chance (depending on width)
    
    Client Redistribution:
    ----------------------
    max_offload_fraction : float
        Maximum fraction of clients that can move from one AP (default: 0.5)
    
    sticky_client_fraction : float
        Fraction of clients that won't roam (device inertia) (default: 0.3)
    
    handover_margin_db : float
        Signal strength hysteresis for switching APs (default: 3.0 dB)
    
    AP Characteristics:
    -------------------
    max_clients_per_ap : int
        Soft limit for client association (default: 50)
    
    target_util_2g : float
        Target 2.4 GHz utilization for new AP (default: 40%)
    
    target_util_5g : float
        Target 5 GHz utilization for new AP (default: 50%)
    
    Placement Constraints:
    ----------------------
    indoor_only : bool
        Only consider indoor locations (within 20m of existing AP) (default: True)
    
    min_distance_to_ap_m : float
        Minimum distance from existing APs (default: 10m)
    
    max_distance_to_ap_m : float
        Maximum distance from existing APs for indoor filter (default: 20m)
    
    Scenario Parameters:
    --------------------
    snapshots_per_profile : int
        Number of snapshots to test per stress profile (default: 5)
    
    target_stress_profile : Optional[StressLevel]
        Which stress profile to optimize for (None = all) (default: HIGH)
    
    Scoring Weights:
    ----------------
    weight_worst_ap : float
        Weight for worst-case AP improvement (default: 0.30)
    
    weight_average : float
        Weight for average conflictivity reduction (default: 0.30)
    
    weight_coverage : float
        Weight for coverage improvement (# of APs improved) (default: 0.20)
    
    weight_neighborhood : float
        Weight for neighborhood health (default: 0.20)
    
    Stress Classification:
    ----------------------
    utilization_threshold_critical : float
        Avg utilization above this = CRITICAL (default: 70%)
    
    utilization_threshold_high : float
        Avg utilization above this = HIGH (default: 50%)
    
    conflictivity_threshold_placement : float
        Minimum conflictivity to consider for placement (default: 0.6)
    """
    
    # Physics
    path_loss_exponent: float = 3.5
    reference_distance_m: float = 1.0
    reference_rssi_dbm: float = -30.0
    min_rssi_dbm: float = -75.0
    
    # Interference
    interference_radius_m: float = 25.0
    cca_increase_factor: float = 0.02
    channel_overlap_prob_2g: float = 0.33
    channel_overlap_prob_5g: float = 0.10
    
    # Redistribution
    max_offload_fraction: float = 0.5
    sticky_client_fraction: float = 0.3
    handover_margin_db: float = 3.0
    
    # AP characteristics
    max_clients_per_ap: int = 50
    target_util_2g: float = 40.0
    target_util_5g: float = 50.0
    
    # Placement constraints
    indoor_only: bool = True
    min_distance_to_ap_m: float = 10.0
    max_distance_to_ap_m: float = 20.0
    
    # Scenario parameters
    snapshots_per_profile: int = 5
    target_stress_profile: Optional[StressLevel] = StressLevel.HIGH
    
    # Scoring weights (must sum to 1.0)
    weight_worst_ap: float = 0.30
    weight_average: float = 0.30
    weight_coverage: float = 0.20
    weight_neighborhood: float = 0.20
    
    # Stress classification
    utilization_threshold_critical: float = 70.0
    utilization_threshold_high: float = 50.0
    conflictivity_threshold_placement: float = 0.6
    
    def __post_init__(self):
        """Validate configuration."""
        total_weight = (
            self.weight_worst_ap +
            self.weight_average +
            self.weight_coverage +
            self.weight_neighborhood
        )
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Scoring weights must sum to 1.0 (got {total_weight})")
    
    def get_weights_dict(self) -> dict:
        """Return scoring weights as a dictionary."""
        return {
            'worst_ap': self.weight_worst_ap,
            'average': self.weight_average,
            'coverage': self.weight_coverage,
            'neighborhood': self.weight_neighborhood,
        }