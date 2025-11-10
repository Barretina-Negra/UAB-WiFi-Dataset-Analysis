"""
Composite scoring with neighborhood optimization.
"""

from enum import Enum
from typing import Dict
import numpy as np


class NeighborhoodOptimizationMode(Enum):
    """How to compute neighborhood health score."""
    IMPROVED_FRACTION = "improved_fraction"  # % of neighbors that improve
    AVG_IMPROVEMENT = "avg_improvement"      # Average improvement of neighbors
    MIN_IMPROVEMENT = "min_improvement"      # Worst neighbor improvement (ensure no degradation)
    BALANCED = "balanced"                    # Combination of above


class CompositeScorer:
    """
    Multi-objective scoring with neighborhood optimization.
    
    Composite Score = w1×worst_AP + w2×average + w3×coverage + w4×neighborhood
    
    where:
    - worst_AP: Improvement of the most overloaded AP (0-1, normalized)
    - average: Average conflictivity reduction across all APs (0-1)
    - coverage: Fraction of APs that improve (0-1)
    - neighborhood: Health of APs within interference radius (0-1)
    """
    
    def __init__(
        self,
        weight_worst_ap: float = 0.30,
        weight_average: float = 0.30,
        weight_coverage: float = 0.20,
        weight_neighborhood: float = 0.20,
        neighborhood_mode: NeighborhoodOptimizationMode = NeighborhoodOptimizationMode.BALANCED,
        interference_radius_m: float = 50.0,
    ):
        """
        Args:
            weight_worst_ap: Weight for worst-case improvement
            weight_average: Weight for average reduction
            weight_coverage: Weight for fraction of APs improved
            weight_neighborhood: Weight for neighborhood health
            neighborhood_mode: How to compute neighborhood score
            interference_radius_m: Define "neighborhood" as APs within this radius
        """
        self.w_worst = weight_worst_ap
        self.w_avg = weight_average
        self.w_cov = weight_coverage
        self.w_neigh = weight_neighborhood
        self.neigh_mode = neighborhood_mode
        self.radius_m = interference_radius_m
        
        # Validate weights
        total = self.w_worst + self.w_avg + self.w_cov + self.w_neigh
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0 (got {total})")
    
    def compute_component_scores(
        self,
        baseline_conflictivity: np.ndarray,
        updated_conflictivity: np.ndarray,
        neighbor_mask: np.ndarray,
        target_improvement: float = 0.3,  # Target worst-AP improvement
        target_avg_reduction: float = 0.1,  # Target average reduction
    ) -> Dict[str, float]:
        """
        Compute individual score components.
        
        Args:
            baseline_conflictivity: Array of original conflictivity values
            updated_conflictivity: Array of conflictivity after adding new AP
            neighbor_mask: Boolean array indicating which APs are neighbors
            target_improvement: Normalize worst-AP score by this target
            target_avg_reduction: Normalize average score by this target
        
        Returns:
            Dictionary with component scores (all in [0, 1])
        """
        # Deltas
        delta = baseline_conflictivity - updated_conflictivity
        
        # 1. Worst-AP improvement
        worst_ap_improvement = float(delta.max())
        worst_score = min(1.0, worst_ap_improvement / target_improvement)
        worst_score = max(0.0, worst_score)  # Clamp to [0, 1]
        
        # 2. Average reduction
        avg_reduction = float(delta.mean())
        avg_score = min(1.0, avg_reduction / target_avg_reduction)
        avg_score = max(0.0, avg_score)
        
        # 3. Coverage (fraction of APs that improve by at least 0.05)
        num_improved = int((delta > 0.05).sum())
        total_aps = len(baseline_conflictivity)
        coverage_score = num_improved / total_aps if total_aps > 0 else 0.0
        
        # 4. Neighborhood health
        neighborhood_score = self._compute_neighborhood_score(delta, neighbor_mask)
        
        return {
            'worst_ap': worst_score,
            'average': avg_score,
            'coverage': coverage_score,
            'neighborhood': neighborhood_score,
            
            # Raw values for reporting
            'worst_ap_improvement_raw': worst_ap_improvement,
            'avg_reduction_raw': avg_reduction,
            'num_improved': num_improved,
        }
    
    def _compute_neighborhood_score(
        self,
        delta: np.ndarray,
        neighbor_mask: np.ndarray,
    ) -> float:
        """
        Compute neighborhood health score based on selected mode.
        
        Args:
            delta: Conflictivity reduction (positive = improvement)
            neighbor_mask: Boolean array for neighbors
        
        Returns:
            Neighborhood score in [0, 1]
        """
        if not np.any(neighbor_mask):
            return 0.5  # Neutral if no neighbors
        
        neighbor_deltas = delta[neighbor_mask]
        
        if self.neigh_mode == NeighborhoodOptimizationMode.IMPROVED_FRACTION:
            # Fraction of neighbors that improve (delta > 0)
            improved = (neighbor_deltas > 0).sum()
            total = len(neighbor_deltas)
            return float(improved / total) if total > 0 else 0.0
        
        elif self.neigh_mode == NeighborhoodOptimizationMode.AVG_IMPROVEMENT:
            # Average improvement (clamped to [0, 1])
            avg_improvement = float(neighbor_deltas.mean())
            # Map to [0, 1]: assume 0.15 improvement = perfect score
            return np.clip(avg_improvement / 0.15, 0.0, 1.0)
        
        elif self.neigh_mode == NeighborhoodOptimizationMode.MIN_IMPROVEMENT:
            # Worst neighbor improvement (ensure no one degrades badly)
            min_improvement = float(neighbor_deltas.min())
            # Map: -0.1 (10% worse) = 0, 0.0 (no change) = 0.5, +0.1 (10% better) = 1.0
            return np.clip((min_improvement + 0.1) / 0.2, 0.0, 1.0)
        
        else:  # BALANCED
            # Combine: 40% improved fraction + 40% avg + 20% min
            frac_score = (neighbor_deltas > 0).sum() / len(neighbor_deltas)
            avg_score = np.clip(neighbor_deltas.mean() / 0.15, 0.0, 1.0)
            min_score = np.clip((neighbor_deltas.min() + 0.1) / 0.2, 0.0, 1.0)
            
            return 0.4 * frac_score + 0.4 * avg_score + 0.2 * min_score
    
    def compute_composite_score(
        self,
        component_scores: Dict[str, float],
    ) -> float:
        """
        Compute final composite score from components.
        
        Args:
            component_scores: Dictionary from compute_component_scores()
        
        Returns:
            Composite score in [0, 1]
        """
        composite = (
            self.w_worst * component_scores['worst_ap'] +
            self.w_avg * component_scores['average'] +
            self.w_cov * component_scores['coverage'] +
            self.w_neigh * component_scores['neighborhood']
        )
        
        return float(np.clip(composite, 0.0, 1.0))
    
    def generate_warnings(
        self,
        component_scores: Dict[str, float],
        threshold_low: float = 0.3,
    ) -> list[str]:
        """
        Generate warnings for risky placements.
        
        Args:
            component_scores: Dictionary from compute_component_scores()
            threshold_low: Warn if any component below this
        
        Returns:
            List of warning messages
        """
        warnings = []
        
        if component_scores['worst_ap'] < threshold_low:
            warnings.append(
                "⚠️ Low worst-AP improvement: "
                f"Most overloaded AP only improves by {component_scores['worst_ap_improvement_raw']:.2f}"
            )
        
        if component_scores['average'] < threshold_low:
            warnings.append(
                "⚠️ Low average reduction: "
                f"Overall network improves by only {component_scores['avg_reduction_raw']:.2f}"
            )
        
        if component_scores['coverage'] < 0.2:
            warnings.append(
                f"⚠️ Limited coverage: Only {component_scores['num_improved']} APs improve"
            )
        
        if component_scores['neighborhood'] < threshold_low:
            warnings.append(
                "⚠️ Neighborhood degradation: Some neighbors may get worse"
            )
        
        return warnings