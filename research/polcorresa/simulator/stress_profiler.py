"""
Hybrid stress profiler for temporal snapshot classification.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

from .config import StressLevel


class StressProfiler:
    """
    Classify snapshots into stress profiles using hybrid approach:
    1. Time-based heuristics (weekday/weekend, hour of day)
    2. Data-driven subdivision (actual utilization for HIGH vs CRITICAL)
    """
    
    def __init__(
        self,
        snapshots: List[Tuple[Path, datetime]],
        utilization_threshold_critical: float = 70.0,
        utilization_threshold_high: float = 50.0,
    ):
        """
        Args:
            snapshots: List of (file_path, timestamp) tuples
            utilization_threshold_critical: Avg util above this = CRITICAL (%)
            utilization_threshold_high: Avg util above this = HIGH (%)
        """
        self.snapshots = snapshots
        self.util_thresh_critical = utilization_threshold_critical
        self.util_thresh_high = utilization_threshold_high
        
        self._profiles: Dict[StressLevel, List[Tuple[Path, datetime]]] = {}
        self._profile_stats: Dict[StressLevel, dict] = {}
    
    def classify_by_time(self, dt: datetime) -> StressLevel:
        """
        Time-based heuristic classification.
        
        Rules:
        - Weekend: LOW
        - Weekday 7am-12pm: HIGH (will be subdivided by data)
        - Weekday 12pm-6pm: MEDIUM
        - Weekday off-peak: LOW
        """
        weekday = dt.weekday()  # 0=Monday, 6=Sunday
        hour = dt.hour
        
        # Weekend
        if weekday >= 5:
            return StressLevel.LOW
        
        # Weekday peak hours (7am-12pm)
        if 7 <= hour < 12:
            return StressLevel.HIGH  # Will subdivide into HIGH/CRITICAL
        
        # Weekday normal hours (12pm-6pm)
        if 12 <= hour < 18:
            return StressLevel.MEDIUM
        
        # Weekday off-peak
        return StressLevel.LOW
    
    def compute_avg_utilization(self, snapshot_path: Path) -> float:
        """
        Compute average utilization across all APs in a snapshot.
        
        Returns:
            Average utilization (%) or 0.0 if unable to compute
        """
        try:
            import json
            with snapshot_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            utils = []
            for ap in data:
                radios = ap.get('radios') or []
                for r in radios:
                    u = r.get('utilization')
                    if u is not None:
                        utils.append(float(u))
            
            return float(np.mean(utils)) if utils else 0.0
        except Exception:
            return 0.0
    
    def classify_snapshots(self) -> Dict[StressLevel, List[Tuple[Path, datetime]]]:
        """
        Classify all snapshots into stress profiles using hybrid approach.
        
        Returns:
            Dictionary mapping StressLevel -> list of (path, timestamp)
        """
        # Step 1: Time-based classification
        time_profiles = {level: [] for level in StressLevel}
        
        for path, dt in self.snapshots:
            level = self.classify_by_time(dt)
            time_profiles[level].append((path, dt))
        
        # Step 2: Subdivide HIGH into HIGH/CRITICAL using actual data
        high_candidates = time_profiles[StressLevel.HIGH]
        
        if high_candidates:
            # Compute utilization for each HIGH snapshot
            high_with_util = []
            for path, dt in high_candidates:
                avg_util = self.compute_avg_utilization(path)
                high_with_util.append((path, dt, avg_util))
            
            # Classify based on thresholds
            high_profile = []
            critical_profile = []
            
            for path, dt, avg_util in high_with_util:
                if avg_util >= self.util_thresh_critical:
                    critical_profile.append((path, dt))
                else:
                    high_profile.append((path, dt))
            
            time_profiles[StressLevel.HIGH] = high_profile
            time_profiles[StressLevel.CRITICAL] = critical_profile
        
        self._profiles = time_profiles
        self._compute_statistics()
        
        return self._profiles
    
    def _compute_statistics(self):
        """Compute statistics for each stress profile."""
        total = len(self.snapshots)
        
        for level in StressLevel:
            snaps = self._profiles.get(level, [])
            count = len(snaps)
            frequency = count / total if total > 0 else 0.0
            
            self._profile_stats[level] = {
                'count': count,
                'frequency': frequency,
                'percentage': frequency * 100,
            }
    
    def get_profile_statistics(self) -> Dict[StressLevel, dict]:
        """Get statistics for each stress profile."""
        if not self._profile_stats:
            self.classify_snapshots()
        return self._profile_stats
    
    def get_representative_snapshots(
        self,
        stress_level: StressLevel,
        n_samples: int = 5,
    ) -> List[Tuple[Path, datetime]]:
        """
        Get representative snapshots for a stress level.
        
        Samples evenly across the time range to ensure diversity.
        
        Args:
            stress_level: Which stress profile to sample from
            n_samples: Number of snapshots to return
        
        Returns:
            List of (path, timestamp) tuples
        """
        if not self._profiles:
            self.classify_snapshots()
        
        candidates = self._profiles.get(stress_level, [])
        
        if not candidates:
            return []
        
        if len(candidates) <= n_samples:
            return candidates
        
        # Sample evenly across time range
        indices = np.linspace(0, len(candidates) - 1, n_samples, dtype=int)
        return [candidates[i] for i in indices]
    
    def get_all_profiles(self) -> Dict[StressLevel, List[Tuple[Path, datetime]]]:
        """Get all classified profiles."""
        if not self._profiles:
            self.classify_snapshots()
        return self._profiles
    
    def print_summary(self):
        """Print a summary of stress profile classification."""
        stats = self.get_profile_statistics()
        
        print("\n" + "="*60)
        print("📊 STRESS PROFILE CLASSIFICATION SUMMARY")
        print("="*60)
        print(f"Total snapshots: {len(self.snapshots)}")
        print()
        
        for level in [StressLevel.LOW, StressLevel.MEDIUM, StressLevel.HIGH, StressLevel.CRITICAL]:
            stat = stats.get(level, {})
            count = stat.get('count', 0)
            pct = stat.get('percentage', 0.0)
            
            bar_length = int(pct / 2)  # Scale to 50 chars max
            bar = "█" * bar_length
            
            print(f"{level.value.upper():10s} │ {bar} {count:4d} ({pct:5.1f}%)")
        
        print("="*60 + "\n")