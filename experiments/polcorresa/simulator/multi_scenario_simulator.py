"""
Multi-scenario AP placement simulator.

Core simulation engine that:
1. Tests candidate locations across multiple stress scenarios
2. Models RF propagation, client redistribution, and interference
3. Computes composite scores with neighborhood optimization
4. Generates warnings for risky placements
"""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime
from math import log1p

import numpy as np
import pandas as pd

from .config import SimulationConfig, StressLevel
from .stress_profiler import StressProfiler
from .scoring import CompositeScorer, NeighborhoodOptimizationMode
from .spatial import haversine_m, compute_convex_hull_polygon, mask_points_in_polygon


class MultiScenarioSimulator:
    """
    Main simulation engine for multi-scenario AP placement optimization.
    """
    
    def __init__(
        self,
        snapshots: List[Tuple[Path, datetime]],
        geo_df: pd.DataFrame,
        config: Optional[SimulationConfig] = None,
    ):
        """
        Args:
            snapshots: List of (file_path, timestamp) tuples
            geo_df: DataFrame with columns [name, lon, lat]
            config: Simulation configuration (uses defaults if None)
        """
        self.snapshots = snapshots
        self.geo_df = geo_df
        self.config = config or SimulationConfig()
        
        # Initialize profiler and scorer
        self.profiler = StressProfiler(
            snapshots,
            utilization_threshold_critical=self.config.utilization_threshold_critical,
            utilization_threshold_high=self.config.utilization_threshold_high,
        )
        
        self.scorer = CompositeScorer(
            weight_worst_ap=self.config.weight_worst_ap,
            weight_average=self.config.weight_average,
            weight_coverage=self.config.weight_coverage,
            weight_neighborhood=self.config.weight_neighborhood,
            neighborhood_mode=NeighborhoodOptimizationMode.BALANCED,
            interference_radius_m=self.config.interference_radius_m,
        )
        
        # Cache
        self._stress_profiles = None
    
    # -------- Core Simulation Methods --------
    
    def compute_rssi(self, distance_m: float) -> float:
        """
        Compute RSSI using log-distance path loss model.
        
        RSSI(d) = RSSI₀ - 10 × n × log₁₀(d / d₀)
        
        Args:
            distance_m: Distance in meters
        
        Returns:
            RSSI in dBm
        """
        d = max(distance_m, self.config.reference_distance_m)
        
        path_loss = 10 * self.config.path_loss_exponent * np.log10(
            d / self.config.reference_distance_m
        )
        
        return self.config.reference_rssi_dbm - path_loss
    
    def estimate_client_distribution(
        self,
        df_aps: pd.DataFrame,
        new_ap_lat: float,
        new_ap_lon: float,
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Simulate client redistribution when a new AP is added.
        
        Uses hybrid signal-strength + load-balancing model.
        
        Args:
            df_aps: Current AP state
            new_ap_lat, new_ap_lon: New AP location
        
        Returns:
            (updated_df_aps, new_ap_stats)
        """
        df = df_aps.copy()
        
        # Compute distances and RSSI
        df['dist_to_new'] = haversine_m(
            new_ap_lat, new_ap_lon,
            df['lat'].values, df['lon'].values
        )
        
        df['rssi_new'] = df['dist_to_new'].apply(self.compute_rssi)
        
        # Identify APs in range
        df['in_range'] = (
            (df['rssi_new'] >= self.config.min_rssi_dbm) &
            (df['dist_to_new'] <= self.config.interference_radius_m)
        )
        
        # Hybrid redistribution
        total_transferred = 0
        # We consider all APs in range as candidates for offloading, regardless of their current conflictivity.
        # Clients roam based on signal strength, not the AP's stress level.
        candidates = df[df['in_range']].copy()
        candidates = candidates.sort_values('conflictivity', ascending=False)
        
        for idx, row in candidates.iterrows():
            # Signal strength factor
            rssi_factor = (row['rssi_new'] - self.config.min_rssi_dbm) / 20.0
            rssi_factor = max(0.0, min(1.0, rssi_factor))
            
            # Conflictivity factor
            conflict_factor = row['conflictivity']
            
            # Combined transfer fraction
            # Balanced approach: signal quality and current stress both drive migration.
            transfer_fraction = min(
                self.config.max_offload_fraction,
                0.4 * rssi_factor + 0.4 * conflict_factor
            )
            
            # Apply sticky client constraint
            transfer_fraction *= (1 - self.config.sticky_client_fraction)
            
            # Use round() instead of int() to handle small client counts better
            n_transfer = int(round(row['client_count'] * transfer_fraction))
            
            # Reduce utilization proportionally to client loss
            if row['client_count'] > 0 and n_transfer > 0:
                fraction_removed = n_transfer / row['client_count']
                # Assume utilization is roughly proportional to clients
                # We clamp the reduction to avoid going below 0
                # We also assume some baseline utilization (e.g. 5%) that doesn't go away
                
                # Update 2G
                current_2g = row['util_2g'] if not pd.isna(row['util_2g']) else 0.0
                new_2g = max(5.0, current_2g * (1 - fraction_removed))
                df.at[idx, 'util_2g'] = new_2g
                
                # Update 5G
                current_5g = row['util_5g'] if not pd.isna(row['util_5g']) else 0.0
                new_5g = max(5.0, current_5g * (1 - fraction_removed))
                df.at[idx, 'util_5g'] = new_5g
                
            df.at[idx, 'client_count'] -= n_transfer
            total_transferred += n_transfer
        
        # New AP stats
        new_ap_stats = {
            'lat': new_ap_lat,
            'lon': new_ap_lon,
            'client_count': total_transferred,
            'name': 'AP-NEW-SIM',
            'group_code': 'SIM',
        }
        
        # Estimate new AP utilization
        client_fraction = min(1.0, total_transferred / self.config.max_clients_per_ap)
        new_ap_stats['util_2g'] = client_fraction * self.config.target_util_2g
        new_ap_stats['util_5g'] = client_fraction * self.config.target_util_5g
        
        return df, new_ap_stats
    
    def apply_cca_interference(
        self,
        df_aps: pd.DataFrame,
        new_ap_stats: Dict,
    ) -> pd.DataFrame:
        """
        Apply co-channel interference (CCA busy increase) to neighbors.
        
        Args:
            df_aps: Current AP state
            new_ap_stats: New AP statistics
        
        Returns:
            Updated DataFrame with increased utilization
        """
        df = df_aps.copy()
        
        distances = haversine_m(
            new_ap_stats['lat'], new_ap_stats['lon'],
            df['lat'].values, df['lon'].values
        )
        
        in_interference_range = distances <= self.config.interference_radius_m
        
        # Distance-weighted increase
        # We also apply a probability factor for channel overlap.
        # Not all neighbors are on the same channel.
        base_increase = np.where(
            in_interference_range,
            self.config.cca_increase_factor * (1 - distances / self.config.interference_radius_m),
            0.0
        )
        
        # Apply to both bands with respective overlap probabilities
        df['util_2g'] = np.clip(
            df['util_2g'] * (1 + base_increase * self.config.channel_overlap_prob_2g),
            0.0, 100.0
        )
        df['util_5g'] = np.clip(
            df['util_5g'] * (1 + base_increase * self.config.channel_overlap_prob_5g),
            0.0, 100.0
        )
        
        df['agg_util'] = np.maximum(df['util_2g'], df['util_5g'])
        
        affected_count = np.sum(in_interference_range)

        return df
    
    def recalculate_conflictivity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full conflictivity recalculation after network changes.
        
        Args:
            df: DataFrame with updated utilization and client counts
        
        Returns:
            DataFrame with recalculated conflictivity
        """
        def airtime_score(util, band):
            u = max(0.0, min(100.0, util or 0.0))
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
        
        # Airtime scores
        df['air_s_2g'] = df['util_2g'].apply(lambda u: airtime_score(u, "2g") if not np.isnan(u) else np.nan)
        df['air_s_5g'] = df['util_5g'].apply(lambda u: airtime_score(u, "5g") if not np.isnan(u) else np.nan)
        
        df['airtime_score'] = np.nanmax(
            np.vstack([df['air_s_2g'].fillna(-1), df['air_s_5g'].fillna(-1)]),
            axis=0
        )
        df['airtime_score'] = df['airtime_score'].where(df['airtime_score'] >= 0, np.nan)
        
        # Client pressure
        p95 = float(np.nanpercentile(df['client_count'].fillna(0), 95)) if len(df) else 1.0
        df['client_score'] = df['client_count'].apply(
            lambda n: log1p(max(0.0, float(n or 0.0))) / log1p(max(1.0, p95))
        )
        df['client_score'] = df['client_score'].clip(0, 1)
        
        # Resource health (use existing or default)
        if 'cpu_utilization' not in df.columns:
            df['cpu_utilization'] = 0.0
        if 'mem_used_pct' not in df.columns:
            df['mem_used_pct'] = 0.0
        
        def cpu_score(c):
            c = max(0.0, min(100.0, c or 0.0))
            if c <= 70: return 0.0
            if c <= 90: return 0.6 * ((c - 70) / 20.0)
            return 0.6 + 0.4 * ((c - 90) / 10.0)
        
        def mem_score(m):
            m = max(0.0, min(100.0, m or 0.0))
            if m <= 80: return 0.0
            if m <= 95: return 0.6 * ((m - 80) / 15.0)
            return 0.6 + 0.4 * ((m - 95) / 5.0)
        
        df['cpu_score'] = df['cpu_utilization'].apply(cpu_score)
        df['mem_score'] = df['mem_used_pct'].apply(mem_score)
        
        # Zero-client relief
        df['airtime_score_adj'] = [
            (a * 0.8 if (c or 0) == 0 else a) if not np.isnan(a) else np.nan
            for a, c in zip(df['airtime_score'], df['client_count'])
        ]
        
        # Final conflictivity
        W_AIR, W_CL, W_CPU, W_MEM = 0.85, 0.10, 0.02, 0.03
        df['airtime_score_filled'] = df['airtime_score_adj'].fillna(0.4)
        df['conflictivity'] = (
            df['airtime_score_filled'] * W_AIR +
            df['client_score'].fillna(0) * W_CL +
            df['cpu_score'].fillna(0) * W_CPU +
            df['mem_score'].fillna(0) * W_MEM
        ).clip(0, 1)
        
        return df
    
    def simulate_ap_addition(
        self,
        df_baseline: pd.DataFrame,
        new_ap_lat: float,
        new_ap_lon: float,
    ) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Simulate adding a new AP at given location.
        
        Args:
            df_baseline: Current network state
            new_ap_lat, new_ap_lon: New AP location
        
        Returns:
            (updated_df, new_ap_stats, metrics)
        """
        # Step 1: Client redistribution
        df_updated, new_ap_stats = self.estimate_client_distribution(
            df_baseline, new_ap_lat, new_ap_lon
        )
        
        # Step 2: CCA interference
        df_updated = self.apply_cca_interference(df_updated, new_ap_stats)
        
        # Step 3: Recalculate conflictivity
        df_updated = self.recalculate_conflictivity(df_updated)
        
        # Step 4: Compute neighbor mask
        distances = haversine_m(
            new_ap_lat, new_ap_lon,
            df_updated['lat'].values, df_updated['lon'].values
        )
        neighbor_mask = distances <= self.config.interference_radius_m
        
        # Step 5: Compute scores
        baseline_conf = df_baseline['conflictivity'].values
        updated_conf = df_updated['conflictivity'].values
        
        component_scores = self.scorer.compute_component_scores(
            baseline_conf,
            updated_conf,
            neighbor_mask,
        )
        
        composite_score = self.scorer.compute_composite_score(component_scores)
        
        # Step 6: Generate warnings
        warnings = self.scorer.generate_warnings(component_scores)
        
        # Step 7: Build metrics
        metrics = {
            **component_scores,
            'composite_score': composite_score,
            'warnings': warnings,
            
            'avg_conflictivity_before': float(baseline_conf.mean()),
            'avg_conflictivity_after': float(updated_conf.mean()),
            
            'worst_ap_conflictivity_before': float(baseline_conf.max()),
            'worst_ap_conflictivity_after': float(updated_conf.max()),
            
            'num_high_conflict_before': int((baseline_conf > 0.7).sum()),
            'num_high_conflict_after': int((updated_conf > 0.7).sum()),
            
            'new_ap_client_count': new_ap_stats['client_count'],
            'new_ap_util_2g': new_ap_stats['util_2g'],
            'new_ap_util_5g': new_ap_stats['util_5g'],
        }
        
        return df_updated, new_ap_stats, metrics
    
    # -------- Candidate Generation --------
    
    def generate_candidate_locations(
        self,
        df_aps: pd.DataFrame,
        tile_meters: float = 7.0,
    ) -> pd.DataFrame:
        """
        Generate candidate locations for new AP placement.
        
        Args:
            df_aps: Current AP state
            tile_meters: Grid resolution
        
        Returns:
            DataFrame with columns [lon, lat, conflictivity]
        """
        lons = df_aps['lon'].values
        lats = df_aps['lat'].values
        
        hull = compute_convex_hull_polygon(lons, lats)
        if hull is None:
            return pd.DataFrame()
        
        # Create grid
        lat0 = float(np.mean(lats))
        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * np.cos(np.deg2rad(lat0))
        dlat = tile_meters / meters_per_deg_lat
        dlon = tile_meters / meters_per_deg_lon
        
        minx, miny, maxx, maxy = hull.bounds
        lon_centers = np.arange(minx + dlon/2, maxx, dlon)
        lat_centers = np.arange(miny + dlat/2, maxy, dlat)
        
        mask = mask_points_in_polygon(lon_centers, lat_centers, hull)
        
        XX, YY = np.meshgrid(lon_centers, lat_centers)
        centers_in = np.column_stack([XX[mask].ravel(), YY[mask].ravel()])
        
        if len(centers_in) == 0:
            return pd.DataFrame()
        
        # Interpolate conflictivity
        dists = haversine_m(
            centers_in[:, 1][:, None],
            centers_in[:, 0][:, None],
            lats[None, :],
            lons[None, :]
        )
        
        radius_m = 25.0  # Fixed interpolation radius
        W = np.maximum(0, 1 - dists / radius_m)
        W[dists >= radius_m] = 0
        
        cvals = df_aps['conflictivity'].values
        denom = W.sum(axis=1)
        
        with np.errstate(invalid='ignore', divide='ignore'):
            num = (W * cvals[None, :]).sum(axis=1)
            z_pred = np.where(denom > 0, num / denom, 0.0)
        
        # Boundary term
        d_min = dists.min(axis=1)
        boundary_conf = np.clip(d_min / radius_m, 0.0, 1.0)
        z_pred = np.maximum(z_pred, boundary_conf)
        
        candidates = pd.DataFrame({
            'lon': centers_in[:, 0],
            'lat': centers_in[:, 1],
            'conflictivity': z_pred,
        })
        
        # Filter by threshold
        candidates = candidates[
            candidates['conflictivity'] >= self.config.conflictivity_threshold_placement
        ].copy()
        
        # Indoor filter
        if self.config.indoor_only:
            def is_indoor(row):
                d = haversine_m(row['lat'], row['lon'], lats, lons).min()
                return (
                    d >= self.config.min_distance_to_ap_m and
                    d <= self.config.max_distance_to_ap_m
                )
            
            candidates['indoor'] = candidates.apply(is_indoor, axis=1)
            candidates = candidates[candidates['indoor']].copy()
            candidates.drop(columns=['indoor'], inplace=True)
        
        return candidates.reset_index(drop=True)
    
    # -------- Multi-Scenario Evaluation --------
    
    def find_optimal_placement(
        self,
        top_k: int = 5,
        tile_meters: float = 7.0,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Find optimal AP placement locations using multi-scenario approach.
        
        Args:
            top_k: Return top K candidate locations
            tile_meters: Grid resolution for candidate generation
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            DataFrame with top K candidates ranked by composite score
        """
        if progress_callback:
            progress_callback(0, 100, "Classifying stress profiles...")
        
        # Step 1: Classify snapshots
        self._stress_profiles = self.profiler.classify_snapshots()
        self.profiler.print_summary()
        
        # Step 2: Determine target profile
        target_profile = self.config.target_stress_profile
        if target_profile is None:
            # Test all profiles
            profiles_to_test = list(StressLevel)
        else:
            profiles_to_test = [target_profile]
        
        # Step 3: Get representative snapshots
        all_scenarios = []
        for profile in profiles_to_test:
            snaps = self.profiler.get_representative_snapshots(
                profile,
                n_samples=self.config.snapshots_per_profile
            )
            for path, dt in snaps:
                all_scenarios.append((profile, path, dt))
        
        if not all_scenarios:
            raise ValueError(f"No snapshots found for target profile(s): {profiles_to_test}")
        
        print(f"\n📍 Testing {len(all_scenarios)} scenarios across {len(profiles_to_test)} stress profile(s)")
        
        # Step 4: Load first snapshot to generate candidates
        first_path = all_scenarios[0][1]
        df_base = self._read_snapshot_with_geo(first_path)
        
        if progress_callback:
            progress_callback(10, 100, "Generating candidate locations...")
        
        candidates = self.generate_candidate_locations(df_base, tile_meters)
        
        if candidates.empty:
            print(f"⚠️  No candidates found with conflictivity > {self.config.conflictivity_threshold_placement}")
            return pd.DataFrame()
        
        print(f"✅ Found {len(candidates)} candidate locations")
        print(f"   Testing top {min(top_k, len(candidates))} candidates...\n")
        
        # Step 5: Evaluate each candidate across all scenarios
        results = []
        total_sims = min(top_k, len(candidates)) * len(all_scenarios)
        sim_count = 0
        
        for cand_idx, cand_row in candidates.head(top_k).iterrows():
            scenario_results = []
            
            for profile, snap_path, snap_dt in all_scenarios:
                if progress_callback:
                    pct = int((sim_count / total_sims) * 90) + 10
                    progress_callback(pct, 100, f"Evaluating candidate {cand_idx+1}/{min(top_k, len(candidates))}...")
                
                # Load scenario snapshot
                df_scenario = self._read_snapshot_with_geo(snap_path)
                
                # Simulate
                _, _, metrics = self.simulate_ap_addition(
                    df_scenario,
                    cand_row['lat'],
                    cand_row['lon'],
                )
                
                metrics['stress_profile'] = profile.value
                metrics['timestamp'] = snap_dt
                scenario_results.append(metrics)
                
                sim_count += 1
            
            # Aggregate across scenarios
            aggregated = self._aggregate_scenario_results(
                cand_row['lat'],
                cand_row['lon'],
                cand_row['conflictivity'],
                scenario_results,
            )
            
            results.append(aggregated)
        
        if progress_callback:
            progress_callback(100, 100, "Ranking results...")
        
        # Step 6: Rank and return
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('final_score', ascending=False)
        
        print("\n🏆 Top placements found!\n")
        self._print_results_summary(results_df.head(top_k))
        
        return results_df
    
    def _aggregate_scenario_results(
        self,
        lat: float,
        lon: float,
        base_conflictivity: float,
        scenario_results: List[Dict],
    ) -> Dict:
        """Aggregate metrics across scenarios."""
        aggregated = {
            'lat': lat,
            'lon': lon,
            'base_conflictivity': base_conflictivity,
            'n_scenarios': len(scenario_results),
        }
        
        # Extract composite scores
        scores = [r['composite_score'] for r in scenario_results]
        aggregated['final_score'] = float(np.mean(scores))
        aggregated['score_std'] = float(np.std(scores))
        aggregated['score_min'] = float(np.min(scores))
        aggregated['score_max'] = float(np.max(scores))
        
        # Aggregate other metrics
        for key in ['worst_ap_improvement_raw', 'avg_reduction_raw', 'num_improved', 'new_ap_client_count']:
            values = [r.get(key, 0) for r in scenario_results]
            aggregated[f'{key}_mean'] = float(np.mean(values))
            aggregated[f'{key}_std'] = float(np.std(values))
        
        # Collect warnings
        all_warnings = []
        for r in scenario_results:
            all_warnings.extend(r.get('warnings', []))
        
        # Deduplicate and count
        from collections import Counter
        warning_counts = Counter(all_warnings)
        aggregated['warnings'] = [
            f"{msg} (in {count}/{len(scenario_results)} scenarios)"
            for msg, count in warning_counts.most_common()
        ]
        
        # Per-profile breakdown
        by_profile = {}
        for r in scenario_results:
            profile = r['stress_profile']
            if profile not in by_profile:
                by_profile[profile] = []
            by_profile[profile].append(r['composite_score'])
        
        for profile, scores in by_profile.items():
            aggregated[f'score_{profile}'] = float(np.mean(scores))
        
        return aggregated
    
    def _print_results_summary(self, results_df: pd.DataFrame):
        """Print a nice summary of results."""
        for idx, row in results_df.iterrows():
            rank = idx + 1
            print(f"#{rank}  Score: {row['final_score']:.3f} ± {row['score_std']:.3f}")
            print(f"    Location: ({row['lat']:.6f}, {row['lon']:.6f})")
            print(f"    Avg Reduction: {row['avg_reduction_raw_mean']:.3f}")
            print(f"    Worst AP Improvement: {row['worst_ap_improvement_raw_mean']:.3f}")
            print(f"    New AP Clients: {int(row['new_ap_client_count_mean'])}")
            
            if row.get('warnings'):
                print("    ⚠️  Warnings:")
                for warning in row['warnings'][:2]:  # Show max 2
                    print(f"       {warning}")
            
            print()
    
    # -------- Utilities --------
    
    def _read_snapshot_with_geo(self, path: Path) -> pd.DataFrame:
        """Read snapshot and merge with geolocation."""
        from dashboard.conflictivity_dashboard_interpolation import read_ap_snapshot
        
        df = read_ap_snapshot(path, band_mode='worst')
        df = df.merge(self.geo_df, on='name', how='inner')
        
        # Filter to UAB only (exclude Sabadell)
        df = df[df['group_code'] != 'SAB'].copy()
        
        return df