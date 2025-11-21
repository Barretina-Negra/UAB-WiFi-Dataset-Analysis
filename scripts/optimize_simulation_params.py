
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from itertools import product
import time

# Add src and root to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dashboard.data_io import read_ap_snapshot, extract_group, GEOJSON_PATH, read_geoloc_points
from dashboard.simulator_viz import (
    simulate_ap_addition,
    generate_candidate_locations,
    recalculate_conflictivity
)
from experiments.polcorresa.simulator.scoring import CompositeScorer, NeighborhoodOptimizationMode
from experiments.polcorresa.simulator.config import SimulationConfig, StressLevel

def run_benchmark():
    print("Starting parameter optimization benchmark...")
    
    # Load data
    geo_df = read_geoloc_points(GEOJSON_PATH)
    
    # Use a representative snapshot (e.g., one with high stress)
    # For this script, we'll pick one manually or find one.
    # Let's assume we can find one in reducedData/ap
    ap_dir = Path("reducedData/ap")
    snapshots = sorted(list(ap_dir.glob("AP-info-v2-*.json")))
    if not snapshots:
        print("No snapshots found.")
        return

    # Pick a snapshot in the middle/late afternoon which is usually busy
    target_snap = snapshots[-1] # Use the latest one
    print(f"Using snapshot: {target_snap.name}")
    
    df_snap = read_ap_snapshot(target_snap, band_mode='worst')
    df_snap = df_snap.merge(geo_df, on='name', how='inner')
    if "group_code" not in df_snap.columns:
        df_snap["group_code"] = df_snap["name"].apply(extract_group)
    df_snap = df_snap[df_snap['group_code'] != 'SAB'].copy()
    
    # Ensure conflictivity
    if 'conflictivity' not in df_snap.columns:
        df_snap = recalculate_conflictivity(df_snap)
        
    print(f"Loaded {len(df_snap)} APs. Avg Conflictivity: {df_snap['conflictivity'].mean():.3f}")

    # Define parameter grid
    param_grid = {
        'interference_radius': [20, 25, 30, 40],
        'cca_increase': [0.02, 0.05, 0.10, 0.15],
        'weights': [
            # (worst, avg, cov, neigh)
            (0.30, 0.30, 0.20, 0.20), # Balanced
            (0.50, 0.20, 0.15, 0.15), # Focus on worst APs
            (0.20, 0.50, 0.15, 0.15), # Focus on global average
            (0.20, 0.20, 0.20, 0.40), # Focus on neighborhood
        ]
    }
    
    results = []
    
    # Generate candidates ONCE using a standard config to save time
    # We want to evaluate the SCORING of the same candidates under different physics/weights
    print("Generating candidates...")
    candidates = generate_candidate_locations(
        df_snap,
        tile_meters=15, # Coarse grid for speed
        conflictivity_threshold=0.6,
        radius_m=25,
        indoor_only=True,
        neighbor_radius_tiles=2,
        inner_clearance_m=10
    )
    
    if candidates.empty:
        print("No candidates found with default settings. Lowering threshold.")
        candidates = generate_candidate_locations(
            df_snap,
            tile_meters=15,
            conflictivity_threshold=0.4,
            radius_m=25,
            indoor_only=True
        )
    
    print(f"Evaluating {len(candidates)} candidates across parameter combinations...")
    top_candidates = candidates.head(5) # Only test top 5 candidates to save time
    
    combinations = list(product(
        param_grid['interference_radius'],
        param_grid['cca_increase'],
        param_grid['weights']
    ))
    
    for i, (rad, cca, weights) in enumerate(combinations):
        w_worst, w_avg, w_cov, w_neigh = weights
        
        config = SimulationConfig(
            interference_radius_m=rad,
            cca_increase_factor=cca,
            indoor_only=True,
            conflictivity_threshold_placement=0.6,
            snapshots_per_profile=1,
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
            interference_radius_m=rad,
        )
        
        # Test each candidate
        best_score = -1.0
        best_metrics = {}
        
        for _, cand in top_candidates.iterrows():
            _, _, metrics = simulate_ap_addition(
                df_snap,
                cand['lat'],
                cand['lon'],
                config,
                scorer
            )
            
            if metrics['composite_score'] > best_score:
                best_score = metrics['composite_score']
                best_metrics = metrics
        
        results.append({
            'radius': rad,
            'cca': cca,
            'w_worst': w_worst,
            'w_avg': w_avg,
            'w_neigh': w_neigh,
            'final_score': best_score,
            'avg_reduction': best_metrics.get('avg_reduction_raw', 0),
            'worst_improv': best_metrics.get('worst_ap_improvement_raw', 0),
            'neigh_improv': best_metrics.get('neighborhood_improvement_raw', 0),
            'impact_eff': best_metrics.get('impact_efficiency', 0),
            'new_clients': best_metrics.get('new_ap_client_count', 0)
        })
        
        if i % 5 == 0:
            print(f"Processed {i+1}/{len(combinations)} combinations...")

    # Analyze results
    res_df = pd.DataFrame(results)
    
    print("\n--- Top 5 Configurations by Final Score ---")
    print(res_df.sort_values('final_score', ascending=False).head(5).to_string())
    
    print("\n--- Top 5 Configurations by Impact Efficiency ---")
    print(res_df.sort_values('impact_eff', ascending=False).head(5).to_string())
    
    # Find the "Balanced Best"
    # We want high neighborhood improvement AND decent global score
    res_df['balanced_rank'] = (
        res_df['final_score'].rank(pct=True) + 
        res_df['neigh_improv'].rank(pct=True)
    )
    print("\n--- Top 5 Balanced Configurations ---")
    print(res_df.sort_values('balanced_rank', ascending=False).head(5).to_string())

if __name__ == "__main__":
    run_benchmark()
