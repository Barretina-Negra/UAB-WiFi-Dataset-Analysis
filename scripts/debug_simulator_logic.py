import sys
import os
from pathlib import Path
import pandas as pd
import json
import numpy as np

# Add workspace root to path
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import simulator components
from experiments.polcorresa.simulator.multi_scenario_simulator import MultiScenarioSimulator
from experiments.polcorresa.simulator.config import SimulationConfig

# Mock read_ap_snapshot if import fails, but let's try to import from src.dashboard.data_io
try:
    from src.dashboard.data_io import read_ap_snapshot
except ImportError:
    print("Could not import read_ap_snapshot from src.dashboard.data_io, defining locally.")
    def read_ap_snapshot(path: Path, band_mode: str = "worst") -> pd.DataFrame:
        with open(path, 'r') as f:
            data = json.load(f)
        
        rows = []
        for ap in data:
            name = ap.get('name', 'Unknown')
            
            # Extract client count
            clients = 0
            if 'clients' in ap:
                clients = len(ap['clients'])
            
            # Extract utilization
            util_2g = None
            util_5g = None
            
            if 'radios' in ap:
                for radio in ap['radios']:
                    band = radio.get('band', '')
                    util = radio.get('utilization', 0)
                    if '2.4' in band:
                        util_2g = util
                    elif '5' in band:
                        util_5g = util
            
            rows.append({
                'name': name,
                'client_count': clients,
                'util_2g': util_2g,
                'util_5g': util_5g
            })
        
        df = pd.DataFrame(rows)
        return df

def load_geo_df():
    path = Path("realData/geoloc/aps_geolocalizados_wgs84.geojson")
    with open(path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for feature in data['features']:
        props = feature['properties']
        coords = feature['geometry']['coordinates']
        rows.append({
            'name': props.get('USER_NOM_A', ''), # Correct property name
            'lon': coords[0],
            'lat': coords[1]
        })
    return pd.DataFrame(rows)

def main():
    print("Starting simulator debug...")
    
    # Load data
    geo_df = load_geo_df()
    print(f"Loaded {len(geo_df)} AP locations.")
    if not geo_df.empty:
        print(f"Sample geo names: {geo_df['name'].head().tolist()}")
    
    snapshot_path = Path("realData/ap/AP-info-v2-2025-04-29T12_00_01+02_00.json")
    if not snapshot_path.exists():
        print(f"Snapshot not found: {snapshot_path}")
        return

    # Setup simulator
    # We need a list of snapshots for the constructor, even if we only use one for simulate_ap_addition
    snapshots = [(snapshot_path, pd.Timestamp.now())]
    
    sim = MultiScenarioSimulator(snapshots, geo_df)
    
    # Load the specific snapshot for testing
    # Note: _read_snapshot_with_geo uses dashboard.conflictivity_dashboard_interpolation which might fail if not in path
    # So we might need to patch it or ensure path is correct.
    # Let's try to run it.
    try:
        df_base = sim._read_snapshot_with_geo(snapshot_path)
    except Exception as e:
        print(f"Error loading snapshot via simulator method: {e}")
        print("Falling back to manual load...")
        df_base = read_ap_snapshot(snapshot_path)
        print(f"Loaded {len(df_base)} APs from snapshot.")
        if not df_base.empty:
            print(f"Sample names: {df_base['name'].head().tolist()}")

        df_base = df_base.merge(geo_df, on='name', how='inner')
        print(f"Merged with geo: {len(df_base)} APs.")

        # Filter SAB
        # We need group_code.
        if 'group_code' not in df_base.columns:
             # Try to extract group code properly. 
             # Usually it is the first part of the name? Or maybe we should check how it is done in the codebase.
             # In integrated_dashboard.py: tmp["group_code"] = tmp["name"].apply(extract_group)
             # extract_group is likely imported.
             # Let's just assume everything not SAB is UAB for now.
             # If name starts with SAB, it is SAB.
             df_base['group_code'] = df_base['name'].apply(lambda x: 'SAB' if x.startswith('SAB') or 'Sabadell' in x else 'UAB')
        
        df_base = df_base[df_base['group_code'] != 'SAB'].copy()

    print(f"Loaded snapshot with {len(df_base)} APs (after filtering UAB).")
    
    if df_base.empty:
        print("Snapshot is empty after filtering!")
        return

    # Pick a location near a high-conflict AP
    # Find AP with high conflictivity
    # We need to calculate conflictivity first as read_ap_snapshot might not have it?
    # The simulator calculates it inside simulate_ap_addition (recalculate_conflictivity)
    # But let's see if df_base has it.
    
    print("Calculating baseline conflictivity...")
    df_base = sim.recalculate_conflictivity(df_base)
    
    high_conf_ap = df_base.sort_values('conflictivity', ascending=False).iloc[0]
    print(f"Highest conflict AP: {high_conf_ap['name']} (conf={high_conf_ap['conflictivity']:.2f}) at ({high_conf_ap['lat']:.4f}, {high_conf_ap['lon']:.4f})")
    
    # Place new AP slightly offset (e.g. 10 meters)
    # 1 degree lat approx 111km -> 10m approx 0.0001 deg
    new_lat = high_conf_ap['lat'] + 0.0001
    new_lon = high_conf_ap['lon'] + 0.0001
    
    print(f"Simulating new AP at ({new_lat:.4f}, {new_lon:.4f})...")
    
    updated_df, new_stats, metrics = sim.simulate_ap_addition(df_base, new_lat, new_lon)
    
    print("\n--- Neighbor Analysis ---")
    # Identify neighbors (using same logic as simulator)
    from experiments.polcorresa.simulator.spatial import haversine_m
    
    dists = haversine_m(new_lat, new_lon, df_base['lat'].values, df_base['lon'].values)
    df_base['dist_to_new'] = dists
    df_base['in_range'] = dists <= sim.config.interference_radius_m
    
    neighbors_idx = df_base[df_base['in_range']].index
    
    # Compare old vs new
    worsened_count = 0
    for idx in neighbors_idx:
        old_conf = df_base.loc[idx, 'conflictivity']
        new_conf = updated_df.loc[idx, 'conflictivity']
        diff = new_conf - old_conf
        
        if diff > 0.05:
            worsened_count += 1
            print(f"  {df_base.loc[idx, 'name']}: {old_conf:.2f} -> {new_conf:.2f} (Diff: +{diff:.2f})")
            print(f"    Util 2G: {df_base.loc[idx, 'util_2g']:.1f}% -> {updated_df.loc[idx, 'util_2g']:.1f}%")
            print(f"    Util 5G: {df_base.loc[idx, 'util_5g']:.1f}% -> {updated_df.loc[idx, 'util_5g']:.1f}%")
            print(f"    Clients: {df_base.loc[idx, 'client_count']} -> {updated_df.loc[idx, 'client_count']}")
            
    if worsened_count == 0:
        print("No neighbors worsened significantly (>0.05).")
    else:
        print(f"Found {worsened_count} neighbors that worsened significantly.")

    print("\n--- Simulation Results ---")
    print(f"Composite Score: {metrics['composite_score']:.3f}")
    print(f"Avg Conflictivity: {metrics['avg_conflictivity_before']:.4f} -> {metrics['avg_conflictivity_after']:.4f}")
    print(f"Worst AP Conflictivity: {metrics['worst_ap_conflictivity_before']:.4f} -> {metrics['worst_ap_conflictivity_after']:.4f}")
    print(f"New AP Clients: {metrics['new_ap_client_count']}")
    
    print("\n--- Warnings ---")
    for w in metrics['warnings']:
        print(f"- {w}")

if __name__ == "__main__":
    main()
