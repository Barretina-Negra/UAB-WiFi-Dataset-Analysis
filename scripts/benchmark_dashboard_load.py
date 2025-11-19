
import sys
import time
from pathlib import Path
import pandas as pd

# Add src to path so we can import dashboard modules
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from dashboard.data_io import read_geoloc_points, GEOJSON_PATH
from dashboard.voronoi_viz import compute_coverage_regions

def benchmark():
    print("Loading geolocation data...")
    try:
        geo_df = read_geoloc_points(GEOJSON_PATH)
        print(f"Loaded {len(geo_df)} AP locations.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Parameters used in the dashboard
    tile_meters = 7.0
    radius_m = 25.0
    max_tiles = 40000

    print("\n--- Benchmarking compute_coverage_regions ---")
    
    # First run (uncached)
    start_time = time.perf_counter()
    regions = compute_coverage_regions(
        geo_df, 
        tile_meters=tile_meters, 
        radius_m=radius_m, 
        max_tiles=max_tiles
    )
    end_time = time.perf_counter()
    first_run_time = end_time - start_time
    print(f"First run (uncached): {first_run_time:.4f} seconds")
    print(f"Generated {len(regions)} regions.")

    # Second run (cached)
    start_time = time.perf_counter()
    regions_cached = compute_coverage_regions(
        geo_df, 
        tile_meters=tile_meters, 
        radius_m=radius_m, 
        max_tiles=max_tiles
    )
    end_time = time.perf_counter()
    second_run_time = end_time - start_time
    print(f"Second run (cached):  {second_run_time:.4f} seconds")
    
    # Verify results are the same
    assert len(regions) == len(regions_cached)
    
    speedup = first_run_time / second_run_time if second_run_time > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    benchmark()
