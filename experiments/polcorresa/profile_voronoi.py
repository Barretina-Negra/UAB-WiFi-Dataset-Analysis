import time
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from dashboard.voronoi_viz import inverted_weighted_voronoi_edges, uab_tiled_choropleth_layer

def profile():
    # Create dummy data
    n_points = 100
    df = pd.DataFrame({
        'lat': np.random.uniform(41.49, 41.51, n_points),
        'lon': np.random.uniform(2.10, 2.12, n_points),
        'conflictivity': np.random.uniform(0, 1, n_points)
    })
    
    clip_poly = Polygon([
        (2.10, 41.49), (2.12, 41.49), (2.12, 41.51), (2.10, 41.51), (2.10, 41.49)
    ])
    
    print("Profiling inverted_weighted_voronoi_edges...")
    start = time.time()
    edges = inverted_weighted_voronoi_edges(df, clip_polygon=clip_poly)
    print(f"First run: {time.time() - start:.4f}s")
    
    start = time.time()
    edges = inverted_weighted_voronoi_edges(df, clip_polygon=clip_poly)
    print(f"Second run (cached): {time.time() - start:.4f}s")
    
    print("\nProfiling uab_tiled_choropleth_layer...")
    start = time.time()
    uab_tiled_choropleth_layer(df)
    print(f"First run: {time.time() - start:.4f}s")
    
    start = time.time()
    uab_tiled_choropleth_layer(df)
    print(f"Second run (cached): {time.time() - start:.4f}s")

if __name__ == "__main__":
    profile()
