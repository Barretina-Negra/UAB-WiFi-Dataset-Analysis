"""Data loading and preprocessing helpers for the integrated dashboard."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]  # Up 2 levels: dashboard -> src -> repo root
AP_DIR = REPO_ROOT / "reducedData" / "ap"
GEOJSON_PATH = REPO_ROOT / "reducedData" / "geoloc" / "aps_geolocalizados_wgs84.geojson"


__all__ = [
    "REPO_ROOT",
    "AP_DIR",
    "GEOJSON_PATH",
    "norm01",
    "extract_group",
    "find_snapshot_files",
    "read_ap_snapshot",
    "read_geoloc_points",
]


def norm01(series: pd.Series, invert: bool = False) -> pd.Series:
    """Simple min-max normalization, falling back to 0.5 on missing variance."""

    s = series.astype(float)
    rng = s.max() - s.min()
    if rng == 0 or np.isinf(rng) or np.isnan(rng):
        return pd.Series(0.5, index=s.index)
    normalized = (s - s.min()) / rng
    return 1 - normalized if invert else normalized


def extract_group(ap_name: Optional[str]) -> Optional[str]:
    """Extract the building/group prefix from an AP name."""

    if not isinstance(ap_name, str):
        return None
    match = re.match(r"^AP-([A-Za-z]+)", ap_name)
    return match.group(1) if match else None


def find_snapshot_files(ap_dir: Path) -> List[Tuple[Path, datetime]]:
    """Return snapshot files sorted by their embedded timestamp."""

    files = list(ap_dir.glob("AP-info-v2-*.json"))
    files_with_time: List[Tuple[Path, datetime]] = []
    for file_path in files:
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})_(\d{2})", file_path.name)
        if match:
            y, mo, d, h, mi, s = map(int, match.groups())
            files_with_time.append((file_path, datetime(y, mo, d, h, mi, s)))
    files_with_time.sort(key=lambda x: x[1])
    return files_with_time


def read_ap_snapshot(path: Path, band_mode: str = "worst") -> pd.DataFrame:
    """Read an AP snapshot JSON file and compute conflictivity inputs."""

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    rows: List[Dict[str, float | str | None]] = []
    for ap in data:
        name = ap.get("name")
        client_count = ap.get("client_count", 0)
        cpu_util = ap.get("cpu_utilization", 0)
        mem_free = ap.get("mem_free", 0)
        mem_total = ap.get("mem_total", 0)
        group_name = ap.get("group_name")
        site = ap.get("site")
        radios = ap.get("radios") or []

        util_2g: List[float] = []
        util_5g: List[float] = []
        for radio in radios:
            utilization = radio.get("utilization")
            band = radio.get("band")
            if utilization is None:
                continue
            if band == 0:
                util_2g.append(float(utilization))
            elif band == 1:
                util_5g.append(float(utilization))

        max_2g = max(util_2g) if util_2g else np.nan
        max_5g = max(util_5g) if util_5g else np.nan

        if band_mode == "2.4GHz":
            agg_util = max_2g
        elif band_mode == "5GHz":
            agg_util = max_5g
        elif band_mode == "avg":
            valid = [value for value in [max_2g, max_5g] if not np.isnan(value)]
            agg_util = float(np.mean(valid)) if valid else np.nan
        else:
            pair = np.array([max_2g, max_5g], dtype=float)
            agg_util = float(np.nanmax(pair)) if not np.isnan(pair).all() else np.nan

        rows.append(
            {
                "name": name,
                "group_name": group_name,
                "site": site,
                "client_count": pd.to_numeric(client_count, errors="coerce"),
                "cpu_utilization": pd.to_numeric(cpu_util, errors="coerce"),
                "mem_free": pd.to_numeric(mem_free, errors="coerce"),
                "mem_total": pd.to_numeric(mem_total, errors="coerce"),
                "util_2g": max_2g,
                "util_5g": max_5g,
                "agg_util": agg_util,
            }
        )

    df = pd.DataFrame(rows)

    for column in [
        "client_count",
        "cpu_utilization",
        "mem_free",
        "mem_total",
        "util_2g",
        "util_5g",
        "agg_util",
    ]:
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["mem_used_pct"] = (1 - (df["mem_free"] / df["mem_total"])).clip(0, 1) * 100
    df["mem_used_pct"] = df["mem_used_pct"].fillna(0)

    return df


def read_geoloc_points(geojson_path: Path) -> pd.DataFrame:
    """Load AP geolocation data from GeoJSON."""

    with geojson_path.open("r", encoding="utf-8") as file:
        geojson = json.load(file)
    features = geojson.get("features", [])
    rows = []
    for feature in features:
        props = (feature or {}).get("properties", {})
        geometry = (feature or {}).get("geometry", {})
        if (geometry or {}).get("type") != "Point":
            continue
        coords = geometry.get("coordinates") or []
        if len(coords) < 2:
            continue
        name = props.get("USER_NOM_A")
        if not name:
            continue
        lon, lat = float(coords[0]), float(coords[1])
        rows.append({"name": name, "lon": lon, "lat": lat})
    return pd.DataFrame(rows)
