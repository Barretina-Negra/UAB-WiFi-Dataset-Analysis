import json
import os
import glob
from datetime import datetime
from typing import Dict, Any, List, Tuple

import plotly.express as px

# ---------- Config ----------
AP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'realData', 'ap'))
GEOJSON_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'realData', 'geoloc', 'aps_geolocalizados_wgs84.geojson'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'preview'))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Helpers ----------

def norm01(values: List[float]) -> Dict[int, float]:
    vmin = min(values) if values else 0.0
    vmax = max(values) if values else 1.0
    if vmax == vmin:
        return {i: 0.5 for i in range(len(values))}
    return {i: (v - vmin) / (vmax - vmin) for i, v in enumerate(values)}


def list_snapshots() -> List[str]:
    files = sorted(glob.glob(os.path.join(AP_DIR, 'AP-info-v2-*.json')))
    return files


def parse_ts_from_filename(path: str) -> datetime:
    base = os.path.basename(path)
    # Example: AP-info-v2-2025-04-03T08_25_01+02_00.json
    ts_part = base[len('AP-info-v2-'):-len('.json')]
    # Split on '+' to separate naive time and offset we ignore for ordering
    if '+' in ts_part:
        naive, _offset = ts_part.split('+', 1)
    else:
        naive = ts_part
    # naive e.g. 2025-04-03T08_25_01
    return datetime.strptime(naive, '%Y-%m-%dT%H_%M_%S')


def read_snapshot(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Expect list of AP dicts. If wrapped, try common keys.
    if isinstance(data, dict):
        for key in ('aps', 'data', 'items', 'results'):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise ValueError('Unexpected AP snapshot JSON structure')
    return data


def read_geoloc_points(path: str) -> Dict[str, Tuple[float, float, Dict[str, Any]]]:
    with open(path, 'r', encoding='utf-8') as f:
        gj = json.load(f)
    mapping: Dict[str, Tuple[float, float, Dict[str, Any]]] = {}
    for feat in gj.get('features', []):
        prop = feat.get('properties', {}) or {}
        geom = feat.get('geometry', {}) or {}
        coords = geom.get('coordinates') or [None, None]
        name = prop.get('USER_NOM_A')
        if not name:
            continue
        lon, lat = coords[0], coords[1]
        if lon is None or lat is None:
            continue
        mapping[name] = (lat, lon, prop)
    return mapping


# ---------- Core ----------

def compute_conflictivity(ap_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Extract raw metrics
    clients = [int(rec.get('client_count') or 0) for rec in ap_records]
    # Max utilization across radios per AP
    utils = []
    for rec in ap_records:
        radios = rec.get('radios') or []
        max_u = 0.0
        for r in radios:
            u = r.get('utilization')
            try:
                u = float(u)
            except (TypeError, ValueError):
                u = 0.0
            max_u = max(max_u, u)
        utils.append(max_u)

    clients_n = norm01(clients)
    # utilization is already 0..100
    out = []
    for i, rec in enumerate(ap_records):
        name = rec.get('name') or rec.get('ap_name') or rec.get('id')
        client_norm = clients_n.get(i, 0.0)
        util_norm = (utils[i] or 0.0) / 100.0
        conflict = 0.6 * client_norm + 0.4 * util_norm
        out.append({
            'name': name,
            'client_count': clients[i],
            'utilization_max': utils[i],
            'conflictivity': conflict,
        })
    return out


def join_with_geoloc(metrics: List[Dict[str, Any]], geoloc: Dict[str, Tuple[float, float, Dict[str, Any]]]):
    rows = []
    for m in metrics:
        name = m['name']
        if name in geoloc:
            lat, lon, prop = geoloc[name]
            rows.append({
                'name': name,
                'lat': lat,
                'lon': lon,
                'client_count': m['client_count'],
                'utilization_max': m['utilization_max'],
                'conflictivity': m['conflictivity'],
                'building': prop.get('Nom_Edific') or prop.get('USER_EDIFI'),
                'floor': prop.get('USER_PLANT'),
            })
    return rows


def build_maps(rows: List[Dict[str, Any]], ts_label: str):
    if not rows:
        raise SystemExit('No rows to plot after joining geoloc + metrics.')
    # Sort by conflictivity for top list
    rows_sorted = sorted(rows, key=lambda r: r['conflictivity'], reverse=True)

    # Heatmap
    fig_heat = px.density_mapbox(
        rows,
        lat='lat', lon='lon', z='conflictivity',
        radius=30,
        center=dict(lat=41.5009, lon=2.1098),
        zoom=14.5,
        mapbox_style='open-street-map',
        color_continuous_scale='RdYlGn_r',
        title=f'Conflictivity Heatmap — {ts_label}'
    )

    # Points
    fig_pts = px.scatter_mapbox(
        rows,
        lat='lat', lon='lon',
        color='conflictivity',
        size=[max(6, 6 + r['client_count'] / 10) for r in rows],
        hover_name='name',
        hover_data={'client_count': True, 'utilization_max': True, 'conflictivity': ':.2f', 'lat': False, 'lon': False},
        center=dict(lat=41.5009, lon=2.1098),
        zoom=14.5,
        mapbox_style='open-street-map',
        color_continuous_scale='RdYlGn_r',
        title=f'Conflictivity by AP — {ts_label}'
    )

    # Save
    heat_path = os.path.join(OUTPUT_DIR, 'conflictivity_heatmap.html')
    pts_path = os.path.join(OUTPUT_DIR, 'conflictivity_points.html')
    fig_heat.write_html(heat_path, include_plotlyjs='cdn')
    fig_pts.write_html(pts_path, include_plotlyjs='cdn')

    # Also write a small top list
    top_rows = rows_sorted[:25]
    lines = [
        '<h2>Top APs by Conflictivity</h2>',
        '<table border="1" cellspacing="0" cellpadding="6">',
        '<tr><th>#</th><th>AP</th><th>Building</th><th>Floor</th><th>Clients</th><th>Util%</th><th>Conflictivity</th></tr>'
    ]
    for i, r in enumerate(top_rows, start=1):
        lines.append(
            f"<tr><td>{i}</td><td>{r['name']}</td><td>{r.get('building','')}</td><td>{r.get('floor','')}</td>"
            f"<td>{r['client_count']}</td><td>{r['utilization_max']:.0f}</td><td>{r['conflictivity']:.2f}</td></tr>"
        )
    lines.append('</table>')
    with open(os.path.join(OUTPUT_DIR, 'conflictivity_top.html'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return heat_path, pts_path


if __name__ == '__main__':
    snaps = list_snapshots()
    if not snaps:
        raise SystemExit(f'No AP snapshots found in {AP_DIR}')
    latest = sorted(snaps, key=parse_ts_from_filename)[-1]
    ts = parse_ts_from_filename(latest).isoformat()

    print(f'Using snapshot: {os.path.basename(latest)} ({ts})')

    ap_data = read_snapshot(latest)
    metrics = compute_conflictivity(ap_data)
    geoloc = read_geoloc_points(GEOJSON_PATH)
    rows = join_with_geoloc(metrics, geoloc)

    heat, pts = build_maps(rows, ts)
    print('Generated:')
    print(' -', heat)
    print(' -', pts)
    print(' -', os.path.join(OUTPUT_DIR, 'conflictivity_top.html'))
