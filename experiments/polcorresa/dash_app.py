import os
import glob
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State

# ---------- Paths ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
AP_DIR = os.path.join(ROOT, 'realData', 'ap')
GEOJSON_PATH = os.path.join(ROOT, 'realData', 'geoloc', 'aps_geolocalizados_wgs84.geojson')

# ---------- Data helpers ----------

def norm01(values: List[float]) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        return [0.5] * len(values)
    return [(v - vmin) / (vmax - vmin) for v in values]


def list_snapshots() -> List[str]:
    return sorted(glob.glob(os.path.join(AP_DIR, 'AP-info-v2-*.json')))


def parse_ts_from_filename(path: str) -> datetime:
    base = os.path.basename(path)
    ts_part = base[len('AP-info-v2-'):-len('.json')]
    # e.g. 2025-04-03T08_25_01+02_00 -> split off tz
    naive = ts_part.split('+', 1)[0]
    return datetime.strptime(naive, '%Y-%m-%dT%H_%M_%S')


def read_snapshot(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
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


GEOLOC = read_geoloc_points(GEOJSON_PATH)
SNAPSHOTS = list_snapshots()


def compute_conflictivity_frame(ap_records: List[Dict[str, Any]]) -> pd.DataFrame:
    names: List[str] = []
    clients: List[int] = []
    utils_max: List[float] = []

    for rec in ap_records:
        name = rec.get('name') or rec.get('ap_name') or rec.get('id')
        names.append(name)
        clients.append(int(rec.get('client_count') or 0))
        # max utilization across radios
        max_u = 0.0
        for r in (rec.get('radios') or []):
            try:
                u = float(r.get('utilization') or 0.0)
            except (TypeError, ValueError):
                u = 0.0
            if u > max_u:
                max_u = u
        utils_max.append(max_u)

    client_norm = norm01(clients)
    util_norm = [(u or 0.0) / 100.0 for u in utils_max]
    conflictivity = [0.6 * c + 0.4 * u for c, u in zip(client_norm, util_norm)]

    # Join geoloc
    lats, lons, bldg, floor = [], [], [], []
    for n in names:
        if n in GEOLOC:
            lat, lon, prop = GEOLOC[n]
            lats.append(lat)
            lons.append(lon)
            bldg.append(prop.get('Nom_Edific') or prop.get('USER_EDIFI'))
            floor.append(prop.get('USER_PLANT'))
        else:
            lats.append(None)
            lons.append(None)
            bldg.append(None)
            floor.append(None)

    df = pd.DataFrame({
        'name': names,
        'client_count': clients,
        'utilization_max': utils_max,
        'conflictivity': conflictivity,
        'lat': lats,
        'lon': lons,
        'building': bldg,
        'floor': floor,
    })

    # Drop rows without coordinates
    df = df.dropna(subset=['lat', 'lon'])
    return df


# ---------- Dash app ----------
app = Dash(__name__)
app.title = 'UAB WiFi Conflictivity (Dash)'

snapshot_options = [
    {
        'label': f"{parse_ts_from_filename(p).isoformat()} — {os.path.basename(p)}",
        'value': p
    }
    for p in SNAPSHOTS
]

initial_snapshot = snapshot_options[-1]['value'] if snapshot_options else None

app.layout = html.Div([
    html.H2('UAB WiFi Conflictivity Dashboard'),

    html.Div([
        html.Div([
            html.Label('Snapshot'),
            dcc.Dropdown(
                id='snapshot',
                options=snapshot_options,
                value=initial_snapshot,
                clearable=False,
            ),
        ], style={'flex': '3', 'minWidth': 300, 'marginRight': 16}),

        html.Div([
            html.Label('Map type'),
            dcc.RadioItems(
                id='map-type',
                options=[
                    {'label': 'Heatmap', 'value': 'heat'},
                    {'label': 'Points', 'value': 'points'},
                ],
                value='heat',
                inline=True,
            ),
        ], style={'flex': '2', 'minWidth': 200, 'marginRight': 16}),

        html.Div([
            html.Label('Min conflictivity'),
            dcc.Slider(id='min-conf', min=0.0, max=1.0, step=0.05, value=0.0,
                       marks={0: '0.0', 0.5: '0.5', 1.0: '1.0'}),
        ], style={'flex': '4', 'minWidth': 300}),

        html.Div([
            html.Label('Top N (table)'),
            dcc.Input(id='top-n', type='number', min=5, max=200, step=5, value=25,
                      style={'width': '100%'}),
        ], style={'flex': '1', 'minWidth': 120, 'marginLeft': 16}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'alignItems': 'center'}),

    dcc.Graph(id='map-graph', style={'height': '70vh'}),

    html.H3('Top APs by Conflictivity'),
    html.Div(id='top-table'),

    html.Div(id='footer', children=[
        html.Small('Conflictivity = 0.6 * normalized clients + 0.4 * (max utilization / 100). ' \
                   'Geo: aps_geolocalizados_wgs84.geojson | Data: realData/ap/*.json')
    ], style={'marginTop': 16, 'color': '#666'})
])


@app.callback(
    Output('map-graph', 'figure'),
    Output('top-table', 'children'),
    Input('snapshot', 'value'),
    Input('map-type', 'value'),
    Input('min-conf', 'value'),
    Input('top-n', 'value'),
)
def update_view(snapshot_path, map_type, min_conf, top_n):
    if not snapshot_path:
        return px.scatter_mapbox(), html.Div('No snapshot found')

    records = read_snapshot(snapshot_path)
    df = compute_conflictivity_frame(records)

    # Filter by min conflictivity
    if min_conf is None:
        min_conf = 0.0
    df_f = df[df['conflictivity'] >= float(min_conf)].copy()

    center = dict(lat=41.5009, lon=2.1098)
    if map_type == 'heat':
        fig = px.density_mapbox(
            df_f,
            lat='lat', lon='lon', z='conflictivity',
            radius=30,
            center=center,
            zoom=14.5,
            mapbox_style='open-street-map',
            color_continuous_scale='RdYlGn_r',
            title=f'Conflictivity Heatmap — {os.path.basename(snapshot_path)}'
        )
    else:
        sizes = [max(6, 6 + c / 10.0) for c in df_f['client_count']]
        fig = px.scatter_mapbox(
            df_f,
            lat='lat', lon='lon',
            color='conflictivity',
            size=sizes,
            hover_name='name',
            hover_data={'client_count': True, 'utilization_max': True, 'conflictivity': ':.2f', 'lat': False, 'lon': False},
            center=center,
            zoom=14.5,
            mapbox_style='open-street-map',
            color_continuous_scale='RdYlGn_r',
            title=f'Conflictivity by AP — {os.path.basename(snapshot_path)}'
        )

    # Build top table
    df_top = df.sort_values('conflictivity', ascending=False).head(int(top_n or 25))
    rows = []
    rows.append(html.Tr([html.Th('#'), html.Th('AP'), html.Th('Building'), html.Th('Floor'),
                         html.Th('Clients'), html.Th('Util%'), html.Th('Conflictivity')]))
    for idx, row in enumerate(df_top.itertuples(index=False), start=1):
        rows.append(html.Tr([
            html.Td(idx),
            html.Td(row.name),
            html.Td(row.building or ''),
            html.Td(row.floor or ''),
            html.Td(int(row.client_count)),
            html.Td(f"{row.utilization_max:.0f}"),
            html.Td(f"{row.conflictivity:.2f}"),
        ]))
    table = html.Table(rows, style={'borderCollapse': 'collapse', 'width': '100%'},
                       className='top-table')

    return fig, table


if __name__ == '__main__':
    port = int(os.environ.get('PORT', '8050'))
    app.run(host='127.0.0.1', port=port, debug=False)
