"""
AI Heatmap Dashboard Module

Purpose:
    Provides clickable AP heatmap visualization with AINA AI analysis integration.
    Displays WiFi access points with conflictivity scoring and enables interactive
    AI-powered diagnostics.

Features:
    - Optimized geospatial heatmap with Plotly Scattermapbox
    - Location-based AP grouping (multiple APs at same coordinates)
    - AINA AI analysis dialog with comprehensive AP metrics
    - Configurable conflictivity threshold filtering

Usage:
    from dashboard.ai_heatmap import create_optimized_heatmap, render_heatmap_controls
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore[import-untyped]
import requests
import streamlit as st


# ======== CONFIGURATION ========

@dataclass
class HeatmapConfig:
    """Configuration for heatmap visualization."""
    
    min_conflictivity: float = 0.0
    marker_radius: int = 15
    default_zoom: int = 15
    opacity: float = 0.85
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        assert 0.0 <= self.min_conflictivity <= 1.0, "min_conflictivity must be in [0, 1]"
        assert self.marker_radius > 0, "marker_radius must be positive"
        assert self.default_zoom > 0, "default_zoom must be positive"
        assert 0.0 < self.opacity <= 1.0, "opacity must be in (0, 1]"


@dataclass
class APMetrics:
    """Structured AP metrics for display and analysis."""
    
    name: str
    conflictivity: float
    lat: float
    lon: float
    util_2g: float | None = None
    util_5g: float | None = None
    client_count: int = 0
    cpu_utilization: float | None = None
    mem_free: float | None = None
    mem_total: float | None = None
    mem_used_pct: float | None = None
    max_radio_util: float | None = None
    
    def format_metric(self, value: float | None, format_str: str = "{:.1f}", default: str = "N/A") -> str:
        """Format a metric value with fallback for missing data."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return default
        return format_str.format(value)
    
    def to_display_text(self) -> str:
        """Generate formatted display text for AP metrics."""
        lines = [
            f"- **Nom:** {self.name}",
            f"- **Utilització 2.4 GHz:** {self.format_metric(self.util_2g, '{:.1f}%')}",
            f"- **Utilització 5 GHz:** {self.format_metric(self.util_5g, '{:.1f}%')}",
            f"- **Clients connectats:** {self.client_count}",
            f"- **Utilització CPU:** {self.format_metric(self.cpu_utilization, '{:.1f}%')}",
            f"- **Memòria lliure:** {self.format_metric(self.mem_free, '{:.0f} MB')}",
            f"- **Memòria total:** {self.format_metric(self.mem_total, '{:.0f} MB')}",
            f"- **Memòria usada (%):** {self.format_metric(self.mem_used_pct, '{:.1f}%')}",
            f"- **Conflictivitat:** {self.format_metric(self.conflictivity, '{:.3f}')}",
        ]
        return "\n".join(lines)


# ======== HELPER FUNCTIONS ========

def _group_aps_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """Group APs at same geographic location.
    
    Args:
        df: DataFrame with columns: lat, lon, name, conflictivity, and optional metrics
        
    Returns:
        DataFrame with grouped location data including aggregated AP lists
        
    Preconditions:
        - df must contain columns: lat, lon, name, conflictivity
        - lat/lon values must be valid coordinates
    """
    assert len(df) > 0, "DataFrame must not be empty"
    assert all(col in df.columns for col in ["lat", "lon", "name", "conflictivity"]), \
        "DataFrame must contain required columns"
    
    df_copy = df.copy()
    
    # Create location key with 6 decimal precision (~10cm accuracy)
    df_copy["location_key"] = (
        df_copy["lat"].round(6).astype(str) + "," + df_copy["lon"].round(6).astype(str)
    )
    
    # Define aggregation functions with type annotations
    def to_list(series: pd.Series[Any]) -> list[Any]:
        """Convert pandas Series to list."""
        return list(series)
    
    # Define aggregation with explicit types
    agg_dict: dict[str, tuple[str, Any]] = {
        "lat": ("lat", "first"),
        "lon": ("lon", "first"),
        "name": ("name", to_list),
        "conflictivity": ("conflictivity", to_list),
    }
    
    # Add optional columns if present
    if "client_count" in df.columns:
        agg_dict["client_count"] = ("client_count", to_list)
    if "max_radio_util" in df.columns:
        agg_dict["max_radio_util"] = ("max_radio_util", to_list)
    
    # Note: groupby/agg returns partially unknown types due to pandas-stubs limitations
    grouped = df_copy.groupby("location_key").agg(**agg_dict).reset_index()  # type: ignore[misc]
    
    # Calculate derived metrics with explicit type casting
    def get_max_conflictivity(conf_list: Any) -> float:
        """Extract maximum conflictivity from list."""
        return float(max(cast(list[float], conf_list)))
    
    # Note: apply() returns partially unknown types due to pandas-stubs limitations
    grouped["max_conflictivity"] = grouped["conflictivity"].apply(get_max_conflictivity)  # type: ignore[misc]
    grouped["ap_count"] = grouped["name"].apply(len)  # type: ignore[misc]
    
    return grouped


def _create_hover_texts(grouped_df: pd.DataFrame) -> tuple[list[str], list[list[str]]]:
    """Generate hover text and AP name lists for map markers.
    
    Args:
        grouped_df: DataFrame from _group_aps_by_location
        
    Returns:
        Tuple of (hover_texts, ap_names_list) for each location
    """
    assert len(grouped_df) > 0, "Grouped DataFrame must not be empty"
    
    hover_texts: list[str] = []
    ap_names_list: list[list[str]] = []
    
    for _, row in grouped_df.iterrows():
        names: list[str] = row["name"]
        conflicts: list[float] = row["conflictivity"]
        clients: list[Any] | None = row.get("client_count")
        utils: list[Any] | None = row.get("max_radio_util")
        
        # Build AP data tuples
        ap_data = list(zip(
            names,
            conflicts,
            clients if clients is not None else [None] * len(names),
            utils if utils is not None else [None] * len(names),
        ))
        
        # Sort by conflictivity (descending)
        ap_data.sort(key=lambda x: x[1], reverse=True)
        
        if len(ap_data) == 1:
            # Single AP at location
            hover_text = _format_single_ap_hover(*ap_data[0])
            ap_names_list.append([ap_data[0][0]])
        else:
            # Multiple APs at location
            hover_text = _format_multi_ap_hover(ap_data)
            ap_names_list.append([ap[0] for ap in ap_data])
        
        hover_texts.append(hover_text)
    
    return hover_texts, ap_names_list


def _format_single_ap_hover(
    name: str, conflictivity: float, clients: Any, util: Any
) -> str:
    """Format hover text for single AP at location."""
    text = f"<b>{name}</b><br>Conflictivity: {conflictivity:.3f}"
    
    if clients is not None and not np.isnan(clients):
        text += f"<br>Clients: {int(clients)}"
    
    if util is not None and not np.isnan(util):
        text += f"<br>Radio Util: {util:.1f}%"
    
    return text


def _format_multi_ap_hover(ap_data: list[tuple[str, float, Any, Any]]) -> str:
    """Format hover text for multiple APs at same location."""
    text = f"<b>{len(ap_data)} APs at this location</b><br><br>"
    
    for i, (name, conflictivity, clients, util) in enumerate(ap_data):
        text += f"<b>{name}</b><br>  Conflictivity: {conflictivity:.3f}"
        
        if clients is not None and not np.isnan(clients):
            text += f" | Clients: {int(clients)}"
        
        if util is not None and not np.isnan(util):
            text += f" | Radio: {util:.1f}%"
        
        if i < len(ap_data) - 1:
            text += "<br>"
    
    return text


# ======== MAIN VISUALIZATION FUNCTION ========

def create_optimized_heatmap(
    df: pd.DataFrame,
    center_lat: float,
    center_lon: float,
    config: HeatmapConfig | None = None,
) -> go.Figure:
    """Create interactive heatmap with clickable AP points.
    
    Args:
        df: DataFrame with AP data (must include lat, lon, name, conflictivity)
        center_lat: Map center latitude
        center_lon: Map center longitude
        config: Optional configuration (uses defaults if None)
        
    Returns:
        Plotly Figure object with configured heatmap
        
    Preconditions:
        - df must contain at least: lat, lon, name, conflictivity columns
        - center_lat must be in [-90, 90]
        - center_lon must be in [-180, 180]
    """
    assert len(df) > 0, "DataFrame must not be empty"
    assert -90 <= center_lat <= 90, "center_lat must be valid latitude"
    assert -180 <= center_lon <= 180, "center_lon must be valid longitude"
    
    if config is None:
        config = HeatmapConfig()
    
    # Group APs by location
    grouped = _group_aps_by_location(df)
    
    # Filter by minimum conflictivity
    if config.min_conflictivity > 0.0:
        grouped = grouped[grouped["max_conflictivity"] >= config.min_conflictivity]
    
    # Sort for z-order (lower conflictivity draws first)
    grouped = grouped.sort_values("max_conflictivity", ascending=True)
    
    # Generate hover texts
    hover_texts, ap_names_list = _create_hover_texts(grouped)
    
    # Create figure
    fig = go.Figure(
        go.Scattermapbox(
            lat=grouped["lat"].tolist(),
            lon=grouped["lon"].tolist(),
            mode="markers",
            marker={
                "size": config.marker_radius * 2,
                "color": grouped["max_conflictivity"].tolist(),
                "colorscale": [
                    [0.0, "rgb(0, 255, 0)"],
                    [0.5, "rgb(255, 165, 0)"],
                    [1.0, "rgb(255, 0, 0)"],
                ],
                "cmin": 0,
                "cmax": 1,
                "opacity": config.opacity,
                "showscale": True,
                "colorbar": {
                    "title": "Conflictivity",
                    "thickness": 15,
                    "len": 0.7,
                    "tickmode": "linear",
                    "tick0": 0,
                    "dtick": 0.2,
                    "tickformat": ".1f",
                },
            },
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
            customdata=ap_names_list,
        )
    )
    
    # Note: update_layout has partially unknown types due to plotly stub limitations
    fig.update_layout(  # type: ignore[misc]
        mapbox={
            "style": "open-street-map",
            "center": {"lat": center_lat, "lon": center_lon},
            "zoom": config.default_zoom,
        },
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        height=700,
    )
    
    return fig


# ======== AINA AI INTEGRATION ========

def _build_aina_prompt(metrics: APMetrics) -> str:
    """Build AINA API prompt with AP metrics and context.
    
    Args:
        metrics: Structured AP metrics
        
    Returns:
        Complete prompt string for AINA API
    """
    base_info = f"""Dades de l'Access Point:

{metrics.to_display_text()}

Aquest Access Point és conflictiu. Investiga les causes tenint en compte aquests criteris:

**Model de Conflictivitat:**

Entrades per AP:
- util_2g, util_5g: utilització màxima del canal per banda
- client_count: nombre de clients connectats
- cpu_utilization: percentatge d'ús de CPU
- mem_used_pct: percentatge de memòria usada

Passos del càlcul:
1. airtime_score per banda (2g/5g):
   - Si util < 20%: score = 0
   - Si 20% ≤ util < 50%: score proporcional
   - Si util ≥ 50%: score = 1
   - Ajustament: si 0 clients, score *= 0.8

2. airtime_score_adj: promig ponderat de 2g (w=1) i 5g (w=2)

3. client_score (basat en percentil 95 de clients):
   - Si clients < p95*0.5: score = 0
   - Creixement lineal fins p95: score = 1

4. cpu_score i mem_score (similars a client_score amb llindars)

5. **Conflictivitat final** (pesos):
   - airtime: 75%
   - clients: 15%
   - CPU: 5%
   - memòria: 5%

**Tasca:** Analitza les mètriques d'aquest AP i explica:
1. Quins factors contribueixen més a la conflictivitat
2. Possibles causes dels problemes detectats
3. Recomanacions específiques per millorar el rendiment

Respon en català, de manera clara i concisa."""
    
    return base_info


def query_aina_api(prompt: str, api_key: str) -> str:
    """Query AINA AI API with prompt.
    
    Args:
        prompt: Analysis prompt
        api_key: AINA API authentication key
        
    Returns:
        AI analysis response text
        
    Raises:
        requests.RequestException: If API call fails
    """
    assert len(prompt) > 0, "Prompt must not be empty"
    assert len(api_key) > 0, "API key must not be empty"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "UAB-WiFi-Analysis/1.0",
    }
    
    payload = {
        "model": "aina",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.7,
    }
    
    response = requests.post(
        "https://api.aina.bsc.es/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30,
    )
    response.raise_for_status()
    
    data = response.json()
    return str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))


def show_aina_analysis_dialog(ap_row: pd.Series) -> None:
    """Display AINA AI analysis dialog for selected AP.
    
    Args:
        ap_row: Pandas Series with AP metrics
        
    Side Effects:
        - Opens Streamlit dialog
        - Makes API call to AINA
        - Displays analysis results
    """
    # Extract metrics from row
    metrics = APMetrics(
        name=str(ap_row.get("name", "Unknown")),
        conflictivity=float(ap_row.get("conflictivity", 0.0)),
        lat=float(ap_row.get("lat", 0.0)),
        lon=float(ap_row.get("lon", 0.0)),
        util_2g=ap_row.get("util_2g"),
        util_5g=ap_row.get("util_5g"),
        client_count=int(ap_row.get("client_count", 0)),
        cpu_utilization=ap_row.get("cpu_utilization"),
        mem_free=ap_row.get("mem_free"),
        mem_total=ap_row.get("mem_total"),
        mem_used_pct=ap_row.get("mem_used_pct"),
        max_radio_util=ap_row.get("max_radio_util"),
    )
    
    st.subheader(f"Access Point: {metrics.name}")
    
    # Display metrics in expander
    with st.expander("📊 Dades de l'Access Point", expanded=False):
        st.markdown(metrics.to_display_text())
    
    # Check for API key
    api_key = os.getenv("AINA_API_KEY", "")
    if not api_key:
        st.error(
            "❌ AINA_API_KEY no trobada. "
            "Crea un fitxer `.env` amb `AINA_API_KEY=<la_teva_clau>`"
        )
        return
    
    # Query AINA API
    st.subheader("🤖 Anàlisi AINA AI")
    
    with st.spinner("Consultant AINA AI..."):
        try:
            prompt = _build_aina_prompt(metrics)
            analysis = query_aina_api(prompt, api_key)
            
            st.markdown(analysis)
            
        except requests.RequestException as e:
            st.error(f"❌ Error en cridar l'API d'AINA: {e}")
        except Exception as e:
            st.error(f"❌ Error inesperat: {e}")


# ======== STREAMLIT UI HELPERS ========

def render_heatmap_controls() -> tuple[int, float, int]:
    """Render Streamlit sidebar controls for heatmap configuration.
    
    Returns:
        Tuple of (marker_radius, min_conflictivity, top_n_listing)
    """
    radius = st.slider(
        "Marker Radius",
        min_value=3,
        max_value=20,
        value=5,
        step=1,
        help="Size of AP markers on map"
    )
    
    min_conf = st.slider(
        "Minimum Conflictivity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help="Filter APs below this threshold"
    )
    
    top_n = st.slider(
        "Top N Listing (table)",
        min_value=5,
        max_value=50,
        value=15,
        step=5,
        help="Number of top conflictive APs to display"
    )
    
    return radius, min_conf, top_n
