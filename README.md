# 🏴‍☠️ Barretina Negra - UAB WiFi Dashboard

**UAB THE HACK!** 🦜 A pirate-themed hackathon to fix WiFi connectivity issues across the UAB campus.

## 🎯 Mission

Navigate the treacherous waters of campus WiFi and chart a course to optimal network performance! This integrated dashboard helps identify, analyze, and solve WiFi conflicts across the UAB campus using AI-powered analysis and advanced network simulation.

## ⚓ The Arsenal: Three Visualization Modes

Our integrated dashboard (`deliverable/integrated_dashboard.py`) provides three powerful tools to conquer WiFi conflicts:

### 1. 🤖 AI Heatmap - Monitor & Investigate
**Purpose**: Real-time monitoring and AI-powered cause investigation

- **Visual heatmap** of all campus Access Points (APs) colored by conflictivity (🟢 Low → 🔴 High)
- **Click any AP** to get instant AI analysis from AINA explaining:
  - Is it bandwidth saturation? (channel utilization > 40%)
  - Is it client overload? (approaching 50 concurrent clients)
  - Or both?
- **Time-series navigation** through historical snapshots
- **Conflictivity scoring** based on:
  - 75% airtime (channel congestion)
  - 15% client pressure
  - 10% AP resource health (CPU/memory)

### 2. 🗺️ Voronoi - Identify Conflict Zones
**Purpose**: Discover high-conflictivity regions using network topology

- **Interpolated conflict surfaces** showing coverage and stress across campus
- **Weighted Voronoi diagrams** that reveal connectivity boundaries
- **Hotspot detection** identifying the top 3 most conflictive Voronoi vertices
- **Coverage hull visualization** showing network reach
- Helps answer: "Where should we place new APs?"

### 3. 🎯 Simulator - Fix & Optimize
**Purpose**: Simulate AP placement to fix identified issues

- **Voronoi Candidate Discovery**: Detect stable high-conflictivity vertex clusters across multiple network scenarios
- **Multi-scenario testing**: Evaluate placements under LOW, MEDIUM, HIGH, and CRITICAL network stress
- **Interactive selection**: Pick candidate locations from a selectable table
- **Physics-based simulation**:
  - Client redistribution (RSSI-based)
  - Co-channel interference (CCA)
  - Conflictivity recalculation
- **Composite scoring** (worst AP improvement, average reduction, coverage, neighborhood impact)
- **Before/After map preview** showing simulated network state
- **Batch simulation** for multiple AP placements

## 🚀 Quick Start

### Prerequisites
Install [uv](https://docs.astral.sh/uv/) if you don't have it yet:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create and sync the local environment (this creates/updates `.venv` automatically):
```bash
uv sync
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
```

Need optional tooling? Add `--group ml`, `--group llm`, etc. to pull in the extras defined in `pyproject.toml`.

### Environment Setup
Create a `.env` file in the project root:
```
AINA_API_KEY=your_aina_api_key_here
```

### Run the Dashboard
```bash
uv run streamlit run deliverable/integrated_dashboard.py
```

Or with specific server settings:
```bash
uv run streamlit run deliverable/integrated_dashboard.py --server.headless true --server.port 8501
```

## 🗂️ Repository Structure

```
UAB-WiFi-Dataset-Analysis/
├── deliverable/
│   └── integrated_dashboard.py    # 🎯 Main unified dashboard
├── dashboard/
│   ├── conflictivity_dashboard_interpolation.py  # Voronoi algorithms
│   └── dashboard_voronoi_simulator.py            # Simulation engine
├── elies/
│   └── aina_dashboard.py          # AI integration
├── simulator/
│   ├── config.py                  # Simulation parameters
│   ├── stress_profiler.py         # Network stress classification
│   ├── scoring.py                 # Composite scoring system
│   └── spatial.py                 # Spatial utilities
├── realData/
│   ├── ap/                        # AP snapshot JSON files
│   └── geoloc/                    # Geolocation data (GeoJSON)
└── starter_kits/                  # Analysis notebooks
```

## 🧭 Workflow: Monitor → Identify → Simulate → Fix

1. **Monitor** (AI Heatmap): Observe current network state, click conflictive APs for AI diagnosis
2. **Identify** (Voronoi): Find high-conflictivity zones and Voronoi vertices
3. **Simulate** (Simulator): 
   - Detect Voronoi candidates
   - Select promising locations
   - Simulate AP placement
   - View before/after comparison
4. **Fix**: Deploy APs at recommended locations with confidence backed by multi-scenario testing

## 📊 Technical Details

### Conflictivity Model
```
conflictivity = 0.75 × airtime_score 
              + 0.15 × client_pressure 
              + 0.05 × cpu_health 
              + 0.05 × memory_health
```

**Airtime scoring**: Non-linear mapping with stricter thresholds for 2.4GHz (congestion-prone)
**Client pressure**: Logarithmic scale relative to snapshot's 95th percentile
**Resource health**: CPU/memory only matter when truly stressed (>70%/80%)

### Simulation Physics
- **Path Loss Model**: Log-distance with configurable exponent
- **RSSI Calculation**: Reference power - 10n log₁₀(d/d₀)
- **Client Redistribution**: Hybrid model (signal strength + distance + conflict)
- **CCA Interference**: Distance-proportional increase in neighbor utilization
- **Stress Profiles**: Automatic classification (LOW/MEDIUM/HIGH/CRITICAL)

### Voronoi Analysis
- **Weighted Voronoi**: Inverted conflictivity weights create connectivity boundaries
- **Candidate Clustering**: Merge nearby vertices within configurable radius
- **Multi-scenario Stability**: Only vertices appearing across multiple stress profiles

## 🔧 Configuration

Key parameters in the Simulator sidebar:
- **Conflictivity threshold**: Min stress level to consider (default: 0.6)
- **Test scenarios**: Number of snapshots per stress profile (default: 5)
- **Interference radius**: CCA impact range (default: 50m)
- **Scoring weights**: Balance worst-case vs. average improvement

## 📡 Data Format

### AP Snapshots
JSON files: `AP-info-v2-YYYY-MM-DDTHH_MM_SS.json`
```json
{
  "name": "AP-UAB-101",
  "client_count": 23,
  "cpu_utilization": 45.2,
  "mem_free": 128,
  "mem_total": 256,
  "radios": [
    {"band": 0, "utilization": 67.3},  // 2.4GHz
    {"band": 1, "utilization": 42.1}   // 5GHz
  ]
}
```

### Geolocation
GeoJSON: `aps_geolocalizados_wgs84.geojson`
```json
{
  "type": "Feature",
  "properties": {"USER_NOM_A": "AP-UAB-101"},
  "geometry": {"type": "Point", "coordinates": [2.1234, 41.5678]}
}
```

## 🏆 Team: Barretina Negra

Ahoy! We're charting the course to WiFi paradise. 🏴‍☠️

---

## 🔗 Related Info

### Field in Client Logs that relate to APs:
`associated_device` — matches the AP's macaddr field

**Example**: Client has `"associated_device": "AP_8e2d9933ec92"` → matches AP's `"macaddr": "AP_8e2d9933ec92"`
