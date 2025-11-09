"""
Multi-Scenario AP Placement Simulator

Modular, robust simulator for WiFi AP placement optimization
across multiple temporal stress scenarios.
"""

from .config import SimulationConfig, StressLevel
from .stress_profiler import StressProfiler
from .multi_scenario_simulator import MultiScenarioSimulator
from .scoring import CompositeScorer, NeighborhoodOptimizationMode

__all__ = [
    'SimulationConfig',
    'StressLevel',
    'StressProfiler',
    'MultiScenarioSimulator',
    'CompositeScorer',
    'NeighborhoodOptimizationMode',
]

__version__ = '2.0.0'