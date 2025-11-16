from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, MutableMapping, TypeVar


DEFAULT_PARAMS: dict[str, Any] = {
    "top_k": 3,
    "threshold": 0.6,
    "stress_profile": "HIGH",
    "snapshots": 5,
    "interference_radius": 50,
    "cca_increase": 0.15,
    "w_worst": 0.30,
    "w_avg": 0.30,
    "w_cov": 0.20,
    "w_neigh": 0.20,
    "candidate_mode": "Tile-based (uniform grid)",
    "merge_radius": 8,
    "interior_buffer_tiles": 2,
    "inner_clearance_m": 10,
}


def _get_numeric(source: Mapping[str, Any], key: str, default: float) -> float:
    raw_value = source.get(key, default)
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


def _get_int(source: Mapping[str, Any], key: str, default: int) -> int:
    raw_value = source.get(key, default)
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return default


def _get_str(source: Mapping[str, Any], key: str, default: str) -> str:
    raw_value = source.get(key, default)
    if isinstance(raw_value, str):
        return raw_value
    return default


@dataclass(frozen=True)
class SimulationParameters:
    top_k: int
    threshold: float
    stress_profile: str
    snapshots: int
    interference_radius: float
    cca_increase: float
    weight_worst: float
    weight_avg: float
    weight_cov: float
    weight_neigh: float
    candidate_mode: str
    merge_radius: float
    interior_buffer_tiles: int
    inner_clearance_m: float

    @classmethod
    def from_mapping(cls, source: Mapping[str, Any] | None) -> "SimulationParameters":
        src = source or {}
        return cls(
            top_k=_get_int(src, "top_k", DEFAULT_PARAMS["top_k"]),
            threshold=_get_numeric(src, "threshold", DEFAULT_PARAMS["threshold"]),
            stress_profile=_get_str(src, "stress_profile", DEFAULT_PARAMS["stress_profile"]),
            snapshots=_get_int(src, "snapshots", DEFAULT_PARAMS["snapshots"]),
            interference_radius=_get_numeric(src, "interference_radius", DEFAULT_PARAMS["interference_radius"]),
            cca_increase=_get_numeric(src, "cca_increase", DEFAULT_PARAMS["cca_increase"]),
            weight_worst=_get_numeric(src, "w_worst", DEFAULT_PARAMS["w_worst"]),
            weight_avg=_get_numeric(src, "w_avg", DEFAULT_PARAMS["w_avg"]),
            weight_cov=_get_numeric(src, "w_cov", DEFAULT_PARAMS["w_cov"]),
            weight_neigh=_get_numeric(src, "w_neigh", DEFAULT_PARAMS["w_neigh"]),
            candidate_mode=_get_str(src, "candidate_mode", DEFAULT_PARAMS["candidate_mode"]),
            merge_radius=_get_numeric(src, "merge_radius", DEFAULT_PARAMS["merge_radius"]),
            interior_buffer_tiles=_get_int(src, "interior_buffer_tiles", DEFAULT_PARAMS["interior_buffer_tiles"]),
            inner_clearance_m=_get_numeric(src, "inner_clearance_m", DEFAULT_PARAMS["inner_clearance_m"]),
        )

    def as_config_kwargs(self) -> dict[str, float | int | bool]:
        return {
            "interference_radius_m": self.interference_radius,
            "cca_increase_factor": self.cca_increase,
            "conflictivity_threshold_placement": self.threshold,
            "snapshots_per_profile": self.snapshots,
            "weight_worst_ap": self.weight_worst,
            "weight_average": self.weight_avg,
            "weight_coverage": self.weight_cov,
            "weight_neighborhood": self.weight_neigh,
        }


StressEnum = TypeVar("StressEnum", bound=Enum)


def resolve_stress_level(name: str, enum_type: type[StressEnum], fallback: StressEnum) -> StressEnum | None:
    normalized = name.upper().strip()
    if normalized == "ALL":
        return None
    for member in enum_type:
        if member.name == normalized:
            return member
    return fallback


def extract_simulation_params(state: MutableMapping[str, Any]) -> SimulationParameters:
    return SimulationParameters.from_mapping(state.get("sim_params"))
