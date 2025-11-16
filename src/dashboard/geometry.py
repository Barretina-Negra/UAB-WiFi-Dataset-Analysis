from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol


class CoordinateSequence(Protocol):
    """Protocol describing the minimal interface of a coordinate sequence."""

    def __iter__(self) -> Iterator[tuple[float, float]]:  # pragma: no cover - protocol
        ...


class LinearRingLike(Protocol):
    """Protocol for ring-like boundary structures used by polygons."""

    coords: CoordinateSequence


class PolygonLike(Protocol):
    geom_type: str
    exterior: LinearRingLike
    interiors: Iterable[LinearRingLike]


class MultiPolygonLike(Protocol):
    geom_type: str
    geoms: Iterable[PolygonLike]


GeometryLike = PolygonLike | MultiPolygonLike


@dataclass(frozen=True)
class CoverageHull:
    """Structured representation of lon/lat coordinates for coverage hull plots."""

    longitudes: list[float | None]
    latitudes: list[float | None]
    polygon_count: int

    def has_coordinates(self) -> bool:
        return bool(self.longitudes and self.latitudes)


def _append_ring_coordinates(ring: LinearRingLike, lons: list[float | None], lats: list[float | None]) -> None:
    """Append ring coordinates to lon/lat lists with None separator.
    
    Args:
        ring: Ring-like boundary structure
        lons: List to append longitudes to
        lats: List to append latitudes to
    """
    # Shapely stubs incomplete - coords iteration not fully typed
    coords: list[tuple[float, float]] = list(ring.coords)  # type: ignore[arg-type,assignment]
    if not coords:
        return
    for lon, lat in coords:
        lons.append(float(lon))
        lats.append(float(lat))
    lons.append(None)
    lats.append(None)


def extract_coverage_hull(geometry: GeometryLike | None) -> CoverageHull | None:
    """Extract coordinate buffers for the provided polygon or multipolygon.
    
    Args:
        geometry: Polygon or MultiPolygon geometry to extract coordinates from
        
    Returns:
        CoverageHull with extracted coordinates or None if geometry is empty
    """
    if geometry is None:
        return None

    polygons: list[PolygonLike]
    if geometry.geom_type == "Polygon":
        polygons = [geometry]  # type: ignore[list-item]
    else:
        # Shapely stubs incomplete - geoms attribute and iteration not fully typed in Protocol
        polygons = list(geometry.geoms)  # type: ignore[arg-type,attr-defined,union-attr]

    if not polygons:
        return None

    hull_lons: list[float | None] = []
    hull_lats: list[float | None] = []

    for polygon in polygons:
        _append_ring_coordinates(polygon.exterior, hull_lons, hull_lats)
        for ring in polygon.interiors:
            _append_ring_coordinates(ring, hull_lons, hull_lats)

    if not hull_lons or not hull_lats:
        return None

    return CoverageHull(longitudes=hull_lons, latitudes=hull_lats, polygon_count=len(polygons))
