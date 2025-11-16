from __future__ import annotations

from typing import Iterable

from .geometry import BaseGeometry


def unary_union(geoms: Iterable[BaseGeometry]) -> BaseGeometry: ...


def linemerge(geoms: Iterable[BaseGeometry] | BaseGeometry) -> BaseGeometry: ...


def snap(geom: BaseGeometry, reference: BaseGeometry, tolerance: float) -> BaseGeometry: ...
