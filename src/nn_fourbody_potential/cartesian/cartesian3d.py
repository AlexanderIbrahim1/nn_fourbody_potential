"""
This module contains the concrete Cartesian3D class. It is based on the original
'cartesian' package, but has been modified to reduce the number of dependencies
that this project depends on.
"""

from __future__ import annotations
from typing import Iterator

import dataclasses


@dataclasses.dataclass
class Cartesian3D:
    """Defines the interface for concrete Cartesian classes."""

    x: float
    y: float
    z: float

    @property
    def coordinates(self) -> Iterator:
        """Direct access to coordinates, mainly for iteration."""
        return iter([self.x, self.y, self.z])

    @property
    def n_dims(self) -> int:
        return 3

    def __hash__(self):
        return hash(self._coords())

    def __getitem__(self, i_dim: int) -> float:
        """Return the value of the `dim`th dimension of this point."""
        return self._coords()[i_dim]

    def __repr__(self) -> str:
        """Printed representation of the point as a comma-separated tuple."""
        _repr_inner = ", ".join([f"{coord:.6f}" for coord in self._coords()])
        return "(" + _repr_inner + ")"

    def __add__(self, other: Cartesian3D) -> Cartesian3D:
        """Element-wise addition of two points in cartesian space."""
        return Cartesian3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Cartesian3D) -> Cartesian3D:
        """Element-wise subtraction of two points in cartesian space."""
        return Cartesian3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other: float) -> Cartesian3D:
        """Scalar multiplication of the point in cartesian space by a number."""
        return Cartesian3D(other * self.x, other * self.y, other * self.z)

    def __rmul__(self, other: float) -> Cartesian3D:
        """Scalar multiplication of the point in cartesian space by a number."""
        return self.__mul__(other)

    def __floordiv__(self, other: float) -> Cartesian3D:
        return Cartesian3D(self.x // other, self.y // other, self.z // other)

    def __truediv__(self, other: float) -> Cartesian3D:
        return Cartesian3D(self.x / other, self.y / other, self.z / other)

    def _coords(self) -> tuple[float]:
        return (self.x, self.y, self.z)

    @classmethod
    def origin(cls) -> Cartesian3D:
        return cls(0.0, 0.0, 0.0)
