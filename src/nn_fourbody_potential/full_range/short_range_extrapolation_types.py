"""
This module contains POD types that are used for short-range extrapolation.
"""

from dataclasses import astuple
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from nn_fourbody_potential.sidelength_distributions import SixSideLengths


@dataclass
class ExtrapolationEnergies:
    lower: float
    upper: float

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class ExtrapolationSideLengths:
    lower: SixSideLengths
    upper: SixSideLengths

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class ExtrapolationDistanceInfo:
    r_short_range: float
    r_lower: float
    r_upper: float

    def delta_r(self) -> float:
        return self.r_upper - self.r_lower


@dataclass
class LinearEnergyExtrapolator:
    energies: ExtrapolationEnergies
    distances: ExtrapolationDistanceInfo

    @cached_property
    def slope(self) -> float:
        energy_lower, energy_upper = self.energies
        return (energy_upper - energy_lower) / self.distances.delta_r()

    @cached_property
    def energy(self) -> float:
        energy_lower = self.energies.lower
        dist_shift = self.distances.r_short_range - self.distances.r_lower
        return energy_lower + self.slope * dist_shift


@dataclass
class ExponentialEnergyExtrapolator:
    energies: ExtrapolationEnergies
    distances: ExtrapolationDistanceInfo

    @cached_property
    def slope(self) -> float:
        energies_lower, energies_upper = self.energies

        abs_energies_floor = 1.0e-6
        abs_energies_lower_floor = max(abs_energies_floor, abs(energies_lower))
        abs_energies_upper_floor = max(abs_energies_floor, abs(energies_upper))

        return -np.log(abs_energies_upper_floor / abs_energies_lower_floor) / self.distances.delta_r()

    @cached_property
    def energy(self) -> float:
        energies_lower = self.energies.lower
        dist_shift = self.distances.r_short_range - self.distances.r_lower
        return energies_lower * np.exp(-self.slope * dist_shift)

    @cached_property
    def is_magnitude_increasing_with_distance(self) -> float:
        return self.slope < 0.0
