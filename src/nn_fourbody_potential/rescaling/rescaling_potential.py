"""
This module contains the RescalingPotential class, which is an analytic function with
components similar to those of the true four-body potential, but with exaggerated
features.

Its purpose is to help rescale the true ab initio energies into a smaller range of values,
which should improve the effectiveness of training.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import math
import operator
import statistics


class PotentialType(enum.Enum):
    ARITHMETIC = enum.auto()
    GEOMETRIC = enum.auto()


@dataclasses.dataclass(frozen=True)
class RescalingPotential:
    coeff: float
    expon: float
    disp_coeff: float
    pot_type: PotentialType = PotentialType.ARITHMETIC

    def __post_init__(self) -> None:
        PT = PotentialType
        if self.pot_type == PT.ARITHMETIC:
            potential = ArithmeticRescalingPotential(self.coeff, self.expon, self.disp_coeff)
        elif self.pot_type == PT.GEOMETRIC:
            potential = GeometricRescalingPotential(self.coeff, self.expon, self.disp_coeff)

        object.__setattr__(self, "potential", potential)

    def __call__(self, *six_pair_distances: float) -> float:
        return self.potential(six_pair_distances)


@dataclasses.dataclass(frozen=True)
class ArithmeticRescalingPotential:
    coeff: float
    expon: float
    disp_coeff: float

    def __post_init__(self) -> None:
        assert self.coeff > 0.0
        assert self.expon > 0.0
        assert self.disp_coeff > 0.0

    def __call__(self, *six_pair_distances: float) -> float:
        assert len(six_pair_distances) == 6

        average_pairdist = statistics.fmean(six_pair_distances)

        expon_contribution = self.coeff * math.exp(-self.expon * average_pairdist)
        disp_contribution = self.disp_coeff / (average_pairdist**12)

        return expon_contribution + disp_contribution


@dataclasses.dataclass(frozen=True)
class GeometricRescalingPotential:
    coeff: float
    expon: float
    disp_coeff: float

    def __post_init__(self) -> None:
        assert self.coeff > 0.0
        assert self.expon > 0.0
        assert self.disp_coeff > 0.0

    def __call__(self, *six_pair_distances: float) -> float:
        assert len(six_pair_distances) == 6

        pairdist_product = functools.reduce(operator.mul, six_pair_distances, 1.0)
        geometric_average = pairdist_product ** (1.0 / 6.0)

        expon_contribution = self.coeff * math.exp(-self.expon * geometric_average)
        disp_contribution = self.disp_coeff / (pairdist_product**2)

        return expon_contribution + disp_contribution
