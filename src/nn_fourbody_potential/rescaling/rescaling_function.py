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

from nn_fourbody_potential.constants import NUMBER_OF_SIDELENGTHS_FOURBODY


class RescalingFunctionType(enum.Enum):
    ARITHMETIC = enum.auto()
    GEOMETRIC = enum.auto()


@dataclasses.dataclass(frozen=True)
class RescalingFunction:
    coeff: float
    expon: float
    disp_coeff: float
    pot_type: RescalingFunctionType = RescalingFunctionType.ARITHMETIC

    def __post_init__(self) -> None:
        PT = RescalingFunctionType
        if self.pot_type == PT.ARITHMETIC:
            potential = ArithmeticRescalingFunction(self.coeff, self.expon, self.disp_coeff)
        elif self.pot_type == PT.GEOMETRIC:
            potential = GeometricRescalingFunction(self.coeff, self.expon, self.disp_coeff)
        else:
            assert False, "unreachable"

        object.__setattr__(self, "potential", potential)

    def __call__(self, *six_pair_distances: float) -> float:
        return self.potential(*six_pair_distances)  # type: ignore


@dataclasses.dataclass(frozen=True)
class ArithmeticRescalingFunction:
    coeff: float
    expon: float
    disp_coeff: float

    def __post_init__(self) -> None:
        assert self.coeff > 0.0
        assert self.expon > 0.0
        assert self.disp_coeff > 0.0

    def __call__(self, *six_pair_distances: float) -> float:
        assert len(six_pair_distances) == NUMBER_OF_SIDELENGTHS_FOURBODY

        average_pairdist = statistics.fmean(six_pair_distances)

        # tanh_arg = 0.1 * (average_pairdist - 3.0)
        # frac_disp = 0.5 * (1.0 + math.tanh(tanh_arg))
        # frac_expon = 1.0 - frac_disp

        frac_disp = 1.0
        frac_expon = 1.0

        expon_contribution = frac_expon * self.coeff * math.exp(-self.expon * average_pairdist)
        disp_contribution = frac_disp * self.disp_coeff / (average_pairdist**12)

        return expon_contribution + disp_contribution


@dataclasses.dataclass(frozen=True)
class GeometricRescalingFunction:
    coeff: float
    expon: float
    disp_coeff: float

    def __post_init__(self) -> None:
        assert self.coeff > 0.0
        assert self.expon > 0.0
        assert self.disp_coeff > 0.0

    def __call__(self, *six_pair_distances: float) -> float:
        assert len(six_pair_distances) == NUMBER_OF_SIDELENGTHS_FOURBODY

        pairdist_product = functools.reduce(operator.mul, six_pair_distances, 1.0)
        geometric_average = pairdist_product ** (1.0 / 6.0)

        expon_contribution = self.coeff * math.exp(-self.expon * geometric_average)
        disp_contribution = self.disp_coeff / (pairdist_product**2)

        return expon_contribution + disp_contribution
