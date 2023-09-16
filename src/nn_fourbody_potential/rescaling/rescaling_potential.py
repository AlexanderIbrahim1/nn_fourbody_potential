"""
This module contains the RescalingPotential class, which is an analytic function with
components similar to those of the true four-body potential, but with exaggerated
features.

Its purpose is to help rescale the true ab initio energies into a smaller range of values,
which should improve the effectiveness of training.
"""

from __future__ import annotations

import dataclasses
import math
import statistics


@dataclasses.dataclass(frozen=True)
class RescalingPotential:
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
