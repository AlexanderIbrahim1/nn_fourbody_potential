import pytest

import numpy as np

from nn_fourbody_potential.full_range.short_range_extrapolation_types import ExtrapolationDistanceInfo
from nn_fourbody_potential.full_range.short_range_extrapolation_types import ExtrapolationEnergies
from nn_fourbody_potential.full_range.short_range_extrapolation_types import LinearEnergyExtrapolator
from nn_fourbody_potential.full_range.short_range_extrapolation_types import ExponentialEnergyExtrapolator


class TestLinearEnergyExtrapolator:
    def test_basic(self):
        energies = ExtrapolationEnergies(1.0, 0.0)
        extrap_distance_info = ExtrapolationDistanceInfo(0.0, 1.0, 2.0)

        expect_slope = -1.0
        expect_energy = 2.0

        linear_extrapolator = LinearEnergyExtrapolator(energies, extrap_distance_info)

        assert linear_extrapolator.slope == pytest.approx(expect_slope)
        assert linear_extrapolator.energy == pytest.approx(expect_energy)


class TestExponentialEnergyExtrapolator:
    def test_basic(self):
        def expon_func(x: float, coeff: float = 1.0, expon: float = 1.0) -> float:
            return coeff * np.exp(-expon * x)

        r_short_range = 1.0
        r_lower = 3.0
        r_upper = 4.0

        energies = ExtrapolationEnergies(expon_func(r_lower), expon_func(r_upper))
        extrap_distance_info = ExtrapolationDistanceInfo(r_short_range, r_lower, r_upper)

        expect_slope = 1.0
        expect_energy = expon_func(r_short_range)

        expon_extrapolator = ExponentialEnergyExtrapolator(energies, extrap_distance_info)

        assert expon_extrapolator.slope == pytest.approx(expect_slope)
        assert expon_extrapolator.energy == pytest.approx(expect_energy)
        assert not expon_extrapolator.is_magnitude_increasing_with_distance
