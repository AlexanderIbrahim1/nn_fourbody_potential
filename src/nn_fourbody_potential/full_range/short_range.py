# PLAN
#
# - function:
#   - take the three samples and the three energies (output from the NN)
#   - extrapolate to a shorter range

import numpy as np

from dataclasses import astuple
from dataclasses import dataclass
from functools import cached_property

from nn_fourbody_potential.sidelength_distributions import SixSideLengths
from nn_fourbody_potential.full_range.constants import SHORT_RANGE_DISTANCE_CUTOFF


@dataclass
class ExtrapolationEnergies:
    energies_lower: float
    energies_upper: float

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class ExtrapolationSideLengths:
    sidelengths_lower: SixSideLengths
    sidelengths_upper: SixSideLengths

    def __iter__(self):
        return iter(astuple(self))


@dataclass
class LinearEnergyExtrapolator:
    extrapolation_energies: ExtrapolationEnergies
    delta_r: float
    r_lower: float
    r_short_range: float

    @cached_property
    def slope(self) -> float:
        energies_lower, energies_upper = self.extrapolation_energies
        return (energies_upper - energies_lower) / self.delta_r

    @cached_property
    def energy(self) -> float:
        energies_lower = self.extrapolation_energies.energies_lower
        return energies_lower + self.slope * (self.r_short_range - self.r_lower)


@dataclass
class ExponentialEnergyExtrapolator:
    extrapolation_energies: ExtrapolationEnergies
    delta_r: float
    r_lower: float
    r_short_range: float

    @cached_property
    def slope(self) -> float:
        energies_lower, energies_upper = self.extrapolation_energies

        abs_energies_floor = 1.0e-6
        abs_energies_lower_floor = max(abs_energies_floor, abs(energies_lower))
        abs_energies_upper_floor = max(abs_energies_floor, abs(energies_upper))

        return -np.log(abs_energies_upper_floor / abs_energies_lower_floor) / self.delta_r

    @cached_property
    def energy(self) -> float:
        energies_lower = self.extrapolation_energies.energies_lower
        return energies_lower * np.exp(-self.slope * (self.r_short_range - self.r_lower))

    @cached_property
    def is_magnitude_increasing_with_distance(self) -> float:
        return self.slope < 0.0


def short_range_energy_extrapolation(
    extrapolation_sidelengths: ExtrapolationSideLengths,
    extrapolation_energies: ExtrapolationEnergies,
    sample: SixSideLengths,
) -> float:
    sidelengths_lower, sidelengths_upper = extrapolation_sidelengths
    energies_lower, energies_upper = extrapolation_energies

    r_lower = min(sidelengths_lower)
    r_upper = min(sidelengths_upper)
    r_short_range = min(sample)
    delta_r = r_upper - r_lower

    slope_min = 6.0
    slope_max = 8.0

    linear_extrapolator = LinearEnergyExtrapolator(extrapolation_energies, delta_r, r_lower, r_short_range)
    expon_extrapolator = ExponentialEnergyExtrapolator(extrapolation_energies, delta_r, r_lower, r_short_range)

    # an exponential decay does not change the sign of the function; we must extrapolate linearly here
    if _is_different_sign(energies_lower, energies_upper):
        return linear_extrapolator.energy

    # if the magnitude increases with distance, then an exponential decay is not an appropriate fit
    if expon_extrapolator.is_magnitude_increasing_with_distance:
        return linear_extrapolator.energy

    if expon_extrapolator.slope <= slope_min:
        return expon_extrapolator.energy
    elif expon_extrapolator.slope >= slope_max:
        return linear_extrapolator.energy
    else:
        weight_linear = _cosine_transition(expon_extrapolator.slope, slope_min, slope_max)
        weight_expon = 1.0 - weight_linear
        energy_linear = linear_extrapolator.energy
        energy_expon = expon_extrapolator.energy

        return weight_linear * energy_linear + weight_expon * energy_expon


def _cosine_transition(x: float, x_min: float, x_max: float) -> float:
    assert x_min < x_max
    if x <= x_min:
        return 0.0
    elif x >= x_max:
        return 1.0
    else:
        k = (x - x_min) / (x_max - x_min)
        return 0.5 * (1.0 - np.cos(np.pi * k))


def _is_different_sign(energies_lower: float, energies_upper: float) -> bool:
    return energies_lower * energies_upper <= 0.0


def create_extrapolation_samples(
    short_range_sample: SixSideLengths,
    scaling_step: float = 0.05,
    *,
    short_range_cutoff: float = SHORT_RANGE_DISTANCE_CUTOFF,
) -> ExtrapolationSideLengths:
    shortest_side_length = min(short_range_sample)

    _check_positive_scaling_step(scaling_step)
    _check_shortest_side_length_is_short_enough(shortest_side_length, short_range_cutoff)
    _check_shortest_side_length_is_positive(shortest_side_length)

    scaling_ratio_lower = (short_range_cutoff + 0 * scaling_step) / shortest_side_length
    scaling_ratio_upper = (short_range_cutoff + 1 * scaling_step) / shortest_side_length

    sample_lower = tuple([scaling_ratio_lower * sidelen for sidelen in short_range_sample])
    sample_upper = tuple([scaling_ratio_upper * sidelen for sidelen in short_range_sample])

    return ExtrapolationSideLengths(sample_lower, sample_upper)


def _check_positive_scaling_step(scaling_step: float) -> None:
    if scaling_step <= 0.0:
        raise ValueError(
            "The scaling step for the short-range extrapolation must be positive." f"Found: {scaling_step}"
        )


def _check_shortest_side_length_is_short_enough(shortest_side_length: float, short_range_cutoff: float) -> None:
    if shortest_side_length >= short_range_cutoff:
        raise ValueError(
            "Short-range extrapolation is only available for samples with at least one side length\n"
            f"less than {short_range_cutoff}.\n"
            f"Found: {shortest_side_length}"
        )


def _check_shortest_side_length_is_positive(shortest_side_length: float) -> None:
    if shortest_side_length <= 0.0:
        raise ValueError(
            "Short-range extrapolation is only available for samples with all positive side lengths.\n"
            f"Found: ({shortest_side_length})"
        )
