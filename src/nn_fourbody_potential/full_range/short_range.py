import numpy as np

from typing import Tuple

from nn_fourbody_potential.full_range.short_range_extrapolation_types import ExtrapolationEnergies
from nn_fourbody_potential.full_range.short_range_extrapolation_types import ExtrapolationSideLengths
from nn_fourbody_potential.full_range.short_range_extrapolation_types import ExtrapolationDistanceInfo
from nn_fourbody_potential.full_range.short_range_extrapolation_types import LinearEnergyExtrapolator
from nn_fourbody_potential.full_range.short_range_extrapolation_types import ExponentialEnergyExtrapolator
from nn_fourbody_potential.sidelength_distributions import SixSideLengths


def short_range_energy_extrapolation(
    extrap_distance_info: ExtrapolationDistanceInfo,
    energies: ExtrapolationEnergies,
) -> float:
    slope_min = 6.0
    slope_max = 8.0

    linear_extrapolator = LinearEnergyExtrapolator(energies, extrap_distance_info)
    expon_extrapolator = ExponentialEnergyExtrapolator(energies, extrap_distance_info)

    # an exponential decay does not change the sign of the function; we must extrapolate linearly here
    if _is_different_sign(energies.lower, energies.upper):
        return linear_extrapolator.energy

    # if the magnitude increases with distance, then an exponential decay is not an appropriate fit
    if expon_extrapolator.is_magnitude_increasing_with_distance:
        return linear_extrapolator.energy

    if expon_extrapolator.slope <= slope_min:
        return expon_extrapolator.energy
    elif expon_extrapolator.slope >= slope_max:
        return linear_extrapolator.energy
    else:
        weight_linear = _smooth_01_transition(expon_extrapolator.slope, slope_min, slope_max)
        weight_expon = 1.0 - weight_linear
        energy_linear = linear_extrapolator.energy
        energy_expon = expon_extrapolator.energy

        return weight_linear * energy_linear + weight_expon * energy_expon


def _smooth_01_transition(x: float, x_min: float, x_max: float) -> float:
    """Smoothly transition from 0.0 at or before x <= x_min, to 1.0 at or after x >= x_max."""
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


def prepare_short_range_extrapolation_data(
    short_range_sample: SixSideLengths,
    scaling_step: float,
    short_range_cutoff: float,
) -> Tuple[ExtrapolationSideLengths, ExtrapolationDistanceInfo]:
    sidelength_shortest = min(short_range_sample)

    _check_positive_scaling_step(scaling_step)
    _check_shortest_side_length_is_short_enough(sidelength_shortest, short_range_cutoff)
    _check_shortest_side_length_is_positive(sidelength_shortest)

    sidelength_lower = short_range_cutoff
    sidelength_upper = short_range_cutoff + scaling_step
    extrap_distance_info = ExtrapolationDistanceInfo(sidelength_shortest, sidelength_lower, sidelength_upper)

    scaling_ratio_lower = sidelength_lower / sidelength_shortest
    scaling_ratio_upper = sidelength_upper / sidelength_shortest

    sample_lower = tuple([scaling_ratio_lower * sidelen for sidelen in short_range_sample])
    sample_upper = tuple([scaling_ratio_upper * sidelen for sidelen in short_range_sample])

    extrap_sample = ExtrapolationSideLengths(sample_lower, sample_upper)

    return extrap_sample, extrap_distance_info


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
