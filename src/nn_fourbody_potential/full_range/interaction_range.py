f"""
The InteractionRange enum and its related functions are used to classify each four-body
geometry as being short-range, mid-range, or long-range. These classifications are used
to determine how a sample is treated.
"""

import enum

from nn_fourbody_potential.sidelength_distributions import SixSideLengths
from nn_fourbody_potential.models import RegressionMultilayerPerceptron


class InteractionRange(enum.Enum):
    SHORT_RANGE = enum.auto()
    MID_RANGE = enum.auto()
    LONG_RANGE = enum.auto()


def interaction_range_size_allocation(interact_range: InteractionRange) -> int:
    """To apply the corrections for different intermolecular separations, the samples that are
    provided aren't always the samples that are fed into the neural network. This function finds
    out how many samples must be fed into the neural network to calculate a certain energy."""
    if interact_range == InteractionRange.SHORT_RANGE:
        return 3
    elif interact_range == InteractionRange.LONG_RANGE:
        return 0
    else:
        return 1


def classify_interaction_range(
    sample: SixSideLengths,
) -> InteractionRange:
    """Determine which of the three ranges to classify the sample into."""
    if _is_short_range_sample(sample):
        return InteractionRange.SHORT_RANGE
    elif _is_long_range_sample(sample):
        return InteractionRange.LONG_RANGE
    else:
        return InteractionRange.MID_RANGE


def _is_short_range_sample(sample: SixSideLengths) -> bool:
    short_range_distance_cutoff = 2.2
    return any([s < short_range_distance_cutoff for s in sample])


def _is_long_range_sample(sample: SixSideLengths) -> bool:
    n_sidelengths = len(sample)
    long_range_sum_of_sidelengths_cutoff = 3.5 * n_sidelengths

    return sum(sample) > long_range_sum_of_sidelengths_cutoff