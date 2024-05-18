"""
The InteractionRange enum and its related functions are used to classify each four-body
geometry as being short-range, mid-range, or long-range. These classifications are used
to determine how a sample is treated.
"""

import enum
import statistics

from nn_fourbody_potential.common_types import SixSideLengths
from nn_fourbody_potential.full_range.constants import SHORT_RANGE_DISTANCE_CUTOFF
from nn_fourbody_potential.full_range.constants import START_LONG_RANGE_CUTOFF
from nn_fourbody_potential.full_range.constants import END_LONG_RANGE_CUTOFF


class InteractionRange(enum.Enum):
    SHORT_RANGE = enum.auto()
    MID_RANGE = enum.auto()
    MIXED_MID_LONG_RANGE = enum.auto()
    LONG_RANGE = enum.auto()


def interaction_range_size_allocation(interact_range: InteractionRange) -> int:
    """To apply the corrections for different intermolecular separations, the samples that are
    provided aren't always the samples that are fed into the neural network. This function finds
    out how many samples must be fed into the neural network to calculate a certain energy."""
    if interact_range == InteractionRange.SHORT_RANGE:
        return 2
    elif interact_range == InteractionRange.LONG_RANGE:
        return 0
    elif interact_range == InteractionRange.MIXED_MID_LONG_RANGE:
        return 1
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
    elif _is_mixed_mid_long_range_sample(sample):
        return InteractionRange.MIXED_MID_LONG_RANGE
    else:
        return InteractionRange.MID_RANGE


def _is_short_range_sample(sample: SixSideLengths) -> bool:
    return any([s < SHORT_RANGE_DISTANCE_CUTOFF for s in sample])


def _is_mixed_mid_long_range_sample(sample: SixSideLengths) -> bool:
    average_sidelength = statistics.mean(sample)
    return START_LONG_RANGE_CUTOFF < average_sidelength < END_LONG_RANGE_CUTOFF


def _is_long_range_sample(sample: SixSideLengths) -> bool:
    average_sidelength = statistics.mean(sample)
    return average_sidelength > END_LONG_RANGE_CUTOFF
