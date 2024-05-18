"""
The InteractionRange enum and its related functions are used to classify each four-body
geometry as being short-range, mid-range, or long-range. These classifications are used
to determine how a sample is treated.
"""

import enum
import statistics

from nn_fourbody_potential.common_types import SixSideLengths
from nn_fourbody_potential.full_range.constants import SHORT_RANGE_DISTANCE_CUTOFF
from nn_fourbody_potential.full_range.constants import LOWER_MIXED_DISTANCE
from nn_fourbody_potential.full_range.constants import UPPER_MIXED_DISTANCE


class InteractionRange(enum.Enum):
    SHORTMID = enum.auto()
    MIXED = enum.auto()
    LONG = enum.auto()


class ShortMidInteractionRange(enum.Enum):
    SHORT_RANGE = enum.auto()
    MID_RANGE = enum.auto()


def classify_interaction_range(sample: SixSideLengths) -> InteractionRange:
    """Determine which of the three ranges to classify the sample into."""

    average_sidelength = statistics.mean(sample)

    if average_sidelength <= LOWER_MIXED_DISTANCE:
        return InteractionRange.SHORTMID
    elif average_sidelength >= UPPER_MIXED_DISTANCE:
        return InteractionRange.LONG
    else:
        return InteractionRange.MIXED


def classify_shortmid_interaction_range(sample: SixSideLengths) -> ShortMidInteractionRange:
    if any([s < SHORT_RANGE_DISTANCE_CUTOFF for s in sample]):
        return ShortMidInteractionRange.SHORT_RANGE
    else:
        return ShortMidInteractionRange.MID_RANGE
