"""
The InteractionRange enum and its related functions are used to classify each four-body
geometry as being short-range, mid-range, or long-range. These classifications are used
to determine how a sample is treated.
"""

import enum
import statistics

from nn_fourbody_potential.common_types import SixSideLengths
from nn_fourbody_potential.full_range.constants import LOWER_SHORT_DISTANCE
from nn_fourbody_potential.full_range.constants import UPPER_SHORT_DISTANCE
from nn_fourbody_potential.full_range.constants import LOWER_MIXED_DISTANCE
from nn_fourbody_potential.full_range.constants import UPPER_MIXED_DISTANCE


class InteractionRange(enum.Enum):
    ABINITIO_SHORT = enum.auto()
    ABINITIO_SHORTMID = enum.auto()
    ABINITIO_MID = enum.auto()
    MIXED_SHORT = enum.auto()
    MIXED_SHORTMID = enum.auto()
    MIXED_MID = enum.auto()
    LONG = enum.auto()


def classify_interaction_range(sample: SixSideLengths) -> InteractionRange:
    """Determine which of the three ranges to classify the sample into."""

    average_sidelength: float = statistics.mean(sample)

    if average_sidelength > UPPER_MIXED_DISTANCE:
        return InteractionRange.LONG

    is_abinitio: bool = average_sidelength < LOWER_MIXED_DISTANCE
    is_short: bool = any([s < LOWER_SHORT_DISTANCE for s in sample])
    is_shortmid: bool = any([LOWER_SHORT_DISTANCE <= s < UPPER_SHORT_DISTANCE for s in sample])

    if is_abinitio:
        if is_short:
            return InteractionRange.ABINITIO_SHORT
        elif is_shortmid:
            return InteractionRange.ABINITIO_SHORTMID
        else:
            return InteractionRange.ABINITIO_MID
    else:
        if is_short:
            return InteractionRange.MIXED_SHORT
        elif is_shortmid:
            return InteractionRange.MIXED_SHORTMID
        else:
            return InteractionRange.MIXED_MID


def interaction_range_size_allocation(ir: InteractionRange) -> int:
    if ir == InteractionRange.LONG:
        return 0
    elif ir == InteractionRange.ABINITIO_MID or ir == InteractionRange.MIXED_MID:
        return 1
    elif ir == InteractionRange.ABINITIO_SHORT or ir == InteractionRange.MIXED_SHORT:
        return 2
    else:
        return 3


def is_partly_short(ir: InteractionRange) -> bool:
    IR = InteractionRange
    return ir in [IR.ABINITIO_SHORT, IR.ABINITIO_SHORTMID, IR.MIXED_SHORT, IR.MIXED_SHORTMID]
