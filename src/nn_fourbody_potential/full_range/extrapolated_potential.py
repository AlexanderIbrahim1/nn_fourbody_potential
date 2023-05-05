"""
This module contains components for calculating the four-body interaction potential
energy for parahydrogen, incorporating possible short-range or long-range adjustments.
"""

# PLAN
# - accept the data in the same format that the NN PES does; as batches of six side lengths
#   - maybe later accept other formats (a list of collections of four points?)
#   - but leave this for later
# - for each, sample, determine ahead of time if it is short-range, mid-range, or long-range
#
# short-range
# - create the additional samples to be fed into the NNPES (so that the energies can be extrapolated)
# - come up with a way to keep track of them
# - when the energies come out of the NN, use them to extrapolate to short ranges
#
# mid-range
# - just feed them into the NN PES directly
#
# long-range
# - use the attenuation function to determine ahead of time if the energy will be a mix of mid-range and
#   long-range interaction energies, or if it will be entirely long-range
# - if entirely long-range, then don't feed the sample into the NN
#   - just use the disperion potential

# TODO:
# - put the `short_range_distance_cutoff` and the `long_range_sum_of_sidelength_cutoff` somewhere as global constants?

import enum
from typing import Sequence

import numpy as np

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


class ExtrapolatedPotential:
    def __init__(self, neural_network: RegressionMultilayerPerceptron) -> None:
        self._neural_network = neural_network

    def evaluate_batch(self, samples: Sequence[SixSideLengths]) -> Sequence[float]:
        interaction_ranges = [classify_interaction_range(sample) for sample in samples]

        total_size_allocation = sum(
            [interaction_range_size_allocation(interact_range) for interact_range in interaction_ranges]
        )

        batch_index_counter = 0

        batch_sidelengths = np.empty((total_size_allocation, 6), dtype=np.float32)
        for (sample, interact_range) in zip(samples, interaction_ranges):
            if interact_range == InteractionRange.SHORT_RANGE:
                # feed in the 3 modified samples
                batch_index_counter += 3
                pass
            elif interact_range == InteractionRange.LONG_RANGE:
                continue
            else:
                # feed in the 1 sample
                batch_index_counter += 1
                pass


def _is_short_range_sample(sample: SixSideLengths) -> bool:
    short_range_distance_cutoff = 2.2
    return any([s < short_range_distance_cutoff for s in sample])


def _is_long_range_sample(sample: SixSideLengths) -> bool:
    n_sidelengths = len(sample)
    long_range_sum_of_sidelengths_cutoff = 3.5 * n_sidelengths

    return sum(sample) > long_range_sum_of_sidelengths_cutoff
