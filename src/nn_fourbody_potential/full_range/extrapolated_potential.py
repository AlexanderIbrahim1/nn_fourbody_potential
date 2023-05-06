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

from typing import Sequence

import numpy as np

from nn_fourbody_potential.sidelength_distributions import SixSideLengths
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.full_range.interaction_range import InteractionRange
from nn_fourbody_potential.full_range.interaction_range import classify_interaction_range
from nn_fourbody_potential.full_range.interaction_range import interaction_range_size_allocation


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
