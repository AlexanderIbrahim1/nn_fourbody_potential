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

from __future__ import annotations

from typing import Generic
from typing import Sequence
from typing import TypeVar
from typing import Union

import numpy as np
from nn_fourbody_potential.full_range.constants import SHORT_RANGE_DISTANCE_CUTOFF
from nn_fourbody_potential.full_range.constants import SHORT_RANGE_SCALING_STEP

from nn_fourbody_potential.sidelength_distributions import SixSideLengths
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.full_range.interaction_range import InteractionRange
from nn_fourbody_potential.full_range.interaction_range import classify_interaction_range
from nn_fourbody_potential.full_range.interaction_range import interaction_range_size_allocation
from nn_fourbody_potential.full_range.short_range import prepare_short_range_extrapolation_data
from nn_fourbody_potential.full_range.short_range_extrapolation_types import ExtrapolationDistanceInfo


T = TypeVar('T')
class ReservedVector(Generic[T]):
    def __init__(self, n_elements: Union[int, Sequence[int]], type_t: T) -> None:
        """
        NOTE: I cannot set `dtype=T` for the numpy array because it will throw an error such as
            `TypeError: Cannot interpret '~T' as a data type`
        This is why the `type_t` variable must be redundantly passed into the constructor
        
        """
        if n_elements < 0:
            raise ValueError("Cannot reserve a negative number of elements.")
        
        self._elements = np.empty(n_elements, dtype=type_t)
        self._i_elem = 0
    
    def push_back(self, elem: T) -> None:
        if self._i_elem >= self._elements.size:
            raise IndexError("Cannot push back an element beyond the reserved range.")
        
        self._elements[self._i_elem] = elem
        self._i_elem += 1
    
    @property
    def elements(self) -> np.ndarray[T]:
        return self._elements
        


class ExtrapolatedPotential:
    def __init__(self, neural_network: RegressionMultilayerPerceptron) -> None:
        self._neural_network = neural_network

    def evaluate_batch(self, samples: Sequence[SixSideLengths]) -> Sequence[float]:
        interaction_ranges = [classify_interaction_range(sample) for sample in samples]

        total_size_allocation = sum([interaction_range_size_allocation(ir) for ir in interaction_ranges])
        batch_sidelengths = ReservedVector[np.float32](total_size_allocation, np.float32)
        
        n_short_range = sum([1 for ir in interaction_ranges if ir == InteractionRange.SHORT_RANGE])
        distance_infos = ReservedVector[ExtrapolationDistanceInfo](n_short_range, ExtrapolationDistanceInfo)

        for (sample, interact_range) in zip(samples, interaction_ranges):
            if interact_range == InteractionRange.SHORT_RANGE:
                extrap_sidelengths, extrap_distance_info = prepare_short_range_extrapolation_data(sample, SHORT_RANGE_SCALING_STEP, SHORT_RANGE_DISTANCE_CUTOFF)
                batch_sidelengths.push_back(extrap_sidelengths.lower)
                batch_sidelengths.push_back(extrap_sidelengths.upper)
                distance_infos.push_back(extrap_distance_info)
            elif interact_range == InteractionRange.LONG_RANGE:
                continue
            else:
                batch_sidelengths.push_back(sample)

        # feed the batches into the neural network
        
        # loop over samples and interact_range pairs
        # allocate memory for the final array of floats
        # get the energy depending on short-range, mid-range, long-range
        # need to keep track of the indices here!!!