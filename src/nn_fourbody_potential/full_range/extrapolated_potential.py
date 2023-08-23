"""
This module contains components for calculating the four-body interaction potential
energy for parahydrogen, incorporating possible short-range or long-range adjustments.
"""

# TODO:
# - repo appears to work BUT:
#   - still need to test it (all orders of inputs, corner cases, etc.)

from __future__ import annotations

from typing import Callable
from typing import Sequence
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import torch

from nn_fourbody_potential.full_range.constants import SHORT_RANGE_DISTANCE_CUTOFF
from nn_fourbody_potential.full_range.constants import SHORT_RANGE_SCALING_STEP

from nn_fourbody_potential.full_range.interaction_range import InteractionRange
from nn_fourbody_potential.full_range.interaction_range import classify_interaction_range
from nn_fourbody_potential.full_range.interaction_range import interaction_range_size_allocation
from nn_fourbody_potential.full_range.long_range import LongRangeEnergyCorrector
from nn_fourbody_potential.full_range.short_range import prepare_short_range_extrapolation_data
from nn_fourbody_potential.full_range.short_range import short_range_energy_extrapolation
from nn_fourbody_potential.full_range.short_range_extrapolation_types import ExtrapolationDistanceInfo
from nn_fourbody_potential.full_range.short_range_extrapolation_types import ExtrapolationEnergies
from nn_fourbody_potential.reserved_deque import ReservedDeque
from nn_fourbody_potential.sidelength_distributions import SixSideLengths
from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import transform_sidelengths_data


class ExtrapolatedPotential:
    def __init__(
        self,
        neural_network: Callable[[torch.Tensor], torch.Tensor],
        transformers: Sequence[SixSideLengthsTransformer],
    ) -> None:
        self._neural_network = neural_network
        self._transformers = transformers
        self._long_range_corrector = LongRangeEnergyCorrector()

        self._neural_network.eval()

    def evaluate_batch(self, samples: Sequence[SixSideLengths]) -> NDArray:
        interaction_ranges = [classify_interaction_range(sample) for sample in samples]

        batch_sidelengths, distance_infos = self._batch_from_interaction_ranges(samples, interaction_ranges)
        batch_energies = self._calculate_batch_energies(batch_sidelengths)

        extrapolated_energies = np.empty(len(samples), dtype=np.float32)

        for i_extrap, (sample, interact_range) in enumerate(zip(samples, interaction_ranges)):
            if interact_range == InteractionRange.SHORT_RANGE:
                dist_info = distance_infos.pop_front()
                lower_energy = batch_energies.pop_front()
                upper_energy = batch_energies.pop_front()
                extrap_energies = ExtrapolationEnergies(lower_energy, upper_energy)
                extrapolated_energies[i_extrap] = short_range_energy_extrapolation(dist_info, extrap_energies)
            elif interact_range == InteractionRange.MID_RANGE:
                extrapolated_energies[i_extrap] = batch_energies.pop_front()
            elif interact_range == InteractionRange.MIXED_MID_LONG_RANGE:
                abinitio_energy = batch_energies.pop_front()
                extrapolated_energies[i_extrap] = self._long_range_corrector.mixed_from_sidelengths(
                    abinitio_energy, sample
                )
            elif interact_range == InteractionRange.LONG_RANGE:
                extrapolated_energies[i_extrap] = self._long_range_corrector.dispersion_from_sidelengths(sample)
            else:
                assert False, "unreachable"

        return extrapolated_energies

    def _preallocate_batch_sidelengths(
        self, interaction_ranges: Sequence[InteractionRange]
    ) -> ReservedDeque[np.float32]:
        total_size_allocation = sum([interaction_range_size_allocation(ir) for ir in interaction_ranges])
        n_sidelengths = 6
        sidelengths_shape = (total_size_allocation, n_sidelengths)

        return ReservedDeque[np.float32].new(sidelengths_shape, np.float32)

    def _preallocate_distance_infos(
        self, interaction_ranges: Sequence[InteractionRange]
    ) -> ReservedDeque[ExtrapolationDistanceInfo]:
        n_short_range = sum([1 for ir in interaction_ranges if ir == InteractionRange.SHORT_RANGE])

        return ReservedDeque[ExtrapolationDistanceInfo].new(n_short_range, ExtrapolationDistanceInfo)

    def _batch_from_interaction_ranges(
        self, samples: Sequence[SixSideLengths], interaction_ranges: Sequence[InteractionRange]
    ) -> Tuple[ReservedDeque[np.float32], ReservedDeque[ExtrapolationDistanceInfo]]:
        batch_sidelengths = self._preallocate_batch_sidelengths(interaction_ranges)
        distance_infos = self._preallocate_distance_infos(interaction_ranges)

        for sample, interact_range in zip(samples, interaction_ranges):
            if interact_range == InteractionRange.SHORT_RANGE:
                extrap_sidelengths, extrap_distance_info = prepare_short_range_extrapolation_data(
                    sample, SHORT_RANGE_SCALING_STEP, SHORT_RANGE_DISTANCE_CUTOFF
                )
                batch_sidelengths.push_back(extrap_sidelengths.lower)
                batch_sidelengths.push_back(extrap_sidelengths.upper)
                distance_infos.push_back(extrap_distance_info)
            elif interact_range == InteractionRange.MID_RANGE:
                batch_sidelengths.push_back(sample)
            elif interact_range == InteractionRange.MIXED_MID_LONG_RANGE:
                batch_sidelengths.push_back(sample)
            elif interact_range == InteractionRange.LONG_RANGE:
                continue
            else:
                assert False, "unreachable"

        return batch_sidelengths, distance_infos

    def _calculate_batch_energies(self, batch_sidelengths: ReservedDeque[np.float32]) -> ReservedDeque[np.float32]:
        if batch_sidelengths.size != 0:
            input_data = transform_sidelengths_data(batch_sidelengths.elements, self._transformers)
            input_data = torch.from_numpy(
                input_data.astype(np.float32)
            )  # NOTE: I think it's already of type np.float32?

            with torch.no_grad():
                output_data: torch.Tensor = self._neural_network(input_data)
                output_energies = output_data.detach().cpu().numpy()  # NOTE: is .cpu() making too many assumptions?
        else:
            output_energies = np.array([])

        return ReservedDeque[np.float32].from_array(output_energies)
