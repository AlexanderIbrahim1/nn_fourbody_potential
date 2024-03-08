"""
This module contains components for calculating the four-body interaction potential
energy for parahydrogen, incorporating possible short-range or long-range adjustments.
"""

# TODO:
# - repo appears to work BUT:
#   - still need to test it (all orders of inputs, corner cases, etc.)

from __future__ import annotations

from typing import Sequence
from typing import Tuple

import numpy as np
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
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.reserved_deque import ReservedDeque
from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import transform_sidelengths_data
from nn_fourbody_potential import rescaling


ExtrapDistInfo = ExtrapolationDistanceInfo


class ExtrapolatedPotential:
    def __init__(
        self,
        neural_network: RegressionMultilayerPerceptron,
        transformers: Sequence[SixSideLengthsTransformer],
        output_to_energy_rescaler: rescaling.ReverseEnergyRescaler,
    ) -> None:
        self._neural_network = neural_network
        self._transformers = transformers
        self._output_to_energy_rescaler = output_to_energy_rescaler
        self._lr_corrector = LongRangeEnergyCorrector()

    def __call__(self, input_sidelengths: torch.Tensor) -> torch.Tensor:
        # TODO: perform a check on the shape of the input

        # reason for ignore: function takes any element of six floating-point numbers
        interaction_ranges = [classify_interaction_range(sample.tolist()) for sample in input_sidelengths]  # type: ignore

        batch_sidelengths, distance_infos = self._batch_from_interaction_ranges(input_sidelengths, interaction_ranges)
        batch_energies = self._calculate_batch_energies(batch_sidelengths)

        extrapolated_energies = torch.empty(len(input_sidelengths), dtype=torch.float32)

        for i_extrap, (sample, interact_range) in enumerate(zip(input_sidelengths, interaction_ranges)):
            if interact_range == InteractionRange.SHORT_RANGE:
                dist_info = distance_infos.pop_front()
                lower_energy: torch.Tensor = batch_energies.pop_front()
                upper_energy: torch.Tensor = batch_energies.pop_front()
                extrap_energies = ExtrapolationEnergies(lower_energy.item(), upper_energy.item())
                extrapolated_energies[i_extrap] = short_range_energy_extrapolation(dist_info, extrap_energies)
            elif interact_range == InteractionRange.MID_RANGE:
                abinitio_energy: torch.Tensor = batch_energies.pop_front()
                extrapolated_energies[i_extrap] = abinitio_energy.item()
            elif interact_range == InteractionRange.MIXED_MID_LONG_RANGE:
                abinitio_energy: torch.Tensor = batch_energies.pop_front()
                sidelengths = tuple([sl.item() for sl in sample])
                energy = self._lr_corrector.mixed_from_sidelengths(abinitio_energy.item(), sidelengths)
                extrapolated_energies[i_extrap] = energy
            elif interact_range == InteractionRange.LONG_RANGE:
                extrapolated_energies[i_extrap] = self._lr_corrector.dispersion_from_sidelengths(sample)
            else:
                assert False, "unreachable"

        return extrapolated_energies

    def _batch_from_interaction_ranges(
        self, samples: torch.Tensor, interaction_ranges: Sequence[InteractionRange]
    ) -> Tuple[ReservedDeque, ReservedDeque]:
        batch_sidelengths = _preallocate_batch_sidelengths(interaction_ranges)
        distance_infos = _preallocate_distance_infos(interaction_ranges)

        step = SHORT_RANGE_SCALING_STEP
        cutoff = SHORT_RANGE_DISTANCE_CUTOFF

        for sample, interact_range in zip(samples, interaction_ranges):
            if interact_range == InteractionRange.SHORT_RANGE:
                extrap_sidelengths, extrap_dist_info = prepare_short_range_extrapolation_data(sample, step, cutoff)
                batch_sidelengths.push_back(torch.tensor(extrap_sidelengths.lower))
                batch_sidelengths.push_back(torch.tensor(extrap_sidelengths.upper))
                distance_infos.push_back(extrap_dist_info)
            elif interact_range == InteractionRange.MID_RANGE:
                batch_sidelengths.push_back(sample)
            elif interact_range == InteractionRange.MIXED_MID_LONG_RANGE:
                batch_sidelengths.push_back(sample)
            elif interact_range == InteractionRange.LONG_RANGE:
                continue
            else:
                assert False, "unreachable"

        return batch_sidelengths, distance_infos

    def _calculate_batch_energies(self, batch_sidelengths: ReservedDeque) -> ReservedDeque:
        if batch_sidelengths.size == 0:
            return ReservedDeque.with_size(torch.tensor([], dtype=torch.float32))

        input_data = transform_sidelengths_data(batch_sidelengths.elements, self._transformers)
        input_data = torch.from_numpy(input_data.astype(np.float32))

        with torch.no_grad():
            self._neural_network.eval()
            output_data: torch.Tensor = self._neural_network(input_data)

        for i in range(batch_sidelengths.size):
            output = output_data[i].item()
            sidelengths: torch.Tensor = batch_sidelengths[i]
            output_data[i] = self._output_to_energy_rescaler(output, sidelengths.tolist())

        return ReservedDeque.with_size(output_data)


def _preallocate_batch_sidelengths(interaction_ranges: Sequence[InteractionRange]) -> ReservedDeque:
    total_size_allocation = sum([interaction_range_size_allocation(ir) for ir in interaction_ranges])
    n_sidelengths = 6
    sidelengths_shape = (total_size_allocation, n_sidelengths)

    return ReservedDeque.with_no_size(torch.empty(sidelengths_shape, dtype=torch.float32))


def _preallocate_distance_infos(interaction_ranges: Sequence[InteractionRange]) -> ReservedDeque:
    n_short_range = sum([1 for ir in interaction_ranges if ir == InteractionRange.SHORT_RANGE])

    return ReservedDeque.with_no_size(np.empty(n_short_range, dtype=ExtrapDistInfo))
