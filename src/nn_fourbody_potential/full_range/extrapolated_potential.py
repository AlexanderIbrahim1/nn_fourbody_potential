"""
This module contains components for calculating the four-body interaction potential
energy for parahydrogen, incorporating possible short-range or long-range adjustments.
"""

from __future__ import annotations

from typing import Sequence
from typing import Tuple

import numpy as np
import torch

from nn_fourbody_potential.full_range.constants import SHORT_RANGE_DISTANCE_CUTOFF
from nn_fourbody_potential.full_range.constants import SHORT_RANGE_SCALING_STEP
from nn_fourbody_potential.constants import NUMBER_OF_SIDELENGTHS_FOURBODY

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
        device: str,
    ) -> None:
        self._device = device
        self._neural_network = neural_network
        self._transformers = transformers
        self._output_to_energy_rescaler = output_to_energy_rescaler
        self._lr_corrector = LongRangeEnergyCorrector()

    def __call__(self, input_sidelengths: torch.Tensor) -> torch.Tensor:
        if len(input_sidelengths.shape) != 2 or input_sidelengths.shape[1] != NUMBER_OF_SIDELENGTHS_FOURBODY:
            raise RuntimeError("The input sidelengths tensor must be 2D, and the axis=1 dimension must have 6 values.")

        # reason for ignore: function takes any element of six floating-point numbers
        interaction_ranges = [classify_interaction_range(sample.tolist()) for sample in input_sidelengths]  # type: ignore

        batch_sidelengths, distance_infos = self._batch_from_interaction_ranges(input_sidelengths, interaction_ranges)
        batch_energies = self._calculate_batch_energies(batch_sidelengths)

        extrapolated_energies = torch.empty(len(input_sidelengths), dtype=torch.float32)

        def calculate_short_range_energy() -> float:
            dist_info = distance_infos.pop_front()
            lower_energy: torch.Tensor = batch_energies.pop_front()
            upper_energy: torch.Tensor = batch_energies.pop_front()
            extrap_energies = ExtrapolationEnergies(lower_energy.item(), upper_energy.item())
            return short_range_energy_extrapolation(dist_info, extrap_energies)

        def calculate_mid_range_energy() -> float:
            abinitio_energy: torch.Tensor = batch_energies.pop_front()
            return abinitio_energy.item()

        def calculate_long_range_energy(sample: torch.Tensor) -> float:
            return self._lr_corrector.dispersion_from_sidelengths(sample.tolist())  # type: ignore

        def calculate_mixed_range_energy(shortmid_energy: float, sample: torch.Tensor) -> float:
            return self._lr_corrector.mixed_from_sidelengths(shortmid_energy, sample.tolist())  # type: ignore

        for i_extrap, (sample, ir) in enumerate(zip(input_sidelengths, interaction_ranges)):
            if ir == InteractionRange.ABINITIO_SHORT:
                energy = calculate_short_range_energy()
            elif ir == InteractionRange.ABINITIO_MID:
                energy = calculate_mid_range_energy()
            elif ir == InteractionRange.MIXED_SHORT:
                short_energy = calculate_short_range_energy()
                energy = calculate_mixed_range_energy(short_energy, sample)
            elif ir == InteractionRange.MIXED_MID:
                mid_energy = calculate_mid_range_energy()
                energy = calculate_mixed_range_energy(mid_energy, sample)
            elif ir == InteractionRange.LONG:
                energy = calculate_long_range_energy(sample)
            else:
                assert False, "unreachable"

            extrapolated_energies[i_extrap] = energy

        return extrapolated_energies

    def _batch_from_interaction_ranges(
        self, samples: torch.Tensor, interaction_ranges: Sequence[InteractionRange]
    ) -> Tuple[ReservedDeque, ReservedDeque]:
        batch_sidelengths = _preallocate_batch_sidelengths(interaction_ranges)
        distance_infos = _preallocate_distance_infos(interaction_ranges)

        step = SHORT_RANGE_SCALING_STEP
        cutoff = SHORT_RANGE_DISTANCE_CUTOFF

        for sample, ir in zip(samples, interaction_ranges):
            if ir == InteractionRange.ABINITIO_SHORT or ir == InteractionRange.MIXED_SHORT:
                extrap_sidelengths, extrap_dist_info = prepare_short_range_extrapolation_data(sample.tolist(), step, cutoff)  # type: ignore
                batch_sidelengths.push_back(torch.tensor(extrap_sidelengths.lower))
                batch_sidelengths.push_back(torch.tensor(extrap_sidelengths.upper))
                distance_infos.push_back(extrap_dist_info)
            elif ir == InteractionRange.ABINITIO_MID or ir == InteractionRange.MIXED_MID:
                batch_sidelengths.push_back(sample)
            elif ir == InteractionRange.LONG:
                continue
            else:
                assert False, "unreachable"

        return batch_sidelengths, distance_infos

    def _calculate_batch_energies(self, batch_sidelengths: ReservedDeque) -> ReservedDeque:
        if batch_sidelengths.size == 0:
            return ReservedDeque.with_size(torch.tensor([], dtype=torch.float32))

        input_data = transform_sidelengths_data(batch_sidelengths.elements, self._transformers)  # type: ignore
        input_data = torch.from_numpy(input_data.astype(np.float32)).to(self._device)

        with torch.no_grad():
            self._neural_network.eval()
            output_data: torch.Tensor = self._neural_network(input_data)

        for i in range(batch_sidelengths.size):
            output = output_data[i].item()
            sidelengths: torch.Tensor = batch_sidelengths[i]
            output_data[i] = self._output_to_energy_rescaler(output, sidelengths.tolist())  # type: ignore

        return ReservedDeque.with_size(output_data)


def _preallocate_batch_sidelengths(interaction_ranges: Sequence[InteractionRange]) -> ReservedDeque:
    total_size_allocation = sum([interaction_range_size_allocation(ir) for ir in interaction_ranges])
    sidelengths_shape = (total_size_allocation, NUMBER_OF_SIDELENGTHS_FOURBODY)

    return ReservedDeque.with_no_size(torch.empty(sidelengths_shape, dtype=torch.float32))


def _preallocate_distance_infos(interaction_ranges: Sequence[InteractionRange]) -> ReservedDeque:
    def is_short(ir: InteractionRange) -> bool:
        return ir == InteractionRange.MIXED_SHORT or ir == InteractionRange.ABINITIO_SHORT

    n_short_range = sum([1 for ir in interaction_ranges if is_short(ir)])

    return ReservedDeque.with_no_size([None for _ in range(n_short_range)])  # type; ignore
