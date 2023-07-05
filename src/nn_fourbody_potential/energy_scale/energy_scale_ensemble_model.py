from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import torch

from nn_fourbody_potential.energy_scale.energy_scale import EnergyScale
from nn_fourbody_potential.energy_scale.energy_scale import EnergyScaleAssigner
from nn_fourbody_potential.energy_scale.energy_scale import EnergyScaleFraction
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.reserved_deque import ReservedDeque
from nn_fourbody_potential.sidelength_distributions import SixSideLengths
from nn_fourbody_potential.transformations import SixSideLengthsTransformer

from nn_fourbody_potential.transformations.applications import transform_sidelengths_data


class EnergyScaleEnsembleModel:
    """
    The `EnergyScaleEnsembleModel` is responsible for determining which energy scale (low, medium, high) a
    sample belongs to with the coarse, full-scale model, and handling the details of passing the samples into
    the appropriate energy-scale model.

    To make it easier to manage the samples, and to minimize the amount of memory allocations, we use the
    `ReservedDeque` class to create a preallocated, contiguous buffer. Due to this, the maximum number of
    samples must be known ahead of time.
    """

    def __init__(
        self,
        max_n_samples: int,
        full_energy_model: RegressionMultilayerPerceptron,
        low_energy_model: RegressionMultilayerPerceptron,
        medium_energy_model: RegressionMultilayerPerceptron,
        high_energy_model: RegressionMultilayerPerceptron,
        energy_scale_assigner: EnergyScaleAssigner,
        transformers: Sequence[SixSideLengthsTransformer],
    ) -> None:
        self._check_max_n_samples(max_n_samples)

        self._max_n_samples = max_n_samples
        self._full_energy_model = full_energy_model
        self._low_energy_model = low_energy_model
        self._medium_energy_model = medium_energy_model
        self._high_energy_model = high_energy_model
        self._energy_scale_assigner = energy_scale_assigner
        self._transformers = transformers

        self._energy_scale_fraction_deque = ReservedDeque[EnergyScaleFraction].new(max_n_samples, EnergyScaleFraction)
        self._low_samples_deque = ReservedDeque[SixSideLengths].new(max_n_samples, tuple)
        self._medium_samples_deque = ReservedDeque[SixSideLengths].new(max_n_samples, tuple)
        self._high_samples_deque = ReservedDeque[SixSideLengths].new(max_n_samples, tuple)

    def _evaluate_energies(
        self, raw_samples: NDArray, model: RegressionMultilayerPerceptron
    ) -> ReservedDeque[np.float32]:
        samples: NDArray = transform_sidelengths_data(raw_samples, self._transformers)
        samples = samples.astype(np.float32)
        with torch.no_grad():
            samples = torch.from_numpy(samples)
            energies: torch.Tensor = model(samples)
            energies = energies.detach().cpu().numpy()

        return ReservedDeque[np.float32].from_array(energies)

    def evaluate_batch(self, samples: Sequence[SixSideLengths]) -> NDArray:
        n_samples = len(samples)
        self._check_number_of_samples(n_samples)

        if n_samples == 0:
            return np.array([])

        self._reset_all_deques()

        samples = np.array([samples]).reshape(-1, 6)
        coarse_energies = self._evaluate_energies(samples, self._full_energy_model)

        for sample, energy in zip(samples, coarse_energies):
            energy_scale_fraction = self._energy_scale_assigner.assign_energy_scale(energy)
            self._energy_scale_fraction_deque.push_back(energy_scale_fraction)

            if energy_scale_fraction.scale == EnergyScale.LOW:
                self._low_samples_deque.push_back(sample)
            elif energy_scale_fraction.scale == EnergyScale.MIXED_LOW_MEDIUM:
                self._low_samples_deque.push_back(sample)
                self._medium_samples_deque.push_back(sample)
            elif energy_scale_fraction.scale == EnergyScale.MEDIUM:
                self._medium_samples_deque.push_back(sample)
            elif energy_scale_fraction.scale == EnergyScale.MIXED_MEDIUM_HIGH:
                self._medium_samples_deque.push_back(sample)
                self._high_samples_deque.push_back(sample)
            else:
                self._high_samples_deque.push_back(sample)

        low_samples = np.array([sample for sample in self._low_samples_deque])
        low_energies = self._evaluate_energies(low_samples, self._low_energy_model)

        medium_samples = np.array([sample for sample in self._medium_samples_deque])
        medium_energies = self._evaluate_energies(medium_samples, self._medium_energy_model)

        high_samples = np.array([sample for sample in self._high_samples_deque])
        high_energies = self._evaluate_energies(high_samples, self._high_energy_model)

        output_energies = np.empty(n_samples, dtype=np.float32)
        for i_sample in range(n_samples):
            energy_scale_fraction = self._energy_scale_fraction_deque[i_sample]
            if energy_scale_fraction.scale == EnergyScale.LOW:
                energy = low_energies.pop_front()
            elif energy_scale_fraction.scale == EnergyScale.MIXED_LOW_MEDIUM:
                low_energy = low_energies.pop_front()
                medium_energy = medium_energies.pop_front()
                lower_fraction = energy_scale_fraction.lower_fraction
                upper_fraction = energy_scale_fraction.upper_fraction
                energy = lower_fraction * low_energy + upper_fraction * medium_energy
            elif energy_scale_fraction.scale == EnergyScale.MEDIUM:
                energy = medium_energies.pop_front()
            elif energy_scale_fraction.scale == EnergyScale.MIXED_MEDIUM_HIGH:
                medium_energy = medium_energies.pop_front()
                high_energy = high_energies.pop_front()
                lower_fraction = energy_scale_fraction.lower_fraction
                upper_fraction = energy_scale_fraction.upper_fraction
                energy = lower_fraction * medium_energy + upper_fraction * high_energy
            else:
                energy = high_energies.pop_front()

            output_energies[i_sample] = energy

        return output_energies

    def _reset_all_deques(self) -> None:
        self._energy_scale_fraction_deque.reset()
        self._low_samples_deque.reset()
        self._medium_samples_deque.reset()
        self._high_samples_deque.reset()

    def _check_max_n_samples(self, max_n_samples: int) -> None:
        if max_n_samples <= 0:
            raise ValueError("The buffer size for the calculations must be a positive number.")

    def _check_number_of_samples(self, n_samples: int) -> None:
        if n_samples > self._max_n_samples:
            raise ValueError(
                "Attempt to evaluate too many samples.\n"
                f"This is a buffered operation, and the maximum number of samples allowed at one time is {self._max_n_samples}\n"
                f"Found: {n_samples}"
            )
