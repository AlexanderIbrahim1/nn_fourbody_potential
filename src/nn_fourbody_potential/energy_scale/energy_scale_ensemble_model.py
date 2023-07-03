from typing import Sequence

import numpy as np

from nn_fourbody_potential.energy_scale.energy_scale import EnergyScale
from nn_fourbody_potential.energy_scale.energy_scale import EnergyScaleAssigner
from nn_fourbody_potential.energy_scale.energy_scale import EnergyScaleFraction
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.reserved_deque import ReservedDeque
from nn_fourbody_potential.sidelength_distributions import SixSideLengths


# PLAN
# - get energy from the full-range model
# - feed energy into function to determine which region of the energy scale it belongs to
# - assign one of the EnergyScale enums to the sample
# - push the EnergyScale instances into a ReservedVector
# - have separate ReservedVector instances for the MIXED_LOW_MEDIUM and MIXED_MEDIUM_HIGH instances
#   to store the fraction of each section
# - create three ReservedVector instances to store the LOW, MEDIUM, and HIGH samples
# - then calculate the energies for the LOW, MEDIUM, and HIGH samples
# - keep using popleft() on the EnergyScale vector to determine how to popleft() energies and fractions
#   from the other vectors


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
    ) -> None:
        self._check_max_n_samples(max_n_samples)

        self._max_n_samples = max_n_samples
        self._full_energy_model = full_energy_model
        self._low_energy_model = low_energy_model
        self._medium_energy_model = medium_energy_model
        self._high_energy_model = high_energy_model
        self._energy_scale_assigner = energy_scale_assigner

        self._energy_scale_fraction_deque = ReservedDeque[EnergyScaleFraction].new(max_n_samples, EnergyScaleFraction)
        self._low_samples_deque = ReservedDeque[SixSideLengths].new(max_n_samples, SixSideLengths)
        self._medium_samples_deque = ReservedDeque[SixSideLengths].new(max_n_samples, SixSideLengths)
        self._high_samples_deque = ReservedDeque[SixSideLengths].new(max_n_samples, SixSideLengths)

    def evaluate_batch(self, samples: Sequence[SixSideLengths]) -> Sequence[float]:
        n_samples = len(samples)
        self._check_number_of_samples(n_samples)

        self._reset_all_deques()

        coarse_energies = self._full_energy_model(samples)

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

        low_energies = ReservedDeque[float].from_array(self._low_energy_model(self._low_samples_deque))
        medium_energies = ReservedDeque[float].from_array(self._medium_energy_model(self._medium_samples_deque))
        high_energies = ReservedDeque[float].from_array(self._high_energy_model(self._high_samples_deque))

        output_energies = np.empty(n_samples, dtype=np.float32)
        for i, energy_scale_fraction in enumerate(self._energy_scale_fraction_deque):
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

            output_energies[i] = energy

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
