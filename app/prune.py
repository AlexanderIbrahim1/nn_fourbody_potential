"""
This module contains functions for pruning data points from training, testing, and
validation data sets.
"""

import functools
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Sequence

from nn_fourbody_potential.dataio import load_fourbody_training_data
from nn_fourbody_potential.sidelength_distributions.sidelength_types import SixSideLengths


@dataclass
class DataSample:
    sidelengths: SixSideLengths
    energy: float


def prune_data(input_data: Sequence[DataSample], predicate: Callable[[DataSample], bool]) -> Sequence[DataSample]:
    """
    Args:
        input_data (Sequence[DataSample]): the pairs of six side lengths and interaction energies that
            are used as samples for the machine learning model
        predicate (Callable[[DataSample], bool]): a filter function that determines if a certain
            DataSample should be kept; True indicates keeping the sample

    Returns:
        Sequence[DataSample]:
    """


def get_abinitio_hcp_data_filepath() -> Path:
    return Path(".", "data", "abinitio_hcp_data_3901_2.2_4.5.dat")


def main() -> None:
    input_data_filepath = get_abinitio_hcp_data_filepath()
    input_sidelengths, input_energies = load_fourbody_training_data(input_data_filepath)

    input_data = [DataSample(sidelengths, energy) for (sidelengths, energy) in zip(input_sidelengths, input_energies)]

    def sample_filter(sample: DataSample, cutoff_energy: float, cutoff_average_sidelength: float) -> bool:
        assert cutoff_energy > 0.0

        average_sidelength = statistics.mean(sample.sidelengths)

        sidelengths_short_enough = average_sidelength <= cutoff_average_sidelength
        energy_large_enough = abs(sample.energy) >= cutoff_energy

        return sidelengths_short_enough and energy_large_enough

    predicate = functools.partial(sample_filter, cutoff_energy=1.0e-3, cutoff_average_sidelength=4.0)


if __name__ == "__main__":
    main()
