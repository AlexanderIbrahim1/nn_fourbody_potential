"""
This script is used to filter out energies from the testing, training, and validation sets that I don't
want to be included in the model.

These are samples with really small energies and/or samples with really long side lengths.

The samples omitted from the data are those that would be covered in the overall ExtrapolatedPotential
model as "long-distance samples", and thus would be handled with an analytic potential.
"""

from pathlib import Path
from typing import Callable
from typing import Union

import numpy as np
from numpy.typing import NDArray

from nn_fourbody_potential.dataio import save_fourbody_training_data

import script_utils


def write_pruned_data_files(
    side_length_groups: NDArray,
    energies: NDArray,
    output_data_filepath: Union[str, Path],
    energy_filter: Callable[[float], bool],
    mean_side_length_filter: Callable[[float], bool],
) -> None:
    samples = zip(side_length_groups, energies)

    def sample_filter(s: NDArray, e: float) -> bool:
        return mean_side_length_filter(np.mean(s)) and energy_filter(e)

    allowed_sample_indices = np.array([i for (i, (s, e)) in enumerate(samples) if sample_filter(s, e)])
    print(allowed_sample_indices.shape)
    exit()
    side_length_groups = side_length_groups[allowed_sample_indices]
    energies = energies[allowed_sample_indices]

    save_fourbody_training_data(output_data_filepath, side_length_groups, energies)


def energy_filter(e: float) -> bool:
    return abs(e) > 1.0e-3


def mean_side_length_filter(s: float) -> bool:
    return s <= 4.5


def filter_distribution_data() -> None:
    side_lengths, energies = script_utils.load_all_raw_abinitio_sampling_training_data()
    output_data_filepath = Path("whatever.dat")

    write_pruned_data_files(side_lengths, energies, output_data_filepath, energy_filter, mean_side_length_filter)


if __name__ == "__main__":
    filter_distribution_data()
