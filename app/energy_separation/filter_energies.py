"""
This script is used to filter out energies from the testing, training, and validation sets that I don't
want to be included in the model. The samples omitted from the data are those that would be covered in
the overall ExtrapolatedPotential model as "long-distance samples", and thus would be handled with an
analytic potential.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Callable
from typing import Union

import numpy as np

from nn_fourbody_potential.dataio import load_fourbody_training_data
from nn_fourbody_potential.dataio import save_fourbody_training_data


@dataclass
class SampleStatistics:
    max_abs_energy: float
    min_abs_energy: float
    n_samples_below_abs_energy_cutoff: int
    n_samples_above_mean_distance_cutoff: int
    n_samples_violating_both_cutoffs: int


def get_sample_statistics(
    energy_filepath: Union[str, Path],
    energy_predicate: Callable[[float], bool],
    distance_predicate: Callable[[float], bool],
) -> SampleStatistics:
    side_length_groups, energies = load_fourbody_training_data(energy_filepath)

    abs_energies = np.abs(energies)
    mean_side_lengths = np.mean(side_length_groups, axis=1)

    max_abs_energy = np.max(abs_energies)
    min_abs_energy = np.min(abs_energies)
    n_samples_below_abs_energy_cutoff = len([e for e in abs_energies if energy_predicate(e)])
    n_samples_above_mean_distance_cutoff = len([s for s in mean_side_lengths if distance_predicate(s)])
    n_samples_violating_both_cutoffs = len(
        [(e, s) for (e, s) in zip(abs_energies, mean_side_lengths) if energy_predicate(e) and distance_predicate(s)]
    )

    return SampleStatistics(
        max_abs_energy,
        min_abs_energy,
        n_samples_below_abs_energy_cutoff,
        n_samples_above_mean_distance_cutoff,
        n_samples_violating_both_cutoffs,
    )


def write_pruned_data_files(
    input_data_filepath: Union[str, Path],
    output_data_filepath: Union[str, Path],
    energy_filter: Callable[[float], bool],
    mean_side_length_filter: Callable[[float], bool],
) -> None:
    side_length_groups, energies = load_fourbody_training_data(input_data_filepath)

    allowed_sample_indices = np.array(
        [
            i
            for (i, (e, s)) in enumerate(zip(energies, side_length_groups))
            if energy_filter(e) and mean_side_length_filter(np.mean(s))
        ]
    )

    side_length_groups = side_length_groups[allowed_sample_indices]
    energies = energies[allowed_sample_indices]

    save_fourbody_training_data(output_data_filepath, side_length_groups, energies)


if __name__ == "__main__":
    input_data_filepath = Path(".", "data", "all_energy_valid.dat")
    output_data_filepath = Path(".", "data", "all_energy_valid_filtered.dat")
    energy_filter = lambda e: abs(e) > 1.0e-3
    mean_side_length_filter = lambda s: s <= 4.5

    write_pruned_data_files(input_data_filepath, output_data_filepath, energy_filter, mean_side_length_filter)


# - fix the rescaled energy cutoff training pipeline
# - put the tensors on the GPU; maybe I've been missing out on performance?
# - use this `energy_predicate` and `distance_predicate` stuff to find out how many small energy samples remain
#   based on the mean distance cutoff
