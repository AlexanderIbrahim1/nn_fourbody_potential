"""
The SampleStatistics type is used to gather relevant statistical information about the side lengths
and energies of the ab initio four-body interaction energy data. This data is used to make informed
decisions about how to filter out data that doesn't need to be included in the model, or would hurt
the model's performance.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Callable
from typing import Union

import numpy as np
from numpy.typing import NDArray

from nn_fourbody_potential.dataio import load_fourbody_training_data

import script_utils


@dataclass
class SampleStatistics:
    max_abs_energy: float
    min_abs_energy: float
    n_samples_above_abs_energy_cutoff: int
    n_samples_below_mean_distance_cutoff: int
    n_samples_satisfying_both_cutoffs: int

    def __repr__(self) -> str:
        return "\n".join(
            [
                f"max_abs_energy                       : {self.max_abs_energy: 0.8e}",
                f"min_abs_energy                       : {self.min_abs_energy: 0.8e}",
                f"n_samples_above_abs_energy_cutoff    : {self.n_samples_above_abs_energy_cutoff}",
                f"n_samples_below_mean_distance_cutoff : {self.n_samples_below_mean_distance_cutoff}",
                f"n_samples_satisfying_both_cutoffs    : {self.n_samples_satisfying_both_cutoffs}",
            ]
        )


def get_sample_statistics(
    side_length_groups: NDArray,
    energies: NDArray,
    energy_predicate: Callable[[float], bool],
    distance_predicate: Callable[[float], bool],
) -> SampleStatistics:
    abs_energies = np.abs(energies)
    mean_side_lengths = np.mean(side_length_groups, axis=1)

    max_abs_energy = np.max(abs_energies)
    min_abs_energy = np.min(abs_energies)
    n_samples_above_abs_energy_cutoff = len([e for e in abs_energies if energy_predicate(e)])
    n_samples_below_mean_distance_cutoff = len([s for s in mean_side_lengths if distance_predicate(s)])
    n_samples_satisfying_both_cutoffs = len(
        [(e, s) for (e, s) in zip(abs_energies, mean_side_lengths) if energy_predicate(e) and distance_predicate(s)]
    )

    return SampleStatistics(
        max_abs_energy,
        min_abs_energy,
        n_samples_above_abs_energy_cutoff,
        n_samples_below_mean_distance_cutoff,
        n_samples_satisfying_both_cutoffs,
    )


if __name__ == "__main__":

    def energy_predicate(e: float):
        return abs(e) > 1.0e-3

    def mean_side_length_predicate(s: float):
        return s <= 4.5

    side_length_groups, energies = script_utils.load_all_raw_abinitio_sampling_training_data()
    statistics = get_sample_statistics(side_length_groups, energies, energy_predicate, mean_side_length_predicate)

    print(statistics)
