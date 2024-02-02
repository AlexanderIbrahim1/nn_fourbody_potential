"""
The SampleStatistics type is used to gather relevant statistical information about the side lengths
and energies of the ab initio four-body interaction energy data. This data is used to make informed
decisions about how to filter out data that doesn't need to be included in the model, or would hurt
the model's performance.
"""

from dataclasses import dataclass
from typing import Callable

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


# FILTERING RESULTS
# HCP DATA
# apply side length filter (4.5) : 3901 samples -> 1610 samples (remove 2291)
# apply energy filter (1.0e-3)   : 1610 samples -> 1512 samples (remove 98)
# SAMPLED DATA
# apply side length filter (4.5) : 16000 samples -> 16000 samples (remove 0) (obvious; no sample has side length that great)
# apply energy filter (1.0e-3)   : 16000 samples -> 15870 samples (remove 130)
#
# so the energy filter removes about 1.15 % of the data
# - essentially all of them are long-range anyways
# - worth removing a small amount of data from training, to improve the stability
#   - even if their prediction worsens
if __name__ == "__main__":

    def energy_predicate(e: float):
        return abs(e) > 0.0e-3

    def mean_side_length_predicate(s: float):
        return s <= 4.5

    def print_hcp_statistics():
        filepath = script_utils.RAW_ABINITIO_HCP_DATA_FILEPATH
        side_length_groups, energies = load_fourbody_training_data(filepath)
        statistics = get_sample_statistics(side_length_groups, energies, energy_predicate, mean_side_length_predicate)
        print(statistics)
        # max_abs_energy                       :  2.59012482e+02
        # min_abs_energy                       :  9.21450000e-08
        # n_samples_above_abs_energy_cutoff    : 3901
        # n_samples_below_mean_distance_cutoff : 1610
        # n_samples_satisfying_both_cutoffs    : 1610

    def print_distribution_sampled_statistics():
        side_length_groups, energies = script_utils.load_all_raw_abinitio_sampling_training_data()
        statistics = get_sample_statistics(side_length_groups, energies, energy_predicate, mean_side_length_predicate)
        print(statistics)
        # max_abs_energy                       :  1.78182125e+02
        # min_abs_energy                       :  1.16646650e-05
        # n_samples_above_abs_energy_cutoff    : 16000
        # n_samples_below_mean_distance_cutoff : 16000
        # n_samples_satisfying_both_cutoffs    : 16000

    print_hcp_statistics()
    print_distribution_sampled_statistics()
