"""
This script removes the hcp samples from the training data.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nn_fourbody_potential.dataio import save_fourbody_training_data

# PLAN
# read in all the hcp data
# read in all the data from `all_energy_train_filtered.dat`
# for each sample in the hcp data
# - try to find it in the filtered training data
# - if it is there, remove it


def filtered_samples(samples: NDArray) -> NDArray:
    energy_filter = lambda e: abs(e) > 1.0e-3
    mean_side_length_filter = lambda s: s <= 4.5

    sample_filter = lambda sample: mean_side_length_filter(np.mean(sample[:6])) and energy_filter(sample[6])
    index_mask = np.array([sample_filter(s) for s in samples])

    return samples[index_mask]


def main() -> None:
    hcp_filepath = Path(".", "data", "abinitio_hcp_data_3901.dat")
    train_filepath = Path(".", "data", "all_energy_train_filtered.dat")
    hcp_data = np.loadtxt(hcp_filepath)
    train_data = np.loadtxt(train_filepath)

    filtered_hcp_data = filtered_samples(hcp_data)
    print(filtered_hcp_data.shape)
    exit()

    for i_hcp, hcp_sample in enumerate(filtered_hcp_data):
        print(i_hcp)
        flag_found = False
        for i, train_sample in enumerate(train_data):
            if np.allclose(hcp_sample, train_sample):
                print("FOUND!!!")
                train_data = np.delete(train_data, i, axis=0)
                flag_found = True
                break
        if not flag_found:
            raise RuntimeError("COULD NOT FIND THE SAMPLE!!!!")

    filtered_data_filepath = Path(".", "data", "all_energy_train_filtered_no_hcp.dat")
    side_length_groups = train_data[:, :6]
    energies = train_data[:, 6]
    save_fourbody_training_data(filtered_data_filepath, side_length_groups, energies)


if __name__ == "__main__":
    main()
