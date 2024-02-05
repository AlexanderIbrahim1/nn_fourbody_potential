"""
This module creates a stem plot of all the ab initio energy samples, from:
- the the hcp data
- the original training data
- the fast-decay training data
- the very-fast-decay training data

From this information, we can make an informed choice on how to separate the
training data into regimes for the ensemble neural network.
"""

import matplotlib.pyplot as plt

import nn_fourbody_potential.dataio as dataio
import nn_fourbody_potential_data.data_paths as data_paths


def plot_energies() -> None:
    _, energies_slowdecay = dataio.load_fourbody_training_data(data_paths.RAW_ABINITIO_SLOWDECAY_DATA_FILEPATH)
    _, energies_fastdecay = dataio.load_fourbody_training_data(data_paths.RAW_ABINITIO_FASTDECAY_DATA_FILEPATH)
    _, energies_veryslowdecay = dataio.load_fourbody_training_data(data_paths.RAW_ABINITIO_VERYSLOWDECAY_DATA_FILEPATH)
    _, energies_veryfastdecay = dataio.load_fourbody_training_data(data_paths.RAW_ABINITIO_VERYFASTDECAY_DATA_FILEPATH)

    n_bins = 256
    plt.hist(energies_slowdecay, bins=n_bins, color="C0", alpha=0.5)
    plt.hist(energies_fastdecay, bins=n_bins, color="C1", alpha=0.5)
    plt.hist(energies_veryslowdecay, bins=n_bins, color="C2", alpha=0.5)
    plt.hist(energies_veryfastdecay, bins=n_bins, color="C3", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    plot_energies()
