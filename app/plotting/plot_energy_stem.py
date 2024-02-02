"""
This module creates a stem plot of all the ab initio energy samples, from:
- the the hcp data
- the original training data
- the fast-decay training data
- the very-fast-decay training data

From this information, we can make an informed choice on how to separate the
training data into regimes for the ensemble neural network.
"""

import sys

sys.path.insert(0, "..")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

import model_info
import training


def plot_energies() -> None:
    # hcp_filepath = Path("..", "data", "abinitio_hcp_data_3901_2.2_4.5.dat")
    slowdecay_training_filepath = Path("..", "data", "abinitio_training_data_5000_2.2_4.5.dat")
    slowdecay_testing_filepath = Path("..", "data", "abinitio_testing_data_2000_2.2_4.5.dat")
    slowdecay_validation_filepath = Path("..", "data", "abinitio_validation_data_2000_2.2_4.5.dat")
    fastdecay_training_filepath = Path("..", "data", "abinitio_fastdecay_data_1500_2.2_4.5.dat")
    veryfastdecay_training_filepath = Path("..", "data", "abinitio_veryfastdecay_data_1500_2.2_4.5.dat")

    transformers = model_info.get_data_transforms()
    # _, energies_hcp = training.prepared_data(hcp_filepath, transformers)
    _, energies_slowdecay_training = training.prepared_data(slowdecay_training_filepath, transformers)
    _, energies_slowdecay_testing = training.prepared_data(slowdecay_testing_filepath, transformers)
    _, energies_slowdecay_validation = training.prepared_data(slowdecay_validation_filepath, transformers)
    _, energies_fastdecay_training = training.prepared_data(fastdecay_training_filepath, transformers)
    _, energies_veryfastdecay_training = training.prepared_data(veryfastdecay_training_filepath, transformers)

    energies_combined = (
        torch.concatenate(
            (
                #            energies_hcp,
                energies_slowdecay_training,
                energies_slowdecay_testing,
                energies_slowdecay_validation,
                energies_fastdecay_training,
                energies_veryfastdecay_training,
            )
        )
        .reshape(-1)
        .cpu()
        .detach()
        .numpy()
    )

    # energies_combined = np.log(np.abs(energies_combined))

    plt.hist(energies_combined, bins=1000)
    plt.show()


if __name__ == "__main__":
    plot_energies()
