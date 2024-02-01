"""
This module contains utility functions and constants to help manage the data for the project.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from nn_fourbody_potential.dataio import load_fourbody_training_data

RAW_ABINITIO_DATA_DIRPATH = Path("..", "raw_data")
RAW_ABINITIO_FASTDECAY_DATA_FILEPATH = RAW_ABINITIO_DATA_DIRPATH / "abinitio_fastdecay_data_1500_2.2_4.5.dat"
RAW_ABINITIO_SLOWDECAY_DATA_FILEPATH = RAW_ABINITIO_DATA_DIRPATH / "abinitio_slowdecay_data_9000_2.2_4.5.dat"
RAW_ABINITIO_VERYFASTDECAY_DATA_FILEPATH = RAW_ABINITIO_DATA_DIRPATH / "abinitio_veryfastdecay_data_1500_2.2_4.5.dat"
RAW_ABINITIO_VERYSLOWDECAY_DATA_FILEPATH = RAW_ABINITIO_DATA_DIRPATH / "abinitio_veryslowdecay_data_4000_2.8_4.5.dat"
RAW_ABINITIO_HCP_DATA_FILEPATH = RAW_ABINITIO_DATA_DIRPATH / "abinitio_hcp_data_3901_2.2_4.5.dat"


def load_all_raw_abinitio_sampling_training_data() -> tuple[NDArray, NDArray]:
    fastdecay_sides, fastdecay_energies = load_fourbody_training_data(RAW_ABINITIO_FASTDECAY_DATA_FILEPATH)
    veryfastdecay_sides, veryfastdecay_energies = load_fourbody_training_data(RAW_ABINITIO_VERYFASTDECAY_DATA_FILEPATH)
    slowdecay_sides, slowdecay_energies = load_fourbody_training_data(RAW_ABINITIO_SLOWDECAY_DATA_FILEPATH)
    veryslowdecay_sides, veryslowdecay_energies = load_fourbody_training_data(RAW_ABINITIO_VERYSLOWDECAY_DATA_FILEPATH)

    sides = np.concatenate((fastdecay_sides, veryfastdecay_sides, slowdecay_sides, veryslowdecay_sides))
    energies = np.concatenate((fastdecay_energies, veryfastdecay_energies, slowdecay_energies, veryslowdecay_energies))

    return sides, energies


if __name__ == "__main__":
    sides, energies = load_all_raw_abinitio_sampling_training_data()
    print(sides.shape)
    print(energies.shape)
