"""
This script is used to compare the energies calculated by the trained models
against the test energies.
"""

from pathlib import Path
import numpy as np
from numpy.typing import NDArray

SIZE_TO_LABEL: dict[int, str] = {
    8: "8_16_16_8",
    16: "16_32_32_16",
    32: "32_64_64_32",
    64: "64_128_128_64",
}


def rmse(actual: NDArray, predicted: NDArray) -> float:
    return np.sqrt(np.mean((actual - predicted) ** 2))


def main(size: int) -> None:
    energies_filename = f"test_and_predicted_energies_ssp_{SIZE_TO_LABEL[size]}.dat"
    energies_dirpath = Path(".", "test_and_predicted_energies")
    energies_filepath = energies_dirpath / energies_filename
    test_energies, predicted_energies = np.loadtxt(energies_filepath, unpack=True)

    energy_rmse = rmse(test_energies, predicted_energies)

    print(f"The RMSE for the size {size} model is {energy_rmse: 12.8f} wvn")


if __name__ == "__main__":
    size = 64
    main(size)
