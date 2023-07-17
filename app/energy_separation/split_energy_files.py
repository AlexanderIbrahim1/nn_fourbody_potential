"""
This module contains functions that split the hcp and sampled ab initio energies
into low, middle, and high energy regimes, as well as into training, testing, and
validation sets.
"""

import sys

sys.path.insert(0, "..")  # noqa

from pathlib import Path
from typing import Any
from typing import Callable
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import torch

from nn_fourbody_potential.transformations.transformers import SixSideLengthsTransformer

import model_info
import training


def get_prepared_data_as_numpy(
    data_filepath: Path, transformers: Sequence[SixSideLengthsTransformer]
) -> tuple[NDArray, NDArray]:
    x_data, y_data = training.prepared_data(data_filepath, transformers)
    x_data = x_data.cpu().detach().numpy()
    y_data = y_data.cpu().detach().numpy().reshape(-1)

    return x_data, y_data


def get_lower_upper_ends(energy_centre: float, energy_width: float) -> float:
    energy_lower_end = energy_centre - energy_width
    energy_upper_end = energy_centre + energy_width

    return energy_lower_end, energy_upper_end


def filter_data_based_on_y_data(
    x_data: NDArray, y_data: NDArray, y_data_filter: Callable[[Any], bool]
) -> tuple[NDArray, NDArray]:
    mask = np.array([y_data_filter(y_sample) for y_sample in y_data])

    return x_data[mask], y_data[mask]


def split_integer_into_three(value: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    """
    Takes an integer, and splits it into three integers according to the fractions provided.
    This function takes rounding into account, and makes sure that the sum of the output integers
    always equals `value`.
    """
    assert value > 0
    assert all([frac > 0.0 for frac in fractions])

    part_one = round(value * fractions[0] / sum(fractions))
    remainder = value - part_one

    part_two = round(remainder * fractions[1] / sum(fractions[1:]))
    part_three = remainder - part_two

    return part_one, part_two, part_three


def data_boundaries(value: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    n_train, n_test, n_valid = split_integer_into_three(value, fractions)

    return (n_train, n_train + n_test, n_train + n_test + n_valid)


def split_data(
    sampled_data: NDArray, hcp_data: NDArray, i_train: int, i_test: int, i_valid: int
) -> tuple[NDArray, NDArray, NDArray]:
    train_data = np.concatenate((hcp_data, sampled_data[:i_train]))
    test_data = sampled_data[i_train:i_test]
    valid_data = sampled_data[i_test:i_valid]

    return train_data, test_data, valid_data


def save_sidelengths_and_energies(filepath: Path, sidelengths: NDArray, energies: NDArray) -> None:
    energies = energies.reshape(-1, 1)
    savedata = np.hstack((sidelengths, energies))

    np.savetxt(filepath, savedata)


def main() -> None:
    sampled_data_filepath = Path("data", "abinitio_sampled_data_16000.dat")
    hcp_data_filepath = Path("data", "abinitio_hcp_data_3901.dat")

    transformers = []  # model_info.get_data_transforms()

    x_sampled, y_sampled = get_prepared_data_as_numpy(sampled_data_filepath, transformers)
    x_hcp, y_hcp = get_prepared_data_as_numpy(hcp_data_filepath, transformers)

    energy_lowmid_lower_end, energy_lowmid_upper_end = get_lower_upper_ends(energy_centre=1.0, energy_width=0.0)
    energy_midhigh_lower_end, energy_midhigh_upper_end = get_lower_upper_ends(energy_centre=10.0, energy_width=0.0)

    low_filter = lambda eng: np.abs(eng) <= energy_lowmid_upper_end
    mid_filter = lambda eng: energy_lowmid_lower_end < np.abs(eng) < energy_midhigh_upper_end
    high_filter = lambda eng: energy_midhigh_lower_end <= np.abs(eng)

    x_low_sampled, y_low_sampled = filter_data_based_on_y_data(x_sampled, y_sampled, low_filter)
    x_mid_sampled, y_mid_sampled = filter_data_based_on_y_data(x_sampled, y_sampled, mid_filter)
    x_high_sampled, y_high_sampled = filter_data_based_on_y_data(x_sampled, y_sampled, high_filter)

    x_low_hcp, y_low_hcp = filter_data_based_on_y_data(x_hcp, y_hcp, low_filter)
    x_mid_hcp, y_mid_hcp = filter_data_based_on_y_data(x_hcp, y_hcp, mid_filter)
    x_high_hcp, y_high_hcp = filter_data_based_on_y_data(x_hcp, y_hcp, high_filter)

    i_low_train, i_low_test, i_low_valid = data_boundaries(len(x_low_sampled), (0.7, 0.15, 0.15))
    i_mid_train, i_mid_test, i_mid_valid = data_boundaries(len(x_mid_sampled), (0.7, 0.15, 0.15))
    i_high_train, i_high_test, i_high_valid = data_boundaries(len(x_high_sampled), (0.7, 0.15, 0.15))

    x_low_sampled_train, x_low_sampled_test, x_low_sampled_valid = split_data(
        x_low_sampled, x_low_hcp, i_low_train, i_low_test, i_low_valid
    )
    y_low_sampled_train, y_low_sampled_test, y_low_sampled_valid = split_data(
        y_low_sampled, y_low_hcp, i_low_train, i_low_test, i_low_valid
    )
    x_mid_sampled_train, x_mid_sampled_test, x_mid_sampled_valid = split_data(
        x_mid_sampled, x_mid_hcp, i_mid_train, i_mid_test, i_mid_valid
    )
    y_mid_sampled_train, y_mid_sampled_test, y_mid_sampled_valid = split_data(
        y_mid_sampled, y_mid_hcp, i_mid_train, i_mid_test, i_mid_valid
    )
    x_high_sampled_train, x_high_sampled_test, x_high_sampled_valid = split_data(
        x_high_sampled, x_high_hcp, i_high_train, i_high_test, i_high_valid
    )
    y_high_sampled_train, y_high_sampled_test, y_high_sampled_valid = split_data(
        y_high_sampled, y_high_hcp, i_high_train, i_high_test, i_high_valid
    )

    x_sampled_train = np.concatenate((x_low_sampled_train, x_mid_sampled_train, x_high_sampled_train))
    y_sampled_train = np.concatenate((y_low_sampled_train, y_mid_sampled_train, y_high_sampled_train))
    x_sampled_test = np.concatenate((x_low_sampled_test, x_mid_sampled_test, x_high_sampled_test))
    y_sampled_test = np.concatenate((y_low_sampled_test, y_mid_sampled_test, y_high_sampled_test))
    x_sampled_valid = np.concatenate((x_low_sampled_valid, x_mid_sampled_valid, x_high_sampled_valid))
    y_sampled_valid = np.concatenate((y_low_sampled_valid, y_mid_sampled_valid, y_high_sampled_valid))

    save_sidelengths_and_energies(
        Path("data", "all_energy_train.dat"), x_sampled_train, y_sampled_train
    )  # done when energy width = 0 for both
    save_sidelengths_and_energies(
        Path("data", "all_energy_test.dat"), x_sampled_test, y_sampled_test
    )  # done when energy width = 0 for both
    save_sidelengths_and_energies(
        Path("data", "all_energy_valid.dat"), x_sampled_valid, y_sampled_valid
    )  # done when energy width = 0 for both

    # save_sidelengths_and_energies(Path("data", "low_energy_train.dat"), x_low_sampled_train, y_low_sampled_train)
    # save_sidelengths_and_energies(Path("data", "low_energy_test.dat"), x_low_sampled_test, y_low_sampled_test)
    # save_sidelengths_and_energies(Path("data", "low_energy_valid.dat"), x_low_sampled_valid, y_low_sampled_valid)
    # save_sidelengths_and_energies(Path("data", "mid_energy_train.dat"), x_mid_sampled_train, y_mid_sampled_train)
    # save_sidelengths_and_energies(Path("data", "mid_energy_test.dat"), x_mid_sampled_test, y_mid_sampled_test)
    # save_sidelengths_and_energies(Path("data", "mid_energy_valid.dat"), x_mid_sampled_valid, y_mid_sampled_valid)
    # save_sidelengths_and_energies(Path("data", "high_energy_train.dat"), x_high_sampled_train, y_high_sampled_train)
    # save_sidelengths_and_energies(Path("data", "high_energy_test.dat"), x_high_sampled_test, y_high_sampled_test)
    # save_sidelengths_and_energies(Path("data", "high_energy_valid.dat"), x_high_sampled_valid, y_high_sampled_valid)


if __name__ == "__main__":
    # values = split_integer_into_three(2041, (0.7, 0.15, 0.15))
    # print(values, sum(values))
    main()
