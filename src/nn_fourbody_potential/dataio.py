"""
This module contains functions for saving and loading training data for the four-body
potential energy surface.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SIDELENGTH_COLUMNS = (0, 1, 2, 3, 4, 5)
ENERGIES_COLUMNS = (6,)


def save_fourbody_training_data(filename: Path, sidelengths: NDArray, energies: NDArray) -> None:
    """
    The training data is saved as rows of 7 space-separated values. The first six columns
    in each row are the six sidelengths. The last column is the energy.
    """
    _check_training_data_dimensions(sidelengths, energies)

    with open(filename, "w") as fout:
        for sidelen, energy in zip(sidelengths, energies):
            fout.write(_format_data_line(sidelen, energy))


def load_fourbody_training_data(filename: Path) -> tuple[NDArray, NDArray]:
    """
    The training data is saved as rows of 7 space-separated values. The first six columns
    in each row are the six sidelengths. The last column is the energy.
    """
    # NOTE: I don't know how to load the sidelengths and the energies in a single statement,
    # but it isn't really a performance problem
    sidelengths = np.loadtxt(filename, usecols=SIDELENGTH_COLUMNS)
    energies = np.loadtxt(filename, usecols=ENERGIES_COLUMNS)

    return sidelengths, energies


def save_fourbody_sidelengths(filename: Path, sidelengths: NDArray) -> None:
    """The sidelength data is saved as rows of 6 space-separated values."""
    assert sidelengths.shape == (len(sidelengths), len(SIDELENGTH_COLUMNS))

    with open(filename, "w") as fout:
        for sidelen in sidelengths:
            fout.write(_format_sidelength_line(sidelen))


def load_fourbody_sidelengths(filename: Path) -> NDArray:
    """This light wrapper function exists to keep the API consistent."""
    return np.loadtxt(filename, usecols=SIDELENGTH_COLUMNS)


def _format_data_line(sidelengths: tuple[float, ...], energy: float) -> str:
    delimiter = "   "
    formatted_sidelengths = [f"{s: .12e}" for s in sidelengths]
    formatted_energy = f"{energy: .12e}\n"
    return delimiter.join(formatted_sidelengths) + delimiter + formatted_energy


def _check_training_data_dimensions(sidelengths: NDArray, energies: NDArray) -> None:
    """Performs a sanity check on the dimensions of the training data."""
    assert len(sidelengths) == len(energies)

    n_samples = len(sidelengths)
    assert sidelengths.shape == (n_samples, len(SIDELENGTH_COLUMNS))
    assert energies.shape == (n_samples,)


def _format_sidelength_line(sidelengths: tuple[float, ...]) -> str:
    delimiter = "   "
    formatted_sidelengths = [f"{s: .12e}" for s in sidelengths]
    return delimiter.join(formatted_sidelengths) + "\n"
