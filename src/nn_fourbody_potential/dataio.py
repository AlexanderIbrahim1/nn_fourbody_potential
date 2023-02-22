"""
This module contains functions for saving and loading training data for the four-body
potential energy surface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

SIDELENGTH_COLUMNS = (0, 1, 2, 3, 4, 5)
ENERGIES_COLUMNS = (6,)


def save_fourbody_training_data(
    filename: Path, sidelengths: np.ndarray[float, float], energies: np.ndarray[float]
) -> None:
    """
    The training data is saved as rows of 7 space-separated values. The first six columns
    in each row are the six sidelengths. The last column is the energy.
    """
    _check_training_data_dimensions(sidelengths, energies)

    with open(filename, "w") as fout:
        for (sidelen, energy) in zip(sidelengths, energies):
            fout.write(_format_line(sidelen, energy))


def load_fourbody_training_data(
    filename: Path,
) -> Tuple[np.ndarray[float, float], np.ndarray[float]]:
    """
    The training data is saved as rows of 7 space-separated values. The first six columns
    in each row are the six sidelengths. The last column is the energy.
    """
    # NOTE: I don't know how to load the sidelengths and the energies in a single statement,
    # but it isn't really a performance problem
    sidelengths = np.loadtxt(filename, usecols=SIDELENGTH_COLUMNS)
    energies = np.loadtxt(filename, usecols=ENERGIES_COLUMNS)

    return sidelengths, energies


def _format_line(sidelengths: Tuple[float, ...], energy: float) -> str:
    line = ""
    for sidelen in sidelengths:
        line += f"{sidelen: .12e}   "
    line += f"{energy: .12e}\n"

    return line


def _check_training_data_dimensions(
    sidelengths: np.ndarray[float, float], energies: np.ndarray[float]
) -> None:
    """Performs a sanity check on the dimensions of the training data."""
    assert len(sidelengths) == len(energies)

    n_samples = len(sidelengths)
    assert sidelengths.shape == (n_samples, len(SIDELENGTH_COLUMNS))
    assert energies.shape == (n_samples,)
