"""
This module contains functions for saving and loading training data for the four-body
potential energy surface.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_fourbody_training_data(
    filename: Path, sidelengths: np.ndarray[float, float], energies: np.ndarray[float]
) -> None:
    assert len(sidelengths) == len(energies)

    with open(filename, "w") as fout:
        for (sidelen, energy) in zip(sidelengths, energies):
            fout.write(_format_line(sidelen, energy))


def load_fourbody_training_data(filename: Path) -> None:
    sidelengths = np.loadtxt(filename, usecols=(0, 1, 2, 3, 4, 5))
    energies = np.loadtxt(filename, usecols=(6,))

    return sidelengths, energies


def _format_line(sidelengths: Tuple[float, ...], energy: float) -> str:
    line = ""
    for sidelen in sidelengths:
        line += f"{sidelen: .12e}   "
    line += f"{energy: .12e}\n"

    return line

