"""
This module contains functions for pruning data points from training, testing, and
validation data sets.
"""

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Sequence
from typing import Tuple

from nn_fourbody_potential.dataio import load_fourbody_training_data
from nn_fourbody_potential.sidelength_distributions.sidelength_types import SixSideLengths


@dataclass
class DataSample:
    sidelengths: SixSideLengths
    energy: float


def get_abinitio_hcp_data_filepath() -> Path:
    return Path(".", "data", "abinitio_hcp_data_3901_2.2_4.5.dat")


def mean(x: Sequence[float]) -> float:
    # `statistics.mean()` doesn't work with `torch.Tensor` as an input for whatever reason, and
    # I don't want to switch to `torch.mean()` for loss of generality; so I have to recreate the
    # mean() function from scratch
    if len(x) == 0:
        return 0.0
    else:
        return sum(x) / len(x)


def energy_filter(energy: float, abs_minimum_energy_cutoff: float) -> bool:
    return abs(energy) >= abs(abs_minimum_energy_cutoff)


def sidelengths_filter(sidelengths: SixSideLengths, maximum_average_sidelength_cutoff: float) -> bool:
    average_sidelength = mean(sidelengths)
    return average_sidelength <= maximum_average_sidelength_cutoff
