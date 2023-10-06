"""
This script contains code for playing around with the RescalingPotential properties, and for
finding a useful filter that will play well with the new RescalingPotential.

I would prefer to find a filter that only removes samples based on their pair distances, and
not their energy values.

NOTE: the geometric rescaler, even after playing around with lots of settings, doesn't seem to
be working.
"""

import dataclasses
import functools
import math
import operator

from pathlib import Path

import numpy as np
import torch

from dispersion4b.coefficients import c12_parahydrogen_midzuno_kihara
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON
from nn_fourbody_potential import rescaling
from nn_fourbody_potential.rescaling.utils import _rescale_energies

from sample_filter import SampleFilter
from sample_filter import FullSampleFilter
from sample_filter import MaxSideLengthSampleFilter
from sample_filter import apply_filter


@dataclasses.dataclass(frozen=True)
class GeometricRescalingPotential:
    coeff: float
    expon: float
    disp_coeff: float
    r_transition: float = 4.0
    steepness: float = 2.0

    def __post_init__(self) -> None:
        assert self.coeff > 0.0
        assert self.expon > 0.0
        assert self.disp_coeff > 0.0

    def __call__(self, *six_pair_distances: float) -> float:
        assert len(six_pair_distances) == 6

        pairdist_product = functools.reduce(operator.mul, six_pair_distances, 1.0)
        geometric_average = pairdist_product ** (1.0 / 6.0)

        # tanh_arg = self.steepness * (geometric_average - self.r_transition)
        # frac_expon = 0.5 * (1.0 + math.tanh(tanh_arg))
        # frac_disp = 1.0 - frac_expon

        frac_expon = 1.0
        frac_disp = 1.0

        expon_contribution = frac_expon * self.coeff * math.exp(-self.expon * geometric_average)
        disp_contribution = frac_disp * self.disp_coeff / (pairdist_product**2)

        return expon_contribution + disp_contribution


@dataclasses.dataclass(frozen=True)
class FilePaths:
    hcp_filepath: Path
    sampled_filepath: Path


def main(rescaling_potential: GeometricRescalingPotential, filepaths: FilePaths, data_filter: SampleFilter) -> None:
    # hcp_unfiltered = torch.from_numpy(np.loadtxt(filepaths.hcp_filepath)).view(-1, 7)
    sampled_unfiltered = torch.from_numpy(np.loadtxt(filepaths.sampled_filepath)).view(-1, 7)

    # unfiltered_data = torch.concatenate((hcp_unfiltered, sampled_unfiltered))
    unfiltered_data = sampled_unfiltered
    filtered_data = apply_filter(unfiltered_data, data_filter)
    side_length_groups = filtered_data[:, :6]
    energies = filtered_data[:, 6]

    y_train, res_limits = _rescale_energies(
        side_length_groups, energies, rescaling_potential, omit_final_rescaling_step=True
    )

    min_value = y_train.abs().min().item()
    max_value = y_train.abs().max().item()

    print(res_limits)
    print(f"(min, max, ratio) = ({min_value: 0.8f}, {max_value: 0.8f}, {max_value / min_value: 0.4f})")


if __name__ == "__main__":
    rescaling_potential = GeometricRescalingPotential(
        coeff=ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF / 12.0,
        expon=ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON * 5.0,
        disp_coeff=0.5 * c12_parahydrogen_midzuno_kihara(),
    )
    filepaths = FilePaths(
        hcp_filepath=Path("..", "data", "abinitio_hcp_data_3901.dat"),
        sampled_filepath=Path("..", "data", "abinitio_sampled_data_16000.dat"),
    )
    data_filter = MaxSideLengthSampleFilter(4.4)

    main(rescaling_potential, filepaths, data_filter)

# PLAN
# load the unfiltered hcp and sampled data
# use the following:
#    y_train, res_limits = _rescale_energies(
#        side_length_groups_train, energies_train, rescaling_potential, omit_final_rescaling_step=True
#    )
#    print(res_limits)
# print out the rescaling limits
# print out the maximum and minimum absolute values in y_train
