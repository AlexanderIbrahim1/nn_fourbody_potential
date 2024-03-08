"""
This module contains components for generating the training data for the four-body
potential energy surface.

The training data is made up of the 6 relative side lengths of a four-body geometry,
and the corresponding energy value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from cartesian.operations import relative_pair_distances
from hydro4b_coords.geometries import MAP_GEOMETRY_TAG_TO_FUNCTION
from hydro4b_coords.generate.discretized_distribution import DiscretizedDistribution
from hydro4b_coords.generate.distributions import exponential_decay_distribution
from hydro4b_coords.generate.generate import sample_fourbody_geometry

from nn_fourbody_potential.dataio import save_fourbody_sidelengths
from nn_fourbody_potential.common_types import FourCartesianPoints
from nn_fourbody_potential import constants


def get_abinit_tetrahedron_distribution(
    x_min: float,
    x_max: float,
    *,
    coeff: float = constants.ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF,
    decay_rate: float = constants.ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON,
) -> DiscretizedDistribution:
    """
    The 'ab initio tetrahedron distribution' is a probability distribution for the sidelengths
    of four-body geometries.

    The probability distribution for the sidelengths is an exponential decay, and matches the
    decay rate of the ab initio four-body parahydrogen interaction energies of the tetrahedron.
    """
    distrib = exponential_decay_distribution(
        n_terms=1024,
        x_min=x_min,
        x_max=x_max,
        coeff=coeff,
        decay_rate=decay_rate,
    )

    return distrib


def generate_hcp_points() -> Sequence[FourCartesianPoints]:
    """
    Include the sidelengths and energies for all the four-body geometries calculated in
    the investigation of the frozen HCP lattice.
    """
    eps = 1.0e-6  # so arange doesn't miss the last element
    lattice_constants = np.arange(2.2, 4.5 + eps, 0.05)

    groups_of_points = []
    for point_creating_function in MAP_GEOMETRY_TAG_TO_FUNCTION.values():
        groups_of_points.extend([point_creating_function(lc) for lc in lattice_constants])

    return groups_of_points


def main() -> None:
    n_samples = 300
    decay_rate = constants.ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON
    distrib = get_abinit_tetrahedron_distribution(1.5, 6.0, decay_rate=decay_rate)

    dist_points = [sample_fourbody_geometry(distrib) for _ in range(n_samples)]
    dist_sidelengths = np.array([relative_pair_distances(pts) for pts in dist_points])

    # dist_sidelengths = np.array([get_sidelengths(pts) for pts in dist_points]).reshape(-1)

    # fig, ax = plt.subplots()
    # ax.hist(dist_sidelengths, bins="auto")
    # plt.show()

    # hcp_sidelengths, hcp_energies = generate_hcp_pes_data(potential)

    # potential = create_fourbody_analytic_potential()
    # dist_energies = np.array([potential(pts) for pts in dist_points])

    # sidelengths = np.concatenate((dist_sidelengths, hcp_sidelengths))
    sidelengths = dist_sidelengths

    filename = Path("sample_cases", f"all_range_random.dat")
    save_fourbody_sidelengths(filename, sidelengths)


if __name__ == "__main__":
    main()
