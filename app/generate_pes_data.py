"""
This module contains components for generating the training data for the four-body
potential energy surface.

The training data is made up of the 6 relative side lengths of a four-body geometry,
and the corresponding energy value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated
from typing import Sequence
from typing import Tuple

import numpy as np

from cartesian import CartesianND
from cartesian.operations import relative_pair_distances
from hydro4b_coords.geometries import MAP_GEOMETRY_TAG_TO_FUNCTION
from hydro4b_coords.generate.generate import sample_fourbody_geometry

from nn_fourbody_potential.dataio import save_fourbody_sidelengths
from nn_fourbody_potential.fourbody_potential import FourBodyAnalyticPotential
from nn_fourbody_potential.sidelength_distributions import get_abinit_tetrahedron_distribution
from nn_fourbody_potential.sidelength_distributions import get_sidelengths
from nn_fourbody_potential.sidelength_distributions import SixSideLengths

FourCartesianPoints = Annotated[Sequence[CartesianND], 4]


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


def generate_hcp_pes_data(
    potential: FourBodyAnalyticPotential,
) -> Tuple[Sequence[SixSideLengths], Sequence[float]]:
    groups_of_points = generate_hcp_points()
    sidelengths = np.array([relative_pair_distances(points) for points in groups_of_points])
    energies = np.array([potential(points) for points in groups_of_points])

    return sidelengths, energies


def main() -> None:
    n_samples = 2000
    distrib = get_abinit_tetrahedron_distribution(2.2, 4.5)

    dist_points = [sample_fourbody_geometry(distrib) for _ in range(n_samples)]
    dist_sidelengths = np.array([get_sidelengths(pts) for pts in dist_points])
    # hcp_sidelengths, hcp_energies = generate_hcp_pes_data(potential)

    # potential = create_fourbody_analytic_potential()
    # dist_energies = np.array([potential(pts) for pts in dist_points])

    # sidelengths = np.concatenate((dist_sidelengths, hcp_sidelengths))
    sidelengths = dist_sidelengths

    filename = Path("data", f"sample_sidelengths_{len(sidelengths)}_2.2_4.5.dat")
    save_fourbody_sidelengths(filename, sidelengths)


if __name__ == "__main__":
    main()
