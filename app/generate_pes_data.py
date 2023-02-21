"""
This module contains components for generating the training data for the four-body
potential energy surface.

The training data is made up of the 6 relative side lengths of a four-body geometry,
and the corresponding energy value.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Annotated
from typing import Sequence
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from cartesian import Cartesian3D
from cartesian.measure import distance

from hydro4b_coords.generate.distributions import exponential_decay_distribution
from hydro4b_coords.generate.discretized_distribution import DiscretizedDistribution
from hydro4b_coords.generate.generate import sample_fourbody_geometry

from nn_fourbody_potential.fourbody_potential import create_four_body_analytic_potential
from nn_fourbody_potential import constants

from dataio import save_fourbody_training_data


FourCartesianPoints = Annotated[Sequence[Cartesian3D], 4]
SixSideLengths = Tuple[float, float, float, float, float, float]


def get_sidelengths(points: FourCartesianPoints) -> SixSideLengths:
    # NOTE: I *could* modify hydro4b_coords.generate.sample_fourbody_geometry to also return side lengths
    #     -> but they might be out of order from the actually generated points?
    #     -> so I'll recalculate them from the points just in case
    return tuple([distance(p0, p1) for (p0, p1) in combinations(points, 2)])


def generate_training_data(
    n_samples: int, distrib: DiscretizedDistribution
) -> Tuple[Sequence[SixSideLengths], Sequence[float]]:
    """
    Use a 'FourBodyAnalyticPotential' to generate the training data for the NN models.
    """
    assert n_samples > 0

    pot = create_four_body_analytic_potential()

    n_sidelengths = 6
    sample_sidelengths = np.empty((n_samples, n_sidelengths), dtype=float)
    sample_energies = np.empty(n_samples, dtype=float)

    for i_sample in range(n_samples):
        points = sample_fourbody_geometry(distrib)
        sample_energies[i_sample] = pot(points)
        sample_sidelengths[i_sample] = get_sidelengths(points)

    return sample_sidelengths, sample_energies

def get_distribution() -> DiscretizedDistribution:
    distrib = exponential_decay_distribution(
        n_terms=1024,
        x_min=2.2,
        x_max=10.0,
        coeff=constants.ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF,
        decay_rate=constants.ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON,
    )

    return distrib

def main() -> None:
    distrib = get_distribution()
    n_samples = 20000
    filename = Path('data', 'training_data_20000.dat')

    sidelengths, energies = generate_training_data(n_samples, distrib)
    save_fourbody_training_data(filename, sidelengths, energies)

if __name__ == "__main__":
    main()
