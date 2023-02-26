"""
This module contains components for generating the training data for the four-body
potential energy surface.

The training data is made up of the 6 relative side lengths of a four-body geometry,
and the corresponding energy value.
"""

from __future__ import annotations

import numpy as np

from itertools import combinations
from typing import Annotated
from typing import Sequence
from typing import Tuple

from cartesian import Cartesian3D
from cartesian.measure import distance
from dispersion4b.shortrange.four_body_analytic_potential import (
    FourBodyAnalyticPotential,
)
from hydro4b_coords.generate.discretized_distribution import DiscretizedDistribution
from hydro4b_coords.generate.generate import sample_fourbody_geometry

FourCartesianPoints = Annotated[Sequence[Cartesian3D], 4]
SixSideLengths = Tuple[float, float, float, float, float, float]


def generate_training_data(
    n_samples: int,
    distrib: DiscretizedDistribution,
    potential: FourBodyAnalyticPotential,
) -> Tuple[Sequence[SixSideLengths], Sequence[float]]:
    """
    Use the distribution 'distrib' to sample four-body geometries. These geometries
    are used to calculate interaction energies from the 'potential'. The sidelengths
    and interaction energies of these geometries are used as the training data for
    the NN models.
    """
    assert n_samples > 0

    n_sidelengths = 6
    sample_sidelengths = np.empty((n_samples, n_sidelengths), dtype=float)
    sample_energies = np.empty(n_samples, dtype=float)

    for i_sample in range(n_samples):
        points = sample_fourbody_geometry(distrib)
        sample_energies[i_sample] = potential(points)
        sample_sidelengths[i_sample] = get_sidelengths(points)

    return sample_sidelengths, sample_energies


def get_sidelengths(points: FourCartesianPoints) -> SixSideLengths:
    # NOTE: I *could* modify hydro4b_coords.generate.sample_fourbody_geometry to also return side lengths
    #     -> but they might be out of order from the actually generated points?
    #     -> so I'll recalculate them from the points just in case
    # NOTE: this isn't a significant performance bottleneck yet
    return tuple([distance(p0, p1) for (p0, p1) in combinations(points, 2)])
