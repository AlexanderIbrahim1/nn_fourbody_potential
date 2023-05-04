"""
This module contains components for generating the training data for the four-body
potential energy surface.

The training data is made up of the 6 relative side lengths of a four-body geometry,
and the corresponding energy value.
"""

from __future__ import annotations

from itertools import combinations
from cartesian.measure import distance

from nn_fourbody_potential.sidelength_distributions.sidelength_types import FourCartesianPoints
from nn_fourbody_potential.sidelength_distributions.sidelength_types import SixSideLengths


def get_sidelengths(points: FourCartesianPoints) -> SixSideLengths:
    # NOTE: I *could* modify hydro4b_coords.generate.sample_fourbody_geometry to also return side lengths
    #     -> but they might be out of order from the actually generated points?
    #     -> so I'll recalculate them from the points just in case
    # NOTE: this isn't a significant performance bottleneck yet
    return tuple([distance(p0, p1) for (p0, p1) in combinations(points, 2)])
