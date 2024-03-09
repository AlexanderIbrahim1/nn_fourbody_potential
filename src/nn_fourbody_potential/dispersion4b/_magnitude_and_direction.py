from dataclasses import dataclass

from nn_fourbody_potential.cartesian import Cartesian3D
from nn_fourbody_potential.cartesian import norm


@dataclass(frozen=True)
class MagnitudeAndDirection:
    magnitude: float
    direction: Cartesian3D


def distance_and_unit_vector(p_i: Cartesian3D, p_j: Cartesian3D) -> MagnitudeAndDirection:
    """Calculate the distance and unit vector of the separation 'p_i - p_j'."""
    p_ij = p_i - p_j
    distance = norm(p_ij)
    unit_vec = p_ij / distance

    return MagnitudeAndDirection(distance, unit_vec)
