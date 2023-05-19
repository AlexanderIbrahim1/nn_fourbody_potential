"""
The 'ab initio tetrahedron distribution' is a probability distribution for the sidelengths
of four-body geometries.

The probability distribution for the sidelengths is an exponential decay, and matches the
decay rate of the ab initio four-body parahydrogen interaction energies of the tetrahedron.
"""

from hydro4b_coords.generate.discretized_distribution import DiscretizedDistribution
from hydro4b_coords.generate.distributions import exponential_decay_distribution

from nn_fourbody_potential import constants


def get_abinit_tetrahedron_distribution(
    x_min: float,
    x_max: float,
    *,
    coeff: float = constants.ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF,
    decay_rate: float = constants.ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON,
) -> DiscretizedDistribution:
    distrib = exponential_decay_distribution(
        n_terms=1024,
        x_min=x_min,
        x_max=x_max,
        coeff=coeff,
        decay_rate=decay_rate,
    )

    return distrib
