"""
The 'ab initio tetrahedron distribution' is a probability distribution for the sidelengths
of four-body geometries.

The probability distribution for the sidelengths is an exponential decay, and matches the
decay rate of the ab initio four-body parahydrogen interaction energies of the tetrahedron.
"""

from hydro4b_coords.generate.discretized_distribution import DiscretizedDistribution
from hydro4b_coords.generate.distributions import exponential_decay_distribution

from nn_fourbody_potential import constants


def get_abinit_tetrahedron_distribution() -> DiscretizedDistribution:
    distrib = exponential_decay_distribution(
        n_terms=1024,
        x_min=2.2,
        x_max=10.0,
        coeff=constants.ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF,
        decay_rate=constants.ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON,
    )

    return distrib
