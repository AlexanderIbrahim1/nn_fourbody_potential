import numpy as np

# fmt: off
from dispersion4b.coefficients import c12_parahydrogen_midzuno_kihara

from dispersion4b.potential import FourBodyDispersionPotential

from dispersion4b.shortrange.four_body_analytic_potential import FourBodyAnalyticPotential
from dispersion4b.shortrange.attenuation import SilveraGoldmanAttenuation
from dispersion4b.shortrange.distance_parameter_function import DistanceParameterFunction
from dispersion4b.shortrange.distance_parameter_function import sum_of_sidelengths
from dispersion4b.shortrange.short_range_functions import ExponentialDecay
# fmt: on

from nn_fourbody_potential import constants


def create_fourbody_analytic_potential() -> FourBodyAnalyticPotential:
    """
    Create a 'FourBodyAnalyticPotential' with:
        - the Midzuno-Kihara C12 coefficient
        - an exponential decay, using parameters fit to the ab initio tetrahedron energies
        - a Silvera-Goldman attenuation function, with parameters that don't really matter,
          as long as the exponential coefficient isn't too small

    The process of creating the potential is a bit convoluted, because it was designed to
    be able to swap in and out different dispersion potentials, short-range potentials,
    attenuation functions, and even the parameter that the functions depend on.
    """
    c12 = c12_parahydrogen_midzuno_kihara()
    disp_pot = FourBodyDispersionPotential(c12)

    coeff = constants.ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF
    expon = constants.ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON
    decay_function = ExponentialDecay(coeff, expon)
    short_range_pot = DistanceParameterFunction(decay_function, sum_of_sidelengths)

    # because we're using the 'sum_of_sidelengths()' function, we need to multiply the cutoff
    # distance by the number of sidelengths
    n_sidelengths = constants.NUMBER_OF_SIDELENGTHS_FOURBODY
    atten_r_cutoff = constants.ABINIT_TETRAHEDRON_ATTENUATION_R_CUTOFF * n_sidelengths
    atten_expon_coeff = constants.ABINIT_TETRAHEDRON_ATTENUATION_EXPON * n_sidelengths

    atten_function = SilveraGoldmanAttenuation(atten_r_cutoff, atten_expon_coeff)
    short_long_atten = DistanceParameterFunction(atten_function, sum_of_sidelengths)

    return FourBodyAnalyticPotential(disp_pot, short_range_pot, short_long_atten)
