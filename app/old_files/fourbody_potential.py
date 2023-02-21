import numpy as np

from dispersion4b.coefficients import c12_parahydrogen_midzuno_kihara
from dispersion4b.potential import FourBodyDispersionPotential
from dispersion4b.shortrange.four_body_analytic_potential import (
    FourBodyAnalyticPotential,
)
from dispersion4b.shortrange.attenuation import SilveraGoldmanAttenuation
from dispersion4b.shortrange.distance_parameter_function import (
    DistanceParameterFunction,
)
from dispersion4b.shortrange.distance_parameter_function import sum_of_sidelengths
from dispersion4b.shortrange.short_range_functions import ExponentialDecay

ABINIT_TETRA_COEFF = 3.8163129e7
ABINIT_TETRA_EXPON = 9.2092762e-1


def create_four_body_analytic_potential() -> FourBodyAnalyticPotential:
    """
    Create a 'FourBodyAnalyticPotential' with:
        - the Midzuno-Kihara C12 coefficient
        - an exponential decay, using parameters fit to the ab initio tetrahedron energies
        - a Silvera-Goldman attenuation function, with parameters that don't really matter,
          as long as the exponential coefficient isn't too small
    """
    c12 = c12_parahydrogen_midzuno_kihara()
    disp_pot = FourBodyDispersionPotential(c12)

    decay_function = ExponentialDecay(ABINIT_TETRA_COEFF, ABINIT_TETRA_EXPON)
    short_range_pot = DistanceParameterFunction(decay_function, sum_of_sidelengths)

    # because we're using the 'sum_of_sidelengths()' function, we need to multiply the cutoff
    # distance by the number of sidelengths
    n_sidelengths = 6
    r_cutoff = 3.3 * n_sidelengths
    expon_coeff = 1.5
    atten_function = SilveraGoldmanAttenuation(r_cutoff, expon_coeff)
    short_long_atten = DistanceParameterFunction(atten_function, sum_of_sidelengths)

    return FourBodyAnalyticPotential(disp_pot, short_range_pot, short_long_atten)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from hydro4b_coords.geometries import MAP_GEOMETRY_TAG_TO_FUNCTION

    geometry_points = MAP_GEOMETRY_TAG_TO_FUNCTION["1"]

    pot = create_four_body_analytic_potential()
    sidelengths = np.linspace(2.2, 6.0, 256)
    energies = np.array([pot(geometry_points(sidelen)) for sidelen in sidelengths])

    fig, ax = plt.subplots()
    ax.plot(sidelengths, energies)
    plt.show()
