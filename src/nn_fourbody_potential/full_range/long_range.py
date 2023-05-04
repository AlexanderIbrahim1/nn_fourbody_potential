"""
This module contains functions for making long-range adjustments to the four-body interaction
potential energy. The long-range energies converge to the Bade potential.
"""

from itertools import combinations
from typing import Optional

from cartesian.measure import distance

from dispersion4b.coefficients import c12_parahydrogen_midzuno_kihara
from dispersion4b.potential import FourBodyDispersionPotential
from dispersion4b.shortrange.attenuation import SilveraGoldmanAttenuation

from hydro4b_coords.generate.generate import six_side_lengths_to_cartesian

from nn_fourbody_potential.sidelength_distributions import SixSideLengths
from nn_fourbody_potential.sidelength_distributions import FourCartesianPoints

# TODO:
# - make changes such that, if the energy is all ab initio or all dispersion, return early


class LongRangeEnergyCorrector:
    def __init__(
        self,
        *,
        dispersion_potential: Optional[FourBodyDispersionPotential] = None,
        attenuation_function: Optional[SilveraGoldmanAttenuation] = None,
    ) -> None:
        if dispersion_potential is not None:
            self._dispersion_potential = dispersion_potential
        else:
            self._dispersion_potential = self._set_dispersion_potential()

        if attenuation_function is not None:
            self._attenuation_function = attenuation_function
        else:
            self._attenuation_function = self._set_attenuation_function()

    def apply_from_sidelengths(self, abinitio_energy: float, sidelengths: SixSideLengths) -> float:
        """
        Apply the long-range energy corrections, using the six side lengths of the four-body geometry
        to calculate the dispersion interaction energy.

        Calculating the four-body dispersion energy actually requires four points in 3D Cartesian
        space as an input. This is because it uses the six unit vectors between the four points in
        the calculation. This method uses the six side lengths to recreate the Cartesian3D instances.

        This is both slower (requires more calculations) and a bit more error-prone (the six side
        lengths may happen to not correspond to four points in 3D cartesian space) than performing
        the long-range correction using the four points in 3D Cartesian space directly.
        """
        four_points = six_side_lengths_to_cartesian(*sidelengths)
        sum_of_sidelengths = sum(sidelengths)
        return self._apply(abinitio_energy, four_points, sum_of_sidelengths)

    def apply_from_four_points(self, abinitio_energy: float, four_points: FourCartesianPoints) -> float:
        """
        Apply the long-range energy corrections, using the four points in 3D Cartesian space of the
        four-body geometry to calculate the dispersion interaction energy.
        """
        sum_of_sidelengths = sum([distance(p0, p1) for (p0, p1) in combinations(four_points, 2)])
        return self._apply(abinitio_energy, four_points, sum_of_sidelengths)

    def _apply(self, abinitio_energy: float, four_points: FourCartesianPoints, sum_of_sidelengths: float) -> float:
        """
        The helper function for applying the long-range correction to the ab initio four-body interaction
        potential energy.

        Args:
            abinitio_energy (float):
                the raw, uncorrected potential energy from the PES trained on the ab initio energies
            four_points (FourCartesianPoints):
                the four points in 3D cartesian space corresponding to the centres of mass of the four
                molecules
            sum_of_sidelengths (float):
                the sum of the six relative side lengths connecting the four points in 3D cartesian space

        Returns:
            float: the four-body interaction energy with the long-range corrections applied
        """
        dispersion_energy = self._dispersion_potential(*four_points)
        frac_dispersion = self._attenuation_function(sum_of_sidelengths)
        frac_abinitio = 1.0 - frac_dispersion

        return (dispersion_energy * frac_dispersion) + (abinitio_energy * frac_abinitio)

    def _set_dispersion_potential(self) -> FourBodyDispersionPotential:
        c12 = c12_parahydrogen_midzuno_kihara()
        return FourBodyDispersionPotential(c12)

    def _set_attenuation_function(self) -> SilveraGoldmanAttenuation:
        n_sidelengths = 6
        atten_r_cutoff = 3.3 * n_sidelengths
        atten_expon_coeff = 1.5 * n_sidelengths
        return SilveraGoldmanAttenuation(atten_r_cutoff, atten_expon_coeff)
