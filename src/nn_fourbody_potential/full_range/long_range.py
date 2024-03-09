"""
This module contains functions for making long-range adjustments to the four-body interaction
potential energy. The long-range energies converge to the Bade potential.
"""

import statistics
from typing import Optional

from nn_fourbody_potential.cartesian import relative_pair_distances
from nn_fourbody_potential.cartesian import six_side_lengths_to_cartesian
from nn_fourbody_potential.dispersion4b import b12_parahydrogen_avtz_approx
from nn_fourbody_potential.dispersion4b import QuadrupletDispersionPotential
from nn_fourbody_potential.common_types import SixSideLengths
from nn_fourbody_potential.common_types import FourCartesianPoints

from nn_fourbody_potential.full_range.constants import END_LONG_RANGE_CUTOFF
from nn_fourbody_potential.full_range.constants import START_LONG_RANGE_CUTOFF
from nn_fourbody_potential.full_range.utils import smooth_01_transition


class LongRangeEnergyCorrector:
    def __init__(
        self,
        *,
        dispersion_potential: Optional[QuadrupletDispersionPotential] = None,
    ) -> None:
        if dispersion_potential is not None:
            self._dispersion_potential = dispersion_potential
        else:
            self._dispersion_potential = _get_dispersion_potential()

    def dispersion_from_sidelengths(self, sidelengths: SixSideLengths) -> float:
        four_points = six_side_lengths_to_cartesian(*sidelengths)
        return self._dispersion_potential(*four_points)

    def dispersion_from_four_points(self, four_points: FourCartesianPoints) -> float:
        return self._dispersion_potential(*four_points)

    def mixed_from_sidelengths(self, abinitio_energy: float, sidelengths: SixSideLengths) -> float:
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
        average_sidelength = statistics.mean(sidelengths)
        return self._mixed(abinitio_energy, four_points, average_sidelength)

    def mixed_from_four_points(self, abinitio_energy: float, four_points: FourCartesianPoints) -> float:
        """
        Apply the long-range energy corrections, using the four points in 3D Cartesian space of the
        four-body geometry to calculate the dispersion interaction energy.
        """
        sidelengths = relative_pair_distances(four_points)
        average_sidelength = statistics.mean(sidelengths)
        return self._mixed(abinitio_energy, four_points, average_sidelength)

    def _mixed(self, abinitio_energy: float, four_points: FourCartesianPoints, average_sidelength: float) -> float:
        """
        The helper function for applying the long-range correction to the ab initio four-body interaction
        potential energy.

        Args:
            abinitio_energy (float):
                the raw, uncorrected potential energy from the PES trained on the ab initio energies
            four_points (FourCartesianPoints):
                the four points in 3D cartesian space corresponding to the centres of mass of the four
                molecules
            average_sidelength (float):
                the average of the six relative side lengths connecting the four points in 3D cartesian space

        Returns:
            float: the four-body interaction energy with the long-range corrections applied
        """
        dispersion_energy = self._dispersion_potential(*four_points)
        frac_dispersion = smooth_01_transition(average_sidelength, START_LONG_RANGE_CUTOFF, END_LONG_RANGE_CUTOFF)
        frac_abinitio = 1.0 - frac_dispersion

        return (dispersion_energy * frac_dispersion) + (abinitio_energy * frac_abinitio)


def _get_dispersion_potential() -> QuadrupletDispersionPotential:
    b12 = b12_parahydrogen_avtz_approx()
    return QuadrupletDispersionPotential(b12)
