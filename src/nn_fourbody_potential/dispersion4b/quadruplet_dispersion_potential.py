"""
Calculate the quadruple-dipole (typo warning: quadruPLE, and *not* quadruPOLE")
dispersion interaction energy between four identical pointwise particles.

This code is based on equations (1) and (2), with N == 4, taken from:
    W. L. Bade. "Drude-Model calculation of dispersion forces. III. The fourth-order
    contribution", J. Chem Phys., 28 (1957).

This version of the function removes the pair and triplet components of the Bade
interaction potential, leaving only the quadruplet components.
"""

from __future__ import annotations

from nn_fourbody_potential.cartesian import Cartesian3D
from nn_fourbody_potential.dispersion4b._contributions import quadruplet_contribution
from nn_fourbody_potential.dispersion4b._magnitude_and_direction import distance_and_unit_vector


class QuadrupletDispersionPotential:
    """
    Calculate the quadruplet contribution to the dipole^4 dispersion
    interaction energy between four identical pointwise particles.
    """

    _coeff: float  # coefficient determining interaction strength

    def __init__(self, coeff: float) -> None:
        self._check_coeff_positive(coeff)
        self._coeff = coeff

    def __call__(self, p0: Cartesian3D, p1: Cartesian3D, p2: Cartesian3D, p3: Cartesian3D) -> float:
        # calculate the distances and unit vectors between each pair of points
        # i.e. describe the vector as an arrow with a magnitude and direction
        vec10 = distance_and_unit_vector(p1, p0)
        vec20 = distance_and_unit_vector(p2, p0)
        vec30 = distance_and_unit_vector(p3, p0)
        vec21 = distance_and_unit_vector(p2, p1)
        vec31 = distance_and_unit_vector(p3, p1)
        vec32 = distance_and_unit_vector(p3, p2)

        # calculate all the dot product combinations
        # NOTE: the equation in the paper has some of the indices swapped compared to
        #       what I use; however, the unit vectors in each term of the quadruplet
        #       contribution come in pairs, so I don't think swapping the direction
        #       of a unit vector matters.
        #
        #     : physically, this would be like saying "swapping the positions of any
        #       two of the four identical particles should not change the energy",
        #       which makes sense.
        total_energy = 2.0 * (
            quadruplet_contribution(vec30, vec32, vec21, vec10)
            + quadruplet_contribution(vec20, vec32, vec31, vec10)
            + quadruplet_contribution(vec20, vec21, vec31, vec30)
        )

        return -self._coeff * total_energy

    def _check_coeff_positive(self, coeff: float) -> None:
        if coeff <= 0.0:
            raise ValueError("The C12 coefficient for the interaction must be positive.\n" f"Entered: coeff = {coeff}")
