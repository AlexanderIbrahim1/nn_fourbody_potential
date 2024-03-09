"""
Calculate the quadruple-dipole (typo warning: quadruPLE, and *not* quadruPOLE")
dispersion interaction energy between four identical pointwise particles.

This code is based on equations (1) and (2), with N == 4, taken from:
    W. L. Bade. "Drude-Model calculation of dispersion forces. III. The fourth-order
    contribution", J. Chem Phys., 28 (1957).

NOTE: are there better names for this?
    : there was no confusion with "triple-dipole" for the three-body interaction, but
    : "quadruple-dipole" looks too similar to "quadrupole-dipole".
    :
    : "dipole-dipole-dipole-dipole"?
    : "fourth-order dipole"?
"""

from __future__ import annotations

from nn_fourbody_potential.cartesian import Cartesian3D
from nn_fourbody_potential.dispersion4b._contributions import pair_contribution
from nn_fourbody_potential.dispersion4b._contributions import triplet_contribution
from nn_fourbody_potential.dispersion4b._contributions import quadruplet_contribution
from nn_fourbody_potential.dispersion4b._magnitude_and_direction import distance_and_unit_vector


class FourBodyDispersionPotential:
    """
    Calculate the dipole^4 dispersion interaction energy between four identical
    pointwise particles.
    """

    _c12_coeff: float  # coefficient determining interaction strength

    def __init__(self, c12_coeff: float) -> None:
        self._check_c12_coeff_positive(c12_coeff)

        self._c12_coeff = c12_coeff

    def __call__(self, p0: Cartesian3D, p1: Cartesian3D, p2: Cartesian3D, p3: Cartesian3D) -> float:
        # calculate the distances and unit vectors between each pair of points
        # i.e. describe the vector as an arrow with a magnitude and direction
        vec10 = distance_and_unit_vector(p1, p0)
        vec20 = distance_and_unit_vector(p2, p0)
        vec30 = distance_and_unit_vector(p3, p0)
        vec21 = distance_and_unit_vector(p2, p1)
        vec31 = distance_and_unit_vector(p3, p1)
        vec32 = distance_and_unit_vector(p3, p2)

        total_energy = 0.0

        # the pair contribution
        for vec in [vec10, vec20, vec30, vec21, vec31, vec32]:
            total_energy += pair_contribution(vec.magnitude)

        # the triplet contribution
        total_energy += triplet_contribution(vec10, vec20)
        total_energy += triplet_contribution(vec10, vec30)
        total_energy += triplet_contribution(vec20, vec30)
        total_energy += triplet_contribution(vec10, vec21)
        total_energy += triplet_contribution(vec10, vec31)
        total_energy += triplet_contribution(vec21, vec31)
        total_energy += triplet_contribution(vec20, vec21)
        total_energy += triplet_contribution(vec20, vec32)
        total_energy += triplet_contribution(vec21, vec32)
        total_energy += triplet_contribution(vec30, vec31)
        total_energy += triplet_contribution(vec30, vec32)
        total_energy += triplet_contribution(vec31, vec32)

        # calculate all the dot product combinations
        # NOTE: the equation in the paper has some of the indices swapped compared to
        #       what I use; however, the unit vectors in each term of the quadruplet
        #       contribution come in pairs, so I don't think swapping the direction
        #       of a unit vector matters.
        #
        #     : physically, this would be like saying "swapping the positions of any
        #       two of the four identical particles should not change the energy",
        #       which makes sense.
        #
        #     : I should unit test it anyways
        total_energy += 2.0 * quadruplet_contribution(vec30, vec32, vec21, vec10)
        total_energy += 2.0 * quadruplet_contribution(vec20, vec32, vec31, vec10)
        total_energy += 2.0 * quadruplet_contribution(vec20, vec21, vec31, vec30)

        return -self._c12_coeff * total_energy

    def _check_c12_coeff_positive(self, c12_coeff: float) -> None:
        if c12_coeff <= 0.0:
            raise ValueError(
                "The C12 coefficient for the interaction must be positive.\n" f"Entered: c12_coeff = {c12_coeff}"
            )
