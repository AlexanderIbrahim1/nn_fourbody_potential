"""
This module contains functions to calculate the potential terms for the four-body
interaction potential.
"""

from __future__ import annotations

import itertools

from nn_fourbody_potential.cartesian import Cartesian3D
from nn_fourbody_potential.cartesian import dot_product
from nn_fourbody_potential.dispersion4b._magnitude_and_direction import distance_and_unit_vector


class DirectFourBodyDispersionPotential:
    """
    Calculate the dipole^4 dispersion interaction energy between four identical
    pointwise particles.
    """

    _c12_coeff: float  # coefficient determining interaction strength

    def __init__(self, c12_coeff: float) -> None:
        assert c12_coeff > 0.0

        self._c12_coeff = c12_coeff

    def __call__(self, p0: Cartesian3D, p1: Cartesian3D, p2: Cartesian3D, p3: Cartesian3D) -> float:
        total_energy = 0.0

        points = [p0, p1, p2, p3]
        for i, j, k, l in itertools.product(range(4), repeat=4):
            if i != j and j != k and k != l and l != i:
                pi = points[i]
                pj = points[j]
                pk = points[k]
                pl = points[l]

                total_energy += _quadruplet_contribution_from_points(pi, pj, pk, pl)

        return -self._c12_coeff * total_energy


def _quadruplet_contribution_from_points(
    p_i: Cartesian3D,
    p_j: Cartesian3D,
    p_k: Cartesian3D,
    p_l: Cartesian3D,
) -> float:
    """The four-particle contribution to the 4-body dispersion energy."""
    vec_ij = distance_and_unit_vector(p_i, p_j)
    vec_jk = distance_and_unit_vector(p_j, p_k)
    vec_kl = distance_and_unit_vector(p_k, p_l)
    vec_li = distance_and_unit_vector(p_l, p_i)

    # the distance term
    denom = (vec_ij.magnitude * vec_jk.magnitude * vec_kl.magnitude * vec_li.magnitude) ** 3

    prod_ijjk = dot_product(vec_ij.direction, vec_jk.direction)
    prod_ijkl = dot_product(vec_ij.direction, vec_kl.direction)
    prod_ijli = dot_product(vec_ij.direction, vec_li.direction)
    prod_jkkl = dot_product(vec_jk.direction, vec_kl.direction)
    prod_jkli = dot_product(vec_jk.direction, vec_li.direction)
    prod_klli = dot_product(vec_kl.direction, vec_li.direction)

    # begin with the constant contribution
    numer = -1.0

    # the squared pair prods
    numer += prod_ijjk**2 + prod_ijkl**2 + prod_ijli**2 + prod_jkkl**2 + prod_jkli**2 + prod_klli**2

    # the triplets
    numer -= 3.0 * (
        (prod_ijjk * prod_jkkl * prod_ijkl)
        + (prod_ijjk * prod_jkli * prod_ijli)
        + (prod_ijkl * prod_klli * prod_ijli)
        + (prod_jkkl * prod_klli * prod_jkli)
    )

    # the quadruplet term
    numer += 9.0 * (prod_ijjk * prod_jkkl * prod_klli * prod_ijli)

    return numer / denom
