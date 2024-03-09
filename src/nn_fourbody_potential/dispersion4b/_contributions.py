"""
This module contains the functions that make up the different contributions to the
full four-body interaction potential energy surface.
"""

from __future__ import annotations

from nn_fourbody_potential.cartesian import dot_product
from nn_fourbody_potential.dispersion4b._magnitude_and_direction import MagnitudeAndDirection


def pair_contribution(distance: float) -> float:
    """The two-particle contribution to the 4-body dispersion energy."""
    return 1.0 / (distance**12)


def triplet_contribution(
    vec_ij: MagnitudeAndDirection,
    vec_jk: MagnitudeAndDirection,
) -> float:
    """The three-particle contribution to the 4-body dispersion energy."""
    cosine_ijk = dot_product(vec_ij.direction, vec_jk.direction)

    numer = 1.0 + cosine_ijk**2
    denom = (vec_ij.magnitude * vec_jk.magnitude) ** 6

    return numer / denom


def quadruplet_contribution(
    vec_ij: MagnitudeAndDirection,
    vec_jk: MagnitudeAndDirection,
    vec_kl: MagnitudeAndDirection,
    vec_li: MagnitudeAndDirection,
) -> float:
    """The four-particle contribution to the 4-body dispersion energy."""

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
