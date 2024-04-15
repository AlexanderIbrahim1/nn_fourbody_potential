"""
This module contains functions to perform common operations on Cartesian3D points.
"""

import math
from itertools import combinations
from typing import Sequence

from nn_fourbody_potential.cartesian.cartesian3d import Cartesian3D


def distance_squared(point0: Cartesian3D, point1: Cartesian3D) -> float:
    return (point0.x - point1.x) ** 2 + (point0.y - point1.y) ** 2 + (point0.z - point1.z) ** 2


def distance(point0: Cartesian3D, point1: Cartesian3D) -> float:
    return math.sqrt(distance_squared(point0, point1))


def norm_squared(point: Cartesian3D) -> float:
    return point.x**2 + point.y**2 + point.z**2


def norm(point: Cartesian3D) -> float:
    return math.sqrt(norm_squared(point))


def dot_product(p0: Cartesian3D, p1: Cartesian3D) -> float:
    return p0.x * p1.x + p0.y * p1.y + p0.z * p1.z


def cross_product(pa: Cartesian3D, pb: Cartesian3D) -> Cartesian3D:
    result0 = pa.y * pb.z - pa.z * pb.y
    result1 = pa.z * pb.x - pa.x * pb.z
    result2 = pa.x * pb.y - pa.y * pb.x

    return Cartesian3D(result0, result1, result2)


def linear_combination(points: Sequence[Cartesian3D], coeffs: Sequence[float]) -> Cartesian3D:
    assert len(points) == len(coeffs) > 0

    sum_point = coeffs[0] * points[0]
    for point, coeff in zip(points[1:], coeffs[1:]):
        sum_point = sum_point + coeff * point

    return sum_point


def approx_eq(point0: Cartesian3D, point1: Cartesian3D, tolerance: float = 1.0e-6) -> bool:
    """Check if two points are approximately equal to each other, within a given tolerance."""
    dist_sq = distance_squared(point0, point1)
    tol_sq = tolerance**2

    return dist_sq < tol_sq


def centroid(points: Sequence[Cartesian3D]) -> Cartesian3D:
    """
    Calculate the centroid position of a sequence of points.
    This is like the centre of mass, but we give each position the same weight.
    """
    n_points = len(points)
    assert n_points >= 1

    sum_point = points[0]
    for point in points[1:]:
        sum_point += point

    return sum_point / n_points


def relative_pair_distances(points: Sequence[Cartesian3D]) -> list[float]:
    """
    Calculate the distances between all pairs of points in free, non-periodic space.

    If fewer than two points are present, the returned list is empty.
    """
    return [distance(p0, p1) for (p0, p1) in combinations(points, 2)]


def six_side_lengths_to_cartesian(
    r01: float,
    r02: float,
    r03: float,
    r12: float,
    r13: float,
    r23: float,
    sqrt_tolerance: float = 1.0e-6,
) -> tuple[Cartesian3D, Cartesian3D, Cartesian3D, Cartesian3D]:
    """
    This function uses the 6 relative pair distances between the four points to
    recover four Cartesian points in 3D space.

    The four points returned satisfy the following properties:
     - point0 is at the origin
     - point1 lies on the positive x-axis
     - point2 satisfies (y >= 0, z == 0)
     - point3 satisfies (z >= 0)

    The 'pairdistance_to_cartesian()' and the 'cartesian_to_pairdistance()' functions
    are not inverses. We lose information when using the 'cartesian_to_pairdistance()'
    transformation. The relative pair distances only have 6 degrees of freedom (DOF) of
    information to work with, but the four Cartesian points have 12 DOF.

    The three DOF describing the centre of mass position of the four-body system are
    lost when converting from relative pair distances to Cartesian coordinates. The
    three DOF describing the orientation in space of the four-body system are also lost.
    """
    cos_theta102 = (r01**2 + r02**2 - r12**2) / (2.0 * r01 * r02)

    x2 = r02 * cos_theta102
    x3 = (r03**2 - r13**2 + r01**2) / (2.0 * r01)

    y2_inner = r02**2 - x2**2
    if y2_inner > sqrt_tolerance:
        y2 = math.sqrt(y2_inner)
        y3 = (r03**2 - r23**2 + r02**2 - 2.0 * x2 * x3) / (2.0 * y2)
    else:
        y2 = 0.0
        y3 = 0.0

    z2 = 0.0

    z3_inner = r03**2 - x3**2 - y3**2
    if z3_inner > sqrt_tolerance:
        z3 = math.sqrt(z3_inner)
    else:
        z3 = 0.0

    point0 = Cartesian3D(0.0, 0.0, 0.0)
    point1 = Cartesian3D(r01, 0.0, 0.0)
    point2 = Cartesian3D(x2, y2, z2)
    point3 = Cartesian3D(x3, y3, z3)

    return (point0, point1, point2, point3)
