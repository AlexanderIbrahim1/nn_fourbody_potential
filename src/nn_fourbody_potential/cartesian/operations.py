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
