"""
This module contains functions to help generate samples.
"""

import math
from typing import Optional

from nn_fourbody_potential.cartesian import Cartesian3D

from common_types import FourCartesianPoints


def maybe_six_side_lengths_to_cartesian(
    r01: float,
    r02: float,
    r03: float,
    r12: float,
    r13: float,
    r23: float,
    tolerance: float = 1.0e-6,
) -> Optional[FourCartesianPoints]:
    """
    This function uses the 6 relative pair distances between the four points to
    recover four Cartesian points in 3D space.

    The four points returned satisfy the following properties:
     - point0 is at the origin
     - point1 lies on the positive x-axis
     - point2 satisfies (y >= 0, z == 0)
     - point3 satisfies (z >= 0)

    If the six side lengths do not correspond to a valid four-body geometry, this function
    returns None.
    """
    x1 = r01
    x2 = (r01**2 + r02**2 - r12**2) / (2.0 * r01)
    x3 = (r03**2 - r13**2 + r01**2) / (2.0 * r01)

    y2_inner = r02**2 - x2**2

    # an indication that the geometry is invalid
    if y2_inner < -tolerance:
        return None

    y3_numerator = r03**2 - r23**2 + r02**2 - 2.0 * x2 * x3
    y2, y3 = _calculate_y2_and_y3(y2_inner, y3_numerator, tolerance)

    z3_inner = r03**2 - x3**2 - y3**2

    # an indication that the geometry is invalid
    if z3_inner < -tolerance:
        return None

    z3 = math.sqrt(max(0.0, z3_inner))

    point0 = Cartesian3D(0.0, 0.0, 0.0)
    point1 = Cartesian3D(x1, 0.0, 0.0)
    point2 = Cartesian3D(x2, y2, 0.0)
    point3 = Cartesian3D(x3, y3, z3)

    return (point0, point1, point2, point3)


def _calculate_y2_and_y3(y2_inner: float, y3_numerator: float, tolerance: float) -> tuple[float, float]:
    if y2_inner > tolerance:
        y2 = math.sqrt(y2_inner)
        y3 = y3_numerator / (2.0 * y2)
    else:
        y2 = 0.0
        y3 = 0.0

    return y2, y3


if __name__ == "__main__":
    side_lengths = [1.0 for _ in range(6)]
    points = maybe_six_side_lengths_to_cartesian(*side_lengths)

    # side_lengths = [1.0 for _ in range(4)] + [10.0, 10.0]
    # points = maybe_six_side_lengths_to_cartesian(*side_lengths)

    print(points)
