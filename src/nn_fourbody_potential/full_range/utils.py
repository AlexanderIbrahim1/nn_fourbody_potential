"""
This module contains various functions that are reused in different locations in
the 'full_range' section.
"""

import math


def smooth_01_transition(x: float, x_min: float, x_max: float) -> float:
    """Smoothly transition from 0.0 at or before x <= x_min, to 1.0 at or after x >= x_max."""
    assert x_min < x_max
    if x <= x_min:
        return 0.0
    elif x >= x_max:
        return 1.0
    else:
        k = (x - x_min) / (x_max - x_min)
        return 0.5 * (1.0 - math.cos(math.pi * k))


def is_different_sign(a: float, b: float) -> bool:
    return a * b <= 0.0
