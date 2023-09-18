"""
This module contains the RescalingLimits class, which is used by other functions
that concern themselves with the rescaling of energies.
"""

import dataclasses


@dataclasses.dataclass
class RescalingLimits:
    from_left: float
    from_right: float
    to_left: float
    to_right: float


class LinearMap:
    def __init__(self, rl: RescalingLimits) -> None:
        self._slope = (rl.to_right - rl.to_left) / (rl.from_right - rl.from_left)
        self._intercept = rl.to_left - rl.from_left * self._slope

    def __call__(self, x: float) -> float:
        return x * self._slope + self._intercept


def invert_rescaling_limits(rl: RescalingLimits) -> RescalingLimits:
    return RescalingLimits(rl.to_left, rl.to_right, rl.from_left, rl.from_right)
