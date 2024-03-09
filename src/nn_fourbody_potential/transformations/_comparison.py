"""
This module contains functions for comparing two instances of 'SixSideLengths'.

The reason for creating special functions for handling comparisons, is that
sometimes two instances *should* be equal, but one ends up larger than the other
only due to floating-point errors. This happens frequently enough, especially when
comparing the sidelengths of geometries from a frozen lattice, that special functions
should be used.
"""

from dataclasses import dataclass

from nn_fourbody_potential.common_types import SixSideLengths


@dataclass(frozen=True)
class LessThanRounded:
    """
    Round the elements of both tuples to within a certain number of decimal places
    before comparing them.
    """

    n_round: int

    def __post_init__(self) -> None:
        assert self.n_round >= 1

    def __call__(self, s0: SixSideLengths, s1: SixSideLengths) -> bool:
        s0_rounded = tuple([round(val, self.n_round) for val in s0])
        s1_rounded = tuple([round(val, self.n_round) for val in s1])

        return s0_rounded < s1_rounded


@dataclass(frozen=True)
class LessThanEpsilon:
    """
    Usual element-wise comparison of two tuples, except two floating-point values are
    considered equal if they are within 'self.epsilon' of each other.
    """

    epsilon: float

    def __post_init__(self) -> None:
        assert self.epsilon > 0.0

    def __call__(self, s0: SixSideLengths, s1: SixSideLengths) -> bool:
        for sidelen0, sidelen1 in zip(s0, s1):
            if abs(sidelen0 - sidelen1) > self.epsilon:
                return sidelen0 < sidelen1

        return False
