"""
This module contains components that transform the features of the data in some way.
This includes:
- rearranging the order of the features in ways that don't change the meaning of the
  features themselves
- performing a functional transformation on all the features
"""

from __future__ import annotations

import math
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable
from typing import Tuple

from hydro4b_coords.sidelength_swap import LessThanEpsilon
from hydro4b_coords.sidelength_swap import minimum_permutation

TransformedSideLengths = Tuple[float, float, float, float, float, float]
SixSideLengths = Tuple[float, float, float, float, float, float]
SixSideLengthsComparator = Callable[[SixSideLengths, SixSideLengths], bool]


class SixSideLengthsTransformer(ABC):
    @abstractmethod
    def __call__(self, sidelens: SixSideLengths) -> TransformedSideLengths:
        """Transform the six side lengths in some way, to give another 6-tuple of floats."""


@dataclass(frozen=True)
class ReciprocalTransformer(SixSideLengthsTransformer):
    """Transform each side length distance 'r' into its reciprocal '1/r'"""

    def __call__(self, sidelens: SixSideLengths) -> TransformedSideLengths:
        return tuple([1.0 / r for r in sidelens])


@dataclass(frozen=True)
class ExponentialDecayTransformer(SixSideLengthsTransformer):
    """Transform each side length distance 'r' into exp(-r/alpha)"""

    alpha: float

    def __post_init__(self) -> None:
        assert self.alpha > 0.0

    def __call__(self, sidelens: SixSideLengths) -> TransformedSideLengths:
        return tuple([math.exp(-r / self.alpha) for r in sidelens])


@dataclass(frozen=True)
class MinimumPermutationTransformer(SixSideLengthsTransformer):
    """
    Choose the permutation of indices (constrained by the four-body geometry) such that
    it would be lower than or equal to any other possible permutation of indices.
    """

    less_than_comparator: SixSideLengthsComparator = LessThanEpsilon(1.0e-4)

    def __call__(self, sidelens: SixSideLengths) -> TransformedSideLengths:
        return minimum_permutation(sidelens, self.less_than_comparator)

