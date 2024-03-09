"""
This module contains components that transform the features of the data in some way.
This includes:
- rearranging the order of the features in ways that don't change the meaning of the
  features themselves
- performing a functional transformation on all the features
"""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import Callable

from nn_fourbody_potential.common_types import SixSideLengths
from nn_fourbody_potential.common_types import SixSideLengthsComparator
from nn_fourbody_potential.common_types import TransformedSideLengths

from nn_fourbody_potential.transformations._permutations import minimum_permutation
from nn_fourbody_potential.transformations._comparison import LessThanEpsilon


class SixSideLengthsTransformer(abc.ABC):
    @abc.abstractmethod
    def __call__(self, sidelens: SixSideLengths) -> TransformedSideLengths:
        """Transform the six side lengths in some way, to give another 6-tuple of floats."""


@dataclasses.dataclass(frozen=True)
class ReciprocalTransformer(SixSideLengthsTransformer):
    """Transform each side length distance 'r' into its reciprocal '1/r'"""

    def __call__(self, sidelens: SixSideLengths) -> TransformedSideLengths:
        return tuple([1.0 / r for r in sidelens])  # type: ignore


@dataclasses.dataclass(frozen=True)
class ExponentialDecayTransformer(SixSideLengthsTransformer):
    """Transform each side length distance 'r' into exp(-r/alpha)"""

    alpha: float

    def __post_init__(self) -> None:
        assert self.alpha > 0.0

    def __call__(self, sidelens: SixSideLengths) -> TransformedSideLengths:
        return tuple([math.exp(-r / self.alpha) for r in sidelens])  # type: ignore


@dataclasses.dataclass(frozen=True)
class MinimumPermutationTransformer(SixSideLengthsTransformer):
    """
    Choose the permutation of indices (constrained by the four-body geometry) such that
    it would be lower than or equal to any other possible permutation of indices.
    """

    less_than_comparator: SixSideLengthsComparator = LessThanEpsilon(1.0e-4)

    def __call__(self, sidelens: SixSideLengths) -> TransformedSideLengths:
        return minimum_permutation(sidelens, self.less_than_comparator)  # type: ignore


@dataclasses.dataclass(frozen=True)
class StandardizeTransformer(SixSideLengthsTransformer):
    """
    Map all six values linearly from [a, b] to [c, d].
    """

    init_pair: tuple[float, float]
    final_pair: tuple[float, float]
    linear_func: Callable[[float], float] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        assert self.init_pair[0] < self.init_pair[1]
        assert self.final_pair[0] < self.final_pair[1]

        def map_init_pair_to_final_pair(x: float) -> float:
            a = self.init_pair[0]
            b = self.init_pair[1]
            c = self.final_pair[0]
            d = self.final_pair[1]

            # subtract a              ; map (a, b) to (0, b-a)
            # multiply by (d-c)/(b-a) ; map (0, b-a) to (0, d-c)
            # add c                   ; map (0, d-c) to (c, d)

            slope = (d - c) / (b - a)
            return (x - a) * slope + c

        object.__setattr__(self, "linear_func", map_init_pair_to_final_pair)

    def __call__(self, sidelens: SixSideLengths) -> TransformedSideLengths:
        return tuple([self.linear_func(s) for s in sidelens])  # type: ignore
