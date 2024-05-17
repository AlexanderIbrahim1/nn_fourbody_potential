"""
This module contains functions for creating the callables that generate the
side lengths of samples.
"""

import numpy as np

from common_types import PseudoRNG
from common_types import SixSideLengths
from common_types import SideLengthGenerator


def make_prng(minimum: float, maximum: float) -> PseudoRNG:
    def inner() -> float:
        return np.random.uniform(minimum, maximum)

    return inner


def make_extra0_side_length_generator(short: PseudoRNG, long: PseudoRNG) -> SideLengthGenerator:
    def inner() -> SixSideLengths:
        return (short(), long(), long(), long(), long(), long())

    return inner


def make_extra1_side_length_generator(short: PseudoRNG, long: PseudoRNG) -> SideLengthGenerator:
    def inner() -> SixSideLengths:
        return (short(), short(), long(), long(), long(), long())

    return inner


def make_extra2_side_length_generator(short: PseudoRNG, long: PseudoRNG) -> SideLengthGenerator:
    def inner() -> SixSideLengths:
        return (short(), long(), long(), long(), long(), short())

    return inner


def make_extra3_side_length_generator(short: PseudoRNG, long: PseudoRNG) -> SideLengthGenerator:
    def inner() -> SixSideLengths:
        return (short(), short(), short(), long(), long(), long())

    return inner


def make_extra4_side_length_generator(short: PseudoRNG, long: PseudoRNG) -> SideLengthGenerator:
    def inner() -> SixSideLengths:
        return (short(), short(), long(), short(), long(), long())

    return inner


def make_extra5_side_length_generator(short: PseudoRNG, long: PseudoRNG) -> SideLengthGenerator:
    def inner() -> SixSideLengths:
        return (short(), short(), long(), long(), short(), long())

    return inner


def make_extra6_side_length_generator(short: PseudoRNG, long: PseudoRNG) -> SideLengthGenerator:
    def inner() -> SixSideLengths:
        return (short(), short(), short(), short(), long(), long())

    return inner


MAP_CATEGORY_TO_SIDE_LENGTH_GENERATOR = [
    make_extra0_side_length_generator,
    make_extra1_side_length_generator,
    make_extra2_side_length_generator,
    make_extra3_side_length_generator,
    make_extra4_side_length_generator,
    make_extra5_side_length_generator,
    make_extra6_side_length_generator,
]
