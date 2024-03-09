"""
Find the smallest permutation of a 'SixSideLengths' instance, by exhaustively trying
out all 24 possible permutations.
"""

import copy
from typing import Callable

from nn_fourbody_potential.common_types import SixSideLengths
from nn_fourbody_potential.common_types import Permutation

# an exhaustive, hard-coded list of all the possible permutations of index swaps,
# except for the identity (0, 1, 2, 3, 4, 5), which would otherwise be the first
# element of the list
INDEX_SWAP_PERMUTATIONS = [
    (0, 2, 1, 4, 3, 5),
    (0, 3, 4, 1, 2, 5),
    (0, 4, 3, 2, 1, 5),
    (1, 0, 2, 3, 5, 4),
    (1, 2, 0, 5, 3, 4),
    (1, 3, 5, 0, 2, 4),
    (1, 5, 3, 2, 0, 4),
    (2, 0, 1, 4, 5, 3),
    (2, 1, 0, 5, 4, 3),
    (2, 4, 5, 0, 1, 3),
    (2, 5, 4, 1, 0, 3),
    (3, 0, 4, 1, 5, 2),
    (3, 1, 5, 0, 4, 2),
    (3, 4, 0, 5, 1, 2),
    (3, 5, 1, 4, 0, 2),
    (4, 0, 3, 2, 5, 1),
    (4, 2, 5, 0, 3, 1),
    (4, 3, 0, 5, 2, 1),
    (4, 5, 2, 3, 0, 1),
    (5, 1, 3, 2, 4, 0),
    (5, 2, 4, 1, 3, 0),
    (5, 3, 1, 4, 2, 0),
    (5, 4, 2, 3, 1, 0),
]


def minimum_permutation(
    sidelens: SixSideLengths,
    less_than_comparator: Callable[[SixSideLengths, SixSideLengths], bool],
) -> SixSideLengths:
    """
    Find which of the 24 allowed permutations of the entires of the six side lengths
    gives the lowest tuple. This is done by exhaustively trying out all the other 23
    permutations, and comparing them to the original.
    """
    current_sidelens = copy.deepcopy(sidelens)

    for perm in INDEX_SWAP_PERMUTATIONS:
        permuted_sidelens = _sidelengths_permutation(sidelens, perm)
        if less_than_comparator(permuted_sidelens, current_sidelens):
            current_sidelens = permuted_sidelens

    return current_sidelens


def _sidelengths_permutation(s: SixSideLengths, p: Permutation) -> SixSideLengths:
    return (s[p[0]], s[p[1]], s[p[2]], s[p[3]], s[p[4]], s[p[5]])
