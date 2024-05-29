"""
This module contains an implementation of the algorithm to find the minimum permutation
of the tuple of six side lengths, assuming that the shortest and second shortest side
lengths are both unique.
"""

from nn_fourbody_potential.common_types import SixSideLengths
from nn_fourbody_potential.transformations._permutations import INDEX_SWAP_PERMUTATIONS
from nn_fourbody_potential.transformations._permutations import _sidelengths_permutation


SECOND_INDICES: list[tuple[int, int, int, int]] = [
    (1, 2, 3, 4),
    (0, 2, 3, 5),
    (0, 1, 4, 5),
    (0, 1, 4, 5),
    (0, 2, 3, 5),
    (1, 2, 3, 4),
]

NUMBER_OF_SECOND_INDICES: int = 4
ALL_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


def lookupshortest_minimum_permutation(sidelens: SixSideLengths) -> SixSideLengths:
    """
    Find the minimum permutation of the six side lengths, assuming that the shortest
    and second shortest side lengths are both unique.
    """
    i0 = _argmin_over(sidelens, ALL_INDICES)
    i1 = _argmin_over(sidelens, SECOND_INDICES[i0])
    i_perm = i1 + NUMBER_OF_SECOND_INDICES * i0

    permutation = INDEX_SWAP_PERMUTATIONS[i_perm]
    permuted_sidelens = _sidelengths_permutation(sidelens, permutation)

    return permuted_sidelens


def _argmin_over(sidelens: SixSideLengths, indices: tuple[int, ...]) -> int:
    """
    Find which index of `indices` refers to the smallest element of `sidelens`, among
    only the indices in `indices`.
    """
    i_min: int = 0
    minimum: float = sidelens[indices[0]]

    n_indices: int = len(indices)

    for i in range(1, n_indices):
        idx = indices[i]
        if sidelens[idx] < minimum:
            i_min = i
            minimum = sidelens[idx]

    return i_min
