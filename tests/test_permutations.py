import random

import pytest

from nn_fourbody_potential.transformations._approximate_permutations import approximate_minimum_permutation
from nn_fourbody_potential.transformations._approximate_permutations import _argmin_over
from nn_fourbody_potential.transformations._comparison import LessThanEpsilon
from nn_fourbody_potential.transformations._permutations import minimum_permutation


@pytest.fixture
def less_than_comparator() -> LessThanEpsilon:
    return LessThanEpsilon(1.0e-4)


@pytest.mark.parametrize(
    "expected_output, indices",
    [
        (3, (0, 1, 2, 3, 4, 5)),
        (2, (0, 1, 2, 5)),
    ],
)
def test_argmin_over(expected_output: int, indices: tuple[int, ...]) -> None:
    test_input = (10.0, 9.0, 8.0, 1.0, 8.0, 9.0)
    actual_output = _argmin_over(test_input, indices)

    assert expected_output == actual_output


class TestApproximatePermutation:
    def test_basic(self, less_than_comparator) -> None:
        """
        Check that the exact minimum permutation and the approximate minimum
        permutation match each other under the appropriate circumstances; i.e.
        when the shortest and second shortest side lengths are unique.
        """
        test_input = (10.0, 10.0, 10.0, 1.0, 10.0, 2.0)
        exact_output = minimum_permutation(test_input, less_than_comparator)
        approx_output = approximate_minimum_permutation(test_input)

        assert exact_output == pytest.approx(approx_output)

    def test_random(self, less_than_comparator) -> None:
        """
        Check that the exact and approximate results are the same for many random
        collections of six side lengths. Because the generated side lengths are
        floating-point numbers, the shortest and second shortest side lengths
        are almost guaranteed to be unique.
        """
        n_samples: int = 1000
        for _ in range(n_samples):
            test_input = tuple([random.uniform(1.0, 5.0) for _ in range(6)])
            exact_output = minimum_permutation(test_input, less_than_comparator)
            approx_output = approximate_minimum_permutation(test_input)

            assert exact_output == pytest.approx(approx_output)
