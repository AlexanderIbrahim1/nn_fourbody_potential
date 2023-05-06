import pytest

from typing import Sequence

from nn_fourbody_potential.sidelength_distributions import SixSideLengths
from nn_fourbody_potential.full_range.constants import SHORT_RANGE_DISTANCE_CUTOFF
from nn_fourbody_potential.full_range.short_range import create_extrapolation_samples


class Test_create_extrapolation_samples:
    def test_basic_functionality(self) -> None:
        sample = (1.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        scaling_step = 0.05
        short_range_cutoff = 2.0

        extrapolation_samples = create_extrapolation_samples(
            sample, scaling_step, short_range_cutoff=short_range_cutoff
        )

        expect_sample_min = (2.00, 4.0, 4.0, 4.0, 4.0, 4.0)
        expect_sample_mid = (2.05, 4.1, 4.1, 4.1, 4.1, 4.1)

        assert _is_sequence_pytest_approx_equal(extrapolation_samples.sidelengths_lower, expect_sample_min)
        assert _is_sequence_pytest_approx_equal(extrapolation_samples.sidelengths_upper, expect_sample_mid)

    @pytest.mark.parametrize("invalid_scaling_step", [0.0, -0.05])
    def test_raises_nonpositive_scaling_step(self, invalid_scaling_step: float) -> None:
        arb_sample = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            create_extrapolation_samples(arb_sample, invalid_scaling_step)

    def test_raises_sidelengths_to_large(self) -> None:
        too_large_sample = tuple([SHORT_RANGE_DISTANCE_CUTOFF * 1.05] * 6)
        with pytest.raises(ValueError):
            create_extrapolation_samples(too_large_sample)

    @pytest.mark.parametrize("invalid_sidelength", [0.0, -0.05])
    def test_raises_nonpositive_sidelength(self, invalid_sidelength: float) -> None:
        invalid_sample = (invalid_sidelength, 1.0, 1.0, 1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            create_extrapolation_samples(invalid_sample)


def _is_sequence_pytest_approx_equal(seq0: Sequence[float], seq1: Sequence[float]) -> bool:
    return all([s0 == pytest.approx(s1) for (s0, s1) in zip(seq0, seq1)])
