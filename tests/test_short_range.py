import pytest

from typing import Sequence

from nn_fourbody_potential.full_range.constants import LOWER_SHORT_DISTANCE
from nn_fourbody_potential.full_range.constants import SHORT_RANGE_SCALING_STEP
from nn_fourbody_potential.full_range.short_range import prepare_short_range_extrapolation_data


class Test_prepare_extrapolation_data:
    def test_basic_functionality(self) -> None:
        sample = (1.0, 2.0, 2.0, 2.0, 2.0, 2.0)
        scaling_step = 0.05
        short_range_cutoff = 2.0

        extrap_sample, extrap_dist_info = prepare_short_range_extrapolation_data(
            sample, scaling_step, short_range_cutoff=short_range_cutoff
        )

        expect_sample_min = (2.00, 4.0, 4.0, 4.0, 4.0, 4.0)
        expect_sample_mid = (2.05, 4.1, 4.1, 4.1, 4.1, 4.1)
        expect_r_lower = short_range_cutoff
        expect_r_upper = short_range_cutoff + scaling_step
        expect_r_short_range = 1.0

        assert _is_sequence_pytest_approx_equal(extrap_sample.lower, expect_sample_min)
        assert _is_sequence_pytest_approx_equal(extrap_sample.upper, expect_sample_mid)
        assert extrap_dist_info.r_short_range == pytest.approx(expect_r_short_range)
        assert extrap_dist_info.r_lower == pytest.approx(expect_r_lower)
        assert extrap_dist_info.r_upper == pytest.approx(expect_r_upper)

    @pytest.mark.parametrize("invalid_scaling_step", [0.0, -0.05])
    def test_raises_nonpositive_scaling_step(self, invalid_scaling_step: float) -> None:
        arb_sample = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            prepare_short_range_extrapolation_data(
                arb_sample, invalid_scaling_step, short_range_cutoff=LOWER_SHORT_DISTANCE, flag_skip_checks=False
            )

    def test_raises_sidelengths_to_large(self) -> None:
        too_large_sample = tuple([LOWER_SHORT_DISTANCE * 1.05] * 6)
        with pytest.raises(ValueError):
            prepare_short_range_extrapolation_data(
                too_large_sample, SHORT_RANGE_SCALING_STEP, LOWER_SHORT_DISTANCE, flag_skip_checks=False
            )

    @pytest.mark.parametrize("invalid_sidelength", [0.0, -0.05])
    def test_raises_nonpositive_sidelength(self, invalid_sidelength: float) -> None:
        sample_with_invalid_sidelength = (invalid_sidelength, 1.0, 1.0, 1.0, 1.0, 1.0)
        with pytest.raises(ValueError):
            prepare_short_range_extrapolation_data(
                sample_with_invalid_sidelength,
                SHORT_RANGE_SCALING_STEP,
                LOWER_SHORT_DISTANCE,
                flag_skip_checks=False,
            )


def _is_sequence_pytest_approx_equal(seq0: Sequence[float], seq1: Sequence[float]) -> bool:
    return all([s0 == pytest.approx(s1) for (s0, s1) in zip(seq0, seq1)])
