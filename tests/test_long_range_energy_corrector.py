import pytest

from cartesian import Cartesian3D
from cartesian.operations import relative_pair_distances

from dispersion4b.coefficients import c12_parahydrogen_midzuno_kihara
from dispersion4b.potential import FourBodyDispersionPotential

from nn_fourbody_potential.full_range.long_range import LongRangeEnergyCorrector


class TestLongRangeEnergyCorrector:
    def test_same_result(self) -> None:
        """Make sure that calculating the corrected four-body interaction energy using the
        six side lengths, or using the four Cartesian3D instances, gives the same result."""

        scaling_factor = 3.3

        tetrahedron_points = [
            scaling_factor * Cartesian3D(0.000, 0.000, 0.000),
            scaling_factor * Cartesian3D(1.000, 0.000, 0.000),
            scaling_factor * Cartesian3D(0.500, 0.866, 0.000),
            scaling_factor * Cartesian3D(0.500, 0.289, 0.816),
        ]

        six_side_lengths = relative_pair_distances(tetrahedron_points)

        abinitio_energy = 20.0  # arbitrary value

        long_range_corrector = LongRangeEnergyCorrector()
        mixed_energy_from_side_lengths = long_range_corrector.mixed_from_sidelengths(abinitio_energy, six_side_lengths)
        mixed_energy_from_points = long_range_corrector.mixed_from_four_points(abinitio_energy, tetrahedron_points)
        disp_energy_from_side_lengths = long_range_corrector.dispersion_from_sidelengths(six_side_lengths)
        disp_energy_from_points = long_range_corrector.dispersion_from_four_points(tetrahedron_points)

        assert mixed_energy_from_points == pytest.approx(mixed_energy_from_side_lengths)
        assert disp_energy_from_points == pytest.approx(disp_energy_from_side_lengths)