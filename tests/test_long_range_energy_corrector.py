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
        energy_from_side_lengths = long_range_corrector.apply_from_sidelengths(abinitio_energy, six_side_lengths)
        energy_from_points = long_range_corrector.apply_from_four_points(abinitio_energy, tetrahedron_points)

        assert energy_from_points == pytest.approx(energy_from_side_lengths)

    def test_no_dispersion(self) -> None:
        """By making the attenuation function return 0.0, we make sure the entire output
        comes from the ab initio portion."""

        arb_six_side_lengths = tuple([1.0] * 6)
        arb_abinitio_energy = 20.0

        zero_attenuation = lambda _: 0.0
        long_range_corrector = LongRangeEnergyCorrector(attenuation_function=zero_attenuation)
        corrected_energy = long_range_corrector.apply_from_sidelengths(arb_abinitio_energy, arb_six_side_lengths)

        assert corrected_energy == pytest.approx(arb_abinitio_energy)

    def test_no_abinitio(self) -> None:
        """By making the attenuation function return 1.0, we make sure the entire output
        comes from the dispersion portion."""

        arb_abinitio_energy = 20.0
        arb_tetrahedron_points = [
            Cartesian3D(0.000, 0.000, 0.000),
            Cartesian3D(1.000, 0.000, 0.000),
            Cartesian3D(0.500, 0.866, 0.000),
            Cartesian3D(0.500, 0.289, 0.816),
        ]

        c12 = c12_parahydrogen_midzuno_kihara()
        dispersion_potential = FourBodyDispersionPotential(c12)

        one_attenuation = lambda _: 1.0
        long_range_corrector = LongRangeEnergyCorrector(
            dispersion_potential=dispersion_potential, attenuation_function=one_attenuation
        )

        corrected_energy = long_range_corrector.apply_from_four_points(arb_abinitio_energy, arb_tetrahedron_points)
        dispersion_energy = dispersion_potential(*arb_tetrahedron_points)

        assert corrected_energy == pytest.approx(dispersion_energy)
