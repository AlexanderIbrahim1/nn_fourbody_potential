"""
This module contains the ReverseEnergyRescaler, which is responsible for taking the
output of the neural networks (trained on the rescaled energies) and mapping them back
to the actual ab initio energies.
"""

from nn_fourbody_potential.rescaling.rescaling_limits import RescalingLimits
from nn_fourbody_potential.rescaling.rescaling_limits import LinearMap
from nn_fourbody_potential.rescaling.rescaling_potential import RescalingPotential

from nn_fourbody_potential.common_types import SixSideLengths


class ReverseEnergyRescaler:
    def __init__(
        self,
        res_potential: RescalingPotential,
        reverse_res_limits: RescalingLimits,
    ) -> None:
        self._res_potential = res_potential
        self._lin_map = LinearMap(reverse_res_limits)

    def __call__(self, rescaled_energy: float, six_side_lengths: SixSideLengths) -> float:
        rescale_value = self._res_potential(*six_side_lengths)

        reduced_energy = self._lin_map(rescaled_energy)
        energy = rescale_value * reduced_energy

        return energy
