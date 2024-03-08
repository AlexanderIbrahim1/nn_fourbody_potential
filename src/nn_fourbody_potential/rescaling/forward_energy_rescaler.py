"""
This module contains the ForwardEnergyRescaler, which is responsible for taking the
ab initio energies and mapping them to a range of values that is easier for the neural
networks to work with.
"""

from nn_fourbody_potential.rescaling.rescaling_limits import RescalingLimits
from nn_fourbody_potential.rescaling.rescaling_limits import LinearMap
from nn_fourbody_potential.rescaling.rescaling_potential import RescalingPotential

from nn_fourbody_potential.sidelength_distributions import SixSideLengths


class ForwardEnergyRescaler:
    def __init__(
        self,
        res_potential: RescalingPotential,
        forward_res_limits: RescalingLimits,
    ) -> None:
        self._res_potential = res_potential
        self._lin_map = LinearMap(forward_res_limits)

    def __call__(self, energy: float, six_side_lengths: SixSideLengths) -> float:
        rescale_value = self._res_potential(*six_side_lengths)

        reduced_energy = energy / rescale_value
        rescaled_energy = self._lin_map(reduced_energy)

        return rescaled_energy
