"""
This module contains components for learning about the properties of the side length and energy
data for a given rescaling.
"""

import numpy as np
import torch
from numpy.typing import NDArray

from nn_fourbody_potential.rescaling.forward_energy_rescaler import ForwardEnergyRescaler
from nn_fourbody_potential.rescaling.rescaling_limits import RescalingLimits
from nn_fourbody_potential.rescaling.rescaling_potential import RescalingPotential
from nn_fourbody_potential.rescaling.utils import forward_rescale_energies_


def print_reduced_statistics(
    _side_length_groups: NDArray,
    _energies: NDArray,
    rescaling_potential: RescalingPotential,
):
    energies = _energies.reshape(_energies.size, -1)

    side_length_groups = torch.from_numpy(_side_length_groups.astype(np.float32))
    energies = torch.from_numpy(energies.astype(np.float32))

    # use the RescalingPotential to modify the energies, without linearly mapping the outputs
    identity_res_limits = RescalingLimits(0.0, 1.0, 0.0, 1.0)
    dummy_fwd_rescaler = ForwardEnergyRescaler(rescaling_potential, identity_res_limits)
    forward_rescale_energies_(dummy_fwd_rescaler, side_length_groups, energies)  # modifies `energies`

    min_reduced_energy = energies.min().item()
    max_reduced_energy = energies.max().item()
    min_abs_reduced_energy = energies.abs().min().item()
    max_abs_reduced_energy = energies.abs().max().item()

    ratio_abs_reduced_energy = max_abs_reduced_energy / min_abs_reduced_energy

    print(f"min_reduced_energy       = {min_reduced_energy}")
    print(f"max_reduced_energy       = {max_reduced_energy}")
    print(f"min_abs_reduced_energy   = {min_abs_reduced_energy}")
    print(f"max_abs_reduced_energy   = {max_abs_reduced_energy}")
    print(f"ratio_abs_reduced_energy = {ratio_abs_reduced_energy}")
