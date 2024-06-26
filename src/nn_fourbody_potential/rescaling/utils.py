from typing import Sequence

import numpy as np
from numpy.typing import NDArray
import torch

from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import transform_sidelengths_data

from nn_fourbody_potential.rescaling.forward_energy_rescaler import ForwardEnergyRescaler
from nn_fourbody_potential.rescaling.rescaling_limits import invert_rescaling_limits
from nn_fourbody_potential.rescaling.rescaling_limits import LinearMap
from nn_fourbody_potential.rescaling.rescaling_limits import RescalingLimits
from nn_fourbody_potential.rescaling.rescaling_function import RescalingFunction
from nn_fourbody_potential.rescaling.reverse_energy_rescaler import ReverseEnergyRescaler


def forward_and_reverse_energy_rescalers(
    res_potential: RescalingFunction, forward_res_limits: RescalingLimits
) -> tuple[ForwardEnergyRescaler, ReverseEnergyRescaler]:
    reverse_res_limits = invert_rescaling_limits(forward_res_limits)

    forward_rescaler = ForwardEnergyRescaler(res_potential, forward_res_limits)
    reverse_rescaler = ReverseEnergyRescaler(res_potential, reverse_res_limits)

    return forward_rescaler, reverse_rescaler


def forward_rescale_energies_(
    fwd_rescaler: ForwardEnergyRescaler,
    side_length_groups: torch.Tensor,
    energies_to_rescale: torch.Tensor,
) -> None:
    """Rescale all the energies in `energies_to_rescale` in-place."""
    for i, (eng, sidelengths) in enumerate(zip(energies_to_rescale, side_length_groups)):
        energies_to_rescale[i] = fwd_rescaler(eng.item(), tuple(sidelengths.tolist()))


def reverse_rescale_energies_(
    rev_rescaler: ReverseEnergyRescaler,
    side_length_groups: torch.Tensor,
    energies_to_reverse_rescale: torch.Tensor,
) -> None:
    """Reverse-rescale all the energies in `energies_to_reverse_rescale` in-place."""
    for i, (res_eng, sidelengths) in enumerate(zip(energies_to_reverse_rescale, side_length_groups)):
        energies_to_reverse_rescale[i] = rev_rescaler(res_eng.item(), tuple(sidelengths.tolist()))


def prepare_rescaled_data(
    _side_length_groups: NDArray,
    _energies: NDArray,
    transformers: Sequence[SixSideLengthsTransformer],
    res_potential: RescalingFunction,
    target_rescaling_limits: tuple[float, float] = (-1.0, 1.0),
    *,
    omit_final_rescaling_step: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, RescalingLimits]:
    """
    _side_length_groups:
        a numpy array of shape (N, 6) representing the 6-tuples of relative pair distances between
        the four points
    _energies:
        a numpy array of shape (N,) or (N, 1) representing the interaction potential energy of the
        four points
    """
    trans_side_length_groups = transform_sidelengths_data(_side_length_groups, transformers)
    energies = _energies.reshape(_energies.size, -1)

    side_length_groups = torch.from_numpy(_side_length_groups.astype(np.float32))
    trans_side_length_groups = torch.from_numpy(trans_side_length_groups.astype(np.float32))
    energies = torch.from_numpy(energies.astype(np.float32))

    # use the RescalingPotential to modify the energies, without linearly mapping the outputs
    identity_res_limits = RescalingLimits(0.0, 1.0, 0.0, 1.0)
    dummy_fwd_rescaler = ForwardEnergyRescaler(res_potential, identity_res_limits)
    forward_rescale_energies_(dummy_fwd_rescaler, side_length_groups, energies)  # modifies `energies`

    if omit_final_rescaling_step:
        return (trans_side_length_groups, energies, identity_res_limits)
    else:
        # use these reduced energies to get the proper rescaling limits
        fwd_rescaling_limits = _forward_rescaling_limits(energies, target_rescaling_limits)

        # apply the proper rescaling to turn the reduced energies to the fully rescaled energies
        fwd_linear_map = LinearMap(fwd_rescaling_limits)
        energies.apply_(fwd_linear_map)

        return (trans_side_length_groups, energies, fwd_rescaling_limits)


def prepare_rescaled_data_with_rescaling_limits(
    _side_length_groups: NDArray,
    _energies: NDArray,
    transformers: Sequence[SixSideLengthsTransformer],
    res_potential: RescalingFunction,
    forward_res_limits: RescalingLimits,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Similar to `prepare_rescaled_data()`, but the RescalingLimits instance is passed in and used,
    instead of created from the data.
    """
    trans_side_length_groups = transform_sidelengths_data(_side_length_groups, transformers)
    energies = _energies.reshape(_energies.size, -1)

    side_length_groups = torch.from_numpy(_side_length_groups.astype(np.float32))
    trans_side_length_groups = torch.from_numpy(trans_side_length_groups.astype(np.float32))
    energies = torch.from_numpy(energies.astype(np.float32))

    forward_rescaler = ForwardEnergyRescaler(res_potential, forward_res_limits)
    forward_rescale_energies_(forward_rescaler, side_length_groups, energies)

    return (trans_side_length_groups, energies)


def _forward_rescaling_limits(
    reduced_energies: torch.Tensor, target_rescaling_limits: tuple[float, float]
) -> RescalingLimits:
    min_red_energy = reduced_energies.min().item()
    max_red_energy = reduced_energies.max().item()
    min_target = target_rescaling_limits[0]
    max_target = target_rescaling_limits[1]
    return RescalingLimits(min_red_energy, max_red_energy, min_target, max_target)
