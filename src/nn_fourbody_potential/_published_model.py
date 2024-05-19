"""
This module contains code for easily creating the published model for the four-body
interaction PES for parahydrogen.
"""

from pathlib import Path
from typing import Union

import torch

from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON
from nn_fourbody_potential.constants import N_FEATURES
from nn_fourbody_potential.constants import N_OUTPUTS
from nn_fourbody_potential.dispersion4b import b12_parahydrogen_midzuno_kihara
from nn_fourbody_potential.full_range import ExtrapolatedPotential
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import ReciprocalTransformer
from nn_fourbody_potential.transformations import MinimumPermutationTransformer
from nn_fourbody_potential.transformations import StandardizeTransformer
from nn_fourbody_potential import rescaling


_SIZE_TO_LAYERS: dict[str, list[int]] = {
    "size8": [8, 16, 16, 8],
    "size16": [16, 32, 32, 16],
    "size32": [32, 64, 64, 32],
    "size64": [64, 128, 128, 64],
}


_ACTIVATION_LABELS: list[str] = ["relu", "shiftedsoftplus"]


class ShiftedSoftplus(torch.nn.Softplus):
    def __init__(self, beta: int = 1, origin: float = 0.5, threshold: int = 20) -> None:
        super(ShiftedSoftplus, self).__init__(beta, threshold)
        self.origin = origin
        self.sp0 = torch.nn.functional.softplus(torch.zeros(1) + self.origin, self.beta, self.threshold).item()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(input + self.origin, self.beta, self.threshold) - self.sp0


def _published_feature_transformers() -> list[SixSideLengthsTransformer]:
    min_sidelen = 2.2

    return [
        ReciprocalTransformer(),
        StandardizeTransformer((0.0, 1.0 / min_sidelen), (0.0, 1.0)),
        MinimumPermutationTransformer(),
    ]


def _published_rescaling_function() -> rescaling.RescalingFunction:
    # constants chosen by hand, to drastically decrease the dynamic range of the output data;
    # after a certain amount of fiddling, it seems several choices of parameters would be good enough
    coeff = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF / 12.0
    expon = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON * 5.02
    disp_coeff = 0.125 * b12_parahydrogen_midzuno_kihara()

    return rescaling.RescalingFunction(coeff, expon, disp_coeff)


def _published_output_to_energy_rescaler() -> rescaling.ReverseEnergyRescaler:
    rescaling_function = _published_rescaling_function()
    rescaling_limits_to_energies = rescaling.RescalingLimits(
        from_left=-1.0, from_right=1.0, to_left=-3.2619903087615967, to_right=8.64592170715332
    )
    rescaler_output_to_energies = rescaling.ReverseEnergyRescaler(rescaling_function, rescaling_limits_to_energies)

    return rescaler_output_to_energies


def _published_load_model_weights(
    size_label: str, activation_label: str, model_filepath: Path, *, device: str
) -> RegressionMultilayerPerceptron:
    layers = _SIZE_TO_LAYERS[size_label]

    if activation_label == "relu":
        model = RegressionMultilayerPerceptron(N_FEATURES, N_OUTPUTS, layers)
    elif activation_label == "shiftedsoftplus":
        model = RegressionMultilayerPerceptron(
            N_FEATURES, N_OUTPUTS, layers, activation_function_factory=ShiftedSoftplus
        )
    else:
        raise RuntimeError("Impossible branch taken.")

    # TODO: remove this in the final version (temporary workaround)
    if activation_label == "shiftedsoftplus":
        model_state_dict = torch.load(model_filepath, map_location=torch.device(device))
        model_state_dict = model_state_dict["model_state_dict"]
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        return model

    model_state_dict = torch.load(model_filepath, map_location=torch.device(device))
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    return model


def load_potential(
    size_label: str, activation_label: str, model_filepath: Union[str, Path], *, device: str
) -> ExtrapolatedPotential:
    if size_label not in _SIZE_TO_LAYERS:
        raise RuntimeError(
            "Invalid size label found for the neural network model.\n"
            "The size label must be one of: 'size8', 'size16', 'size32', 'size64'"
        )

    activation_label = str.lower(activation_label)
    if activation_label not in _ACTIVATION_LABELS:
        raise RuntimeError(
            "Invalid activation label found for the neural network model.\n"
            "The activation label must be one of: 'relu', 'shiftedsoftplus'"
        )

    transformers = _published_feature_transformers()
    model = _published_load_model_weights(size_label, activation_label, Path(model_filepath), device=device)
    rescaler = _published_output_to_energy_rescaler()
    return ExtrapolatedPotential(model, transformers, rescaler, device=device)
