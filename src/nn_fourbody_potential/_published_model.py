"""
This module contains code for easily creating the published model for the four-body
interaction PES for parahydrogen.
"""

from pathlib import Path

import torch

from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON
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


def _published_feature_transformers() -> list[SixSideLengthsTransformer]:
    min_sidelen = 2.2

    return [
        ReciprocalTransformer(),
        StandardizeTransformer((0.0, 1.0 / min_sidelen), (0.0, 1.0)),
        MinimumPermutationTransformer(),
    ]


def _published_rescaling_function() -> rescaling.RescalingPotential:
    # constants chosen so that the ratio of the absolute values of the minimum and maximum reduced
    # energies is the lowest possible
    coeff = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF / 12.0
    expon = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON * 5.02
    disp_coeff = 0.125 * b12_parahydrogen_midzuno_kihara()

    return rescaling.RescalingPotential(coeff, expon, disp_coeff)


def _published_output_to_energy_rescaler() -> rescaling.ReverseEnergyRescaler:
    rescaling_function = _published_rescaling_function()
    rescaling_limits_to_energies = rescaling.RescalingLimits(
        from_left=-1.0, from_right=1.0, to_left=-3.2619903087615967, to_right=8.64592170715332
    )
    rescaler_output_to_energies = rescaling.ReverseEnergyRescaler(rescaling_function, rescaling_limits_to_energies)

    return rescaler_output_to_energies


def _published_load_model_weights(size_label: str, model_filepath: Path) -> RegressionMultilayerPerceptron:
    n_features = 6
    n_outputs = 1
    layers = _SIZE_TO_LAYERS[size_label]
    model = RegressionMultilayerPerceptron(n_features, n_outputs, layers)

    model_state_dict = torch.load(model_filepath)
    model.load_state_dict(model_state_dict)

    return model


def load_potential(size_label: str, model_filepath: Path) -> ExtrapolatedPotential:
    # TODO: add input sanitizing
    transformers = _published_feature_transformers()
    model = _published_load_model_weights(size_label, model_filepath)
    rescaler = _published_output_to_energy_rescaler()
    return ExtrapolatedPotential(model, transformers, rescaler)
