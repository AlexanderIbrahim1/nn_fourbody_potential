"""
This module contains code for easily creating the published model for the four-body
interaction PES for parahydrogen.
"""

from pathlib import Path

import numpy as np
import torch

from dispersion4b.coefficients import c12_parahydrogen_midzuno_kihara
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON
from nn_fourbody_potential.dataio import load_fourbody_training_data
from nn_fourbody_potential.full_range import ExtrapolatedPotential
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import ReciprocalTransformer
from nn_fourbody_potential.transformations import MinimumPermutationTransformer
from nn_fourbody_potential.transformations import StandardizeTransformer
from nn_fourbody_potential.transformations import transform_sidelengths_data
from nn_fourbody_potential import rescaling

from nn_fourbody_potential_data.data_paths import FILTERED_SPLIT_ABINITIO_TEST_DATA_DIRPATH


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
    disp_coeff = 0.5 * c12_parahydrogen_midzuno_kihara()

    return rescaling.RescalingPotential(coeff, expon, disp_coeff)


def _published_output_to_energy_rescaler() -> rescaling.ReverseEnergyRescaler:
    rescaling_function = _published_rescaling_function()
    rescaling_limits_to_energies = rescaling.RescalingLimits(
        from_left=-1.0, from_right=1.0, to_left=-3.2619903087615967, to_right=8.64592170715332
    )
    rescaler_output_to_energies = rescaling.ReverseEnergyRescaler(rescaling_function, rescaling_limits_to_energies)

    return rescaler_output_to_energies


def _published_load_model_weights(size: str, model_filepath: Path) -> RegressionMultilayerPerceptron:
    n_features = 6
    n_outputs = 1
    layers = _SIZE_TO_LAYERS[size]
    model = RegressionMultilayerPerceptron(n_features, n_outputs, layers)

    model_state_dict = torch.load(model_filepath)
    model.load_state_dict(model_state_dict)

    return model


def load_model(size: str, model_filepath: Path) -> ExtrapolatedPotential:
    # TODO: add input sanitizing
    transformers = _published_feature_transformers()
    model = _published_load_model_weights()
    rescaler = _published_output_to_energy_rescaler()
    potential = ExtrapolatedPotential(model, transformers)


def main(size: int) -> None:
    test_filepath = FILTERED_SPLIT_ABINITIO_TEST_DATA_DIRPATH
    side_length_groups, test_energies = load_fourbody_training_data(test_filepath)

    model = _published_load_model(size)
    transformers = _published_feature_transformers()

    rescaling_function = _published_rescaling_function()
    rescaling_limits_to_energies = rescaling.RescalingLimits(
        from_left=-1.0, from_right=1.0, to_left=-3.2619903087615967, to_right=8.64592170715332
    )
    rescaler_output_to_energies = rescaling.ReverseEnergyRescaler(rescaling_function, rescaling_limits_to_energies)

    input_data = transform_sidelengths_data(side_length_groups, transformers)
    input_data = torch.from_numpy(input_data.astype(np.float32))

    with torch.no_grad():
        model.eval()

        output_data: torch.Tensor = model(input_data)
        output_data = output_data.reshape(-1)

        predicted_energies = np.array(
            [
                rescaler_output_to_energies(output.item(), side_length_group)
                for (output, side_length_group) in zip(output_data, side_length_groups)
            ]
        )

    output_filename = f"test_and_predicted_energies_{SIZE_TO_LABEL[size]}.dat"
    output_dirpath = Path(".", "test_and_predicted_energies")
    output_filepath = output_dirpath / output_filename

    output_data = np.vstack((test_energies, predicted_energies))
    output_data = np.transpose(output_data)

    np.savetxt(output_filepath, output_data)


if __name__ == "__main__":
    size = 64
    main(size)
