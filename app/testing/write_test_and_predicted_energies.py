"""
This script is used to compare the energies calculated by the trained models
against the test energies.
"""

from pathlib import Path

import numpy as np
import torch

from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON
from nn_fourbody_potential.constants import N_FEATURES
from nn_fourbody_potential.constants import N_OUTPUTS
from nn_fourbody_potential.dispersion4b import b12_parahydrogen_midzuno_kihara
from nn_fourbody_potential.dataio import load_fourbody_training_data
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import ReciprocalTransformer
from nn_fourbody_potential.transformations import MinimumPermutationTransformer
from nn_fourbody_potential.transformations import StandardizeTransformer
from nn_fourbody_potential.transformations import transform_sidelengths_data
from nn_fourbody_potential import rescaling

from nn_fourbody_potential_data.data_paths import FILTERED_SPLIT_ABINITIO_TEST_DATA_DIRPATH

SIZE_TO_LABEL: dict[int, str] = {
    8: "8_16_16_8",
    16: "16_32_32_16",
    32: "32_64_64_32",
    64: "64_128_128_64",
}

SIZE_TO_LAYERS: dict[int, list[int]] = {
    8: [8, 16, 16, 8],
    16: [16, 32, 32, 16],
    32: [32, 64, 64, 32],
    64: [64, 128, 128, 64],
}


def feature_transformers() -> list[SixSideLengthsTransformer]:
    min_sidelen = 2.2

    return [
        ReciprocalTransformer(),
        StandardizeTransformer((0.0, 1.0 / min_sidelen), (0.0, 1.0)),
        MinimumPermutationTransformer(),
    ]


def get_rescaling_function() -> rescaling.RescalingFunction:
    # constants chosen so that the ratio of the absolute values of the minimum and maximum reduced
    # energies is the lowest possible
    coeff = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF / 12.0
    expon = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON * 5.02
    disp_coeff = 0.125 * b12_parahydrogen_midzuno_kihara()

    return rescaling.RescalingFunction(coeff, expon, disp_coeff)


def model_filepath(size: int) -> Path:
    model_dirpath = Path("/home/a68ibrah/research/four_body_interactions/nn_fourbody_potential/models")
    model_filename = f"fourbodypara_{SIZE_TO_LABEL[size]}.pth"

    return model_dirpath / model_filename


def load_model(size: int) -> RegressionMultilayerPerceptron:
    layers = SIZE_TO_LAYERS[size]
    model = RegressionMultilayerPerceptron(N_FEATURES, N_OUTPUTS, layers)

    filepath = model_filepath(size)
    model_state_dict = torch.load(filepath)
    model.load_state_dict(model_state_dict)

    return model


def main(size: int) -> None:
    test_filepath = FILTERED_SPLIT_ABINITIO_TEST_DATA_DIRPATH
    side_length_groups, test_energies = load_fourbody_training_data(test_filepath)

    model = load_model(size)
    transformers = feature_transformers()

    rescaling_function = get_rescaling_function()
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
