"""
This module is for experimenting with the idea of training a neural network using
output energies that are rescaled by an analytic toy exponential decay potential.

The hope is that this rescaling brings all the energies into a narrower range of
values, thus making the training more effective.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch  # type: ignore

from dispersion4b.coefficients import c12_parahydrogen_midzuno_kihara
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON
from nn_fourbody_potential.full_range import ExtrapolatedPotential
from nn_fourbody_potential.modelio import write_training_parameters
from nn_fourbody_potential.modelio.utils import get_model_filename
from nn_fourbody_potential.models import TrainingParameters
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import ReciprocalTransformer
from nn_fourbody_potential.transformations import MinimumPermutationTransformer
from nn_fourbody_potential.transformations import StandardizeTransformer
from nn_fourbody_potential import rescaling

import model_info
import training


def get_data_transforms_flattening() -> list[SixSideLengthsTransformer]:
    min_sidelen = 2.2

    return [
        ReciprocalTransformer(),
        StandardizeTransformer((0.0, 1.0 / min_sidelen), (0.0, 1.0)),
        MinimumPermutationTransformer(),
    ]


def get_training_parameters(
    data_filepath: Path,
    data_transforms: list[SixSideLengthsTransformer],
    other_info: str,
) -> TrainingParameters:
    return TrainingParameters(
        seed=42,
        layers=[32, 64, 64, 32],
        learning_rate=2.0e-4,
        weight_decay=0.0,
        training_size=model_info.number_of_lines(data_filepath),
        total_epochs=10000,
        batch_size=64,
        transformations=data_transforms,
        apply_batch_norm=False,
        other=other_info,
    )


def get_toy_decay_potential() -> rescaling.RescalingPotential:
    # constants chosen so that the ratio of the absolute values of the minimum and maximum reduced
    # energies is the lowest possible
    coeff = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF / 12.0
    expon = ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON * 5.02
    disp_coeff = 0.5 * c12_parahydrogen_midzuno_kihara()

    return rescaling.RescalingPotential(coeff, expon, disp_coeff)


def train_with_rescaling() -> None:
    training_data_filepath = Path("energy_separation", "data_splitting", "filtered_split_data", "train.dat")
    training_nohcp_data_filepath = Path("energy_separation", "data_splitting", "filtered_split_data", "train_nohcp.dat")
    testing_data_filepath = Path("energy_separation", "data_splitting", "filtered_split_data", "test.dat")
    validation_data_filepath = Path("energy_separation", "data_splitting", "filtered_split_data", "valid.dat")
    other_info = "_rescaling_model_filtered11"

    rescaling_potential = get_toy_decay_potential()
    transforms = get_data_transforms_flattening()

    params = get_training_parameters(training_data_filepath, transforms, other_info)
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    # fmt: off
    side_length_groups_train, energies_train = training.load_fourbody_training_data(training_data_filepath)
    side_length_groups_train_nohcp, energies_train_nohcp = training.load_fourbody_training_data(training_nohcp_data_filepath)
    side_length_groups_test, energies_test = training.load_fourbody_training_data(testing_data_filepath)
    side_length_groups_valid, energies_valid = training.load_fourbody_training_data(validation_data_filepath)
    # fmt: on

    x_train, y_train, res_limits = rescaling.prepare_rescaled_data(
        side_length_groups_train, energies_train, transforms, rescaling_potential
    )
    print(res_limits)

    x_train_nohcp, y_train_nohcp = rescaling.prepare_rescaled_data_with_rescaling_limits(
        side_length_groups_train_nohcp, energies_train_nohcp, transforms, rescaling_potential, res_limits
    )

    x_test, y_test = rescaling.prepare_rescaled_data_with_rescaling_limits(
        side_length_groups_test, energies_test, transforms, rescaling_potential, res_limits
    )

    x_valid, y_valid = rescaling.prepare_rescaled_data_with_rescaling_limits(
        side_length_groups_valid, energies_valid, transforms, rescaling_potential, res_limits
    )

    model = RegressionMultilayerPerceptron(training.N_FEATURES, training.N_OUTPUTS, params.layers)

    # moving everything to the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_train_nohcp = x_train_nohcp.to(device)
    y_train_nohcp = y_train_nohcp.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    x_valid = x_valid.to(device)
    y_valid = y_valid.to(device)
    model = model.to(device)

    modelpath = model_info.get_path_to_model(params)
    if not modelpath.exists():
        modelpath.mkdir()

    if not (params_filepath := model_info.get_training_parameters_filepath(params)).exists():
        write_training_parameters(params_filepath, params, overwrite=False)

    saved_models_dirpath = model_info.get_saved_models_dirpath(params)

    training.train_model(
        x_train,
        y_train,
        x_train_nohcp,
        y_train_nohcp,
        x_valid,
        y_valid,
        params,
        model,
        modelpath,
        save_every=50,
        # continue_training_from_epoch=4500,
    )

    last_model_filename = get_model_filename(saved_models_dirpath, params.total_epochs - 1)
    test_loss = training.test_model(x_test, y_test, model, last_model_filename)
    print(f"test loss mse = {test_loss}")
    print(f"test loss rmse = {np.sqrt(test_loss)}")


if __name__ == "__main__":
    train_with_rescaling()

    # OLD LIMITS
    # res_limits = rescaling.RescalingLimits(
    #     from_left=-3.2540090084075928, from_right=8.625899314880371, to_left=-1.0, to_right=1.0
    # )
    # res_limits = rescaling.RescalingLimits(
    #     from_left=-3.2619903087615967, from_right=8.64592170715332, to_left=-1.0, to_right=1.0
    # )

    # rescaling_potential = get_toy_decay_potential()
    # rev_rescaler = rescaling.ReverseEnergyRescaler(rescaling_potential, rescaling.invert_rescaling_limits(res_limits))

    # model_filename = Path(
    #     "/home/a68ibrah/research/four_body_interactions/nn_fourbody_potential/app/models/nnpes_rescaling_model4_layers64_128_128_64_lr_0.000200_datasize_12621/models/nnpes_19999.pth"
    #     "/home/a68ibrah/research/four_body_interactions/nn_fourbody_potential/app/models/nnpes_rescaling_model6_layers32_64_64_32_lr_0.000200_datasize_12621/models/nnpes_19999.pth"
    # )
    # checkpoint = torch.load(model_filename)
    # model = RegressionMultilayerPerceptron(
    #     training.N_FEATURES, training.N_OUTPUTS, [32, 64, 64, 32]
    # )  # [64, 128, 128, 64])
    # model.load_state_dict(checkpoint["model_state_dict"])

    # energy_model = rescaling.RescalingEnergyModel(model, rev_rescaler)

    # transforms = model_info.get_data_transforms_flattening()
    # extrapolated_potential = ExtrapolatedPotential(energy_model, transforms, pass_in_sidelengths_to_network=True)

    # sidelengths = np.linspace(1.9, 5.0, 256)
    # sidelength_groups = np.array([(s, s, s, s, s, s) for s in sidelengths]).reshape(-1, 6).astype(np.float32)
    # output_energies = extrapolated_potential.evaluate_batch(sidelength_groups)

    # _, ax = plt.subplots()
    # ax.plot(sidelengths, output_energies)
    # plt.show()
