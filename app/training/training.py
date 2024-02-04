"""
This module is for experimenting with the idea of training a neural network using
output energies that are rescaled by an analytic toy exponential decay potential.

The hope is that this rescaling brings all the energies into a narrower range of
values, thus making the training more effective.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch  # type: ignore

from dispersion4b.coefficients import c12_parahydrogen_midzuno_kihara
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_COEFF
from nn_fourbody_potential.constants import ABINIT_TETRAHEDRON_SHORTRANGE_DECAY_EXPON
from nn_fourbody_potential.dataio import load_fourbody_training_data
from nn_fourbody_potential.models import TrainingParameters
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import ReciprocalTransformer
from nn_fourbody_potential.transformations import MinimumPermutationTransformer
from nn_fourbody_potential.transformations import StandardizeTransformer
from nn_fourbody_potential import rescaling

import nn_fourbody_potential.modelio as modelio

from nn_fourbody_potential_data.data_paths import FILTERED_SPLIT_ABINITIO_TEST_DATA_DIRPATH
from nn_fourbody_potential_data.data_paths import FILTERED_SPLIT_ABINITIO_TRAIN_DATA_DIRPATH
from nn_fourbody_potential_data.data_paths import FILTERED_SPLIT_ABINITIO_TRAIN_NOHCP_DATA_DIRPATH
from nn_fourbody_potential_data.data_paths import FILTERED_SPLIT_ABINITIO_VALID_DATA_DIRPATH

import training_functions
from training_utils import N_FEATURES
from training_utils import N_OUTPUTS


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
        layers=[64, 128, 128, 64],
        learning_rate=2.0e-4,
        weight_decay=0.0,
        training_size=modelio.number_of_lines(data_filepath),
        total_epochs=20000,
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


def train_fourbody_model() -> None:
    training_data_filepath = FILTERED_SPLIT_ABINITIO_TRAIN_DATA_DIRPATH
    training_nohcp_data_filepath = FILTERED_SPLIT_ABINITIO_TRAIN_NOHCP_DATA_DIRPATH
    testing_data_filepath = FILTERED_SPLIT_ABINITIO_TEST_DATA_DIRPATH
    validation_data_filepath = FILTERED_SPLIT_ABINITIO_VALID_DATA_DIRPATH
    other_info = "_rescaling_model_large0"

    rescaling_potential = get_toy_decay_potential()
    transforms = get_data_transforms_flattening()

    params = get_training_parameters(training_data_filepath, transforms, other_info)
    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    model = RegressionMultilayerPerceptron(N_FEATURES, N_OUTPUTS, params.layers)

    # fmt: off
    side_length_groups_train, energies_train = load_fourbody_training_data(training_data_filepath)
    side_length_groups_train_nohcp, energies_train_nohcp = load_fourbody_training_data(training_nohcp_data_filepath)
    side_length_groups_test, energies_test = load_fourbody_training_data(testing_data_filepath)
    side_length_groups_valid, energies_valid = load_fourbody_training_data(validation_data_filepath)

    x_train, y_train, res_limits = rescaling.prepare_rescaled_data(side_length_groups_train, energies_train, transforms, rescaling_potential)
    x_train_nohcp, y_train_nohcp = rescaling.prepare_rescaled_data_with_rescaling_limits(side_length_groups_train_nohcp, energies_train_nohcp, transforms, rescaling_potential, res_limits)
    x_test, y_test = rescaling.prepare_rescaled_data_with_rescaling_limits(side_length_groups_test, energies_test, transforms, rescaling_potential, res_limits)
    x_valid, y_valid = rescaling.prepare_rescaled_data_with_rescaling_limits(side_length_groups_valid, energies_valid, transforms, rescaling_potential, res_limits)
    # fmt: on

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

    modelpath = modelio.get_path_to_model(params, Path.cwd())
    if not modelpath.exists():
        modelpath.mkdir()

    if not (params_filepath := modelio.get_training_parameters_filepath(params, Path.cwd())).exists():
        modelio.write_training_parameters(params_filepath, params, overwrite=False)

    saved_models_dirpath = modelio.get_saved_models_dirpath(params, Path.cwd())

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.99)

    training_functions.train_model(
        x_train,
        y_train,
        x_train_nohcp,
        y_train_nohcp,
        x_valid,
        y_valid,
        params,
        model,
        optimizer,
        scheduler,
        modelpath,
        save_every=50,
        # continue_training_from_epoch=15200,
    )

    last_model_filename = modelio.get_model_filename(saved_models_dirpath, params.total_epochs - 1)
    test_loss = training_functions.test_model(x_test, y_test, model, last_model_filename)
    print(f"test loss mse  = {test_loss}")
    print(f"test loss rmse = {np.sqrt(test_loss)}")


if __name__ == "__main__":
    train_fourbody_model()
