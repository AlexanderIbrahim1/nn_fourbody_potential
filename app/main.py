from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

import torch
from torchtyping import patch_typeguard

from nn_fourbody_potential.dataio import load_fourbody_training_data
from nn_fourbody_potential.dataset import PotentialDataset
from nn_fourbody_potential.fourbody_potential import create_fourbody_analytic_potential
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.models import TrainingParameters
from nn_fourbody_potential.sidelength_distributions import get_abinit_tetrahedron_distribution
from nn_fourbody_potential.sidelength_distributions import generate_training_data

from nn_fourbody_potential.modelio import ModelSaver
from nn_fourbody_potential.modelio import write_training_parameters
from nn_fourbody_potential.modelio import TestingErrorWriter

from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import apply_transformations_to_sidelengths_data

import model_info

patch_typeguard()

N_FEATURES = 6
N_OUTPUTS = 1


def add_dummy_dimension(data: np.ndarray[float]) -> np.ndarray[float, float]:
    """Takes an array of shape (n,), and returns an array of shape (n, 1)"""
    assert len(data.shape) == 1

    return data.reshape(data.size, -1)


def check_training_data_sizes(xdata, ydata) -> None:
    # this is a situation where the number of features and the number of outputs is unlikely
    # to change any time soon; it is currently more convenient to fix them as global constants;
    # this function performs a sanity check to make sure the loaded data has the correct dimensions
    n_samples_xdata, n_features = xdata.shape
    n_samples_ydata, n_outputs = ydata.shape
    assert n_samples_xdata == n_samples_ydata
    assert n_features == N_FEATURES
    assert n_outputs == N_OUTPUTS


def train_model(
    traindata_filename: Path,
    params: TrainingParameters,
    model: RegressionMultilayerPerceptron,
    modelpath: Path,
    model_saver: ModelSaver,
    save_every: Optional[int] = None,
) -> None:
    if save_every is None:
        save_every = params.total_epochs

    np.random.seed(params.seed)

    testing_error_writer = TestingErrorWriter(modelpath)

    sidelengths_train, energies_train = load_fourbody_training_data(traindata_filename)

    sidelengths_train = apply_transformations_to_sidelengths_data(
        sidelengths_train, params.transformations
    )
    energies_train = add_dummy_dimension(energies_train)

    x_train = torch.from_numpy(sidelengths_train.astype(np.float32))
    y_train = torch.from_numpy(energies_train.astype(np.float32))

    check_training_data_sizes(x_train, y_train)

    trainset = PotentialDataset(x_train, y_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=params.batch_size, shuffle=True, num_workers=1
    )

    loss_calculator = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )

    batch_to_datasize_ratio = params.batch_size / params.training_size

    for i_epoch in range(params.total_epochs):

        current_loss = 0.0
        for (x_batch, y_batch) in trainloader:
            y_batch_predicted = model(x_batch)

            loss = loss_calculator(y_batch, y_batch_predicted)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            current_loss += loss.item()

        epoch_loss = current_loss * batch_to_datasize_ratio

        testing_error_writer.append(i_epoch, epoch_loss)
        print(f"(epoch, loss) = ({i_epoch}, {epoch_loss:.4f})")

        if i_epoch % save_every == 0 and i_epoch != 0:
            model_saver.save_model(model, epoch=i_epoch)

    model_saver.save_model(model, epoch=params.total_epochs - 1)


def calculate_mse(y_test: torch.Tensor, y_estim: torch.Tensor) -> float:
    assert y_test.shape == y_estim.shape

    n_samples = y_test.shape[0]

    total_square_error = np.sum(
        [
            (y_test_val - y_estim_val) ** 2
            for (y_test_val, y_estim_val) in zip(y_test[:, 0], y_estim[:, 0])
        ]
    )

    return total_square_error / n_samples


def test_model(
    model: RegressionMultilayerPerceptron,
    modelfile: Path,
    n_samples: int,
    data_transforms: Optional[list[SixSideLengthsTransformer]] = None,
) -> None:
    model.load_state_dict(torch.load(modelfile))
    model.eval()

    distrib = get_abinit_tetrahedron_distribution(2.2, 4.5)
    potential = create_fourbody_analytic_potential()
    sidelengths_test, energies_test = generate_training_data(n_samples, distrib, potential)

    sidelengths_test = apply_transformations_to_sidelengths_data(sidelengths_test, data_transforms)
    energies_test = add_dummy_dimension(energies_test)

    with torch.no_grad():
        x_test = torch.from_numpy(sidelengths_test.astype(np.float32))
        y_test = torch.from_numpy(energies_test.astype(np.float32))

        y_estim = model(x_test)

        y_test.detach()
        y_estim.detach()
        mse = calculate_mse(y_test, y_estim)
        print(f"mse = {mse: .12f}")


if __name__ == "__main__":
    training_data_filepath = model_info.get_training_data_filepath()
    data_transforms = model_info.get_data_transforms()
    params = model_info.get_training_parameters(training_data_filepath, data_transforms)

    model = RegressionMultilayerPerceptron(N_FEATURES, N_OUTPUTS, params.layers)

    modelpath = model_info.get_path_to_model(params)
    if not modelpath.exists():
        modelpath.mkdir()

    training_parameters_filepath = model_info.get_training_parameters_filepath(params)
    write_training_parameters(training_parameters_filepath, params, overwrite=False)

    saved_models_dirpath = model_info.get_saved_models_dirpath(params)
    model_saver = ModelSaver(saved_models_dirpath)

    train_model(training_data_filepath, params, model, modelpath, model_saver, 20)
    test_model(
        model,
        model_saver.get_model_filename(params.total_epochs - 1),
        50000,
        data_transforms,
    )
