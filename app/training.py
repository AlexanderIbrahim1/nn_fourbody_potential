from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader

from nn_fourbody_potential.dataio import load_fourbody_training_data
from nn_fourbody_potential.dataset import PotentialDataset
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.models import TrainingParameters

from nn_fourbody_potential.modelio import ModelSaver
from nn_fourbody_potential.modelio import ErrorWriter

from nn_fourbody_potential.transformations import transform_sidelengths_data

N_FEATURES = 6
N_OUTPUTS = 1


def add_dummy_dimension(data: np.ndarray[float]) -> np.ndarray[float, float]:
    """Takes an array of shape (n,), and returns an array of shape (n, 1)"""
    assert len(data.shape) == 1

    return data.reshape(data.size, -1)


def check_data_sizes(
    xdata: torch.Tensor, ydata: torch.Tensor, n_features_expected: int, n_outputs_expected: int
) -> None:
    # this is a situation where the number of features and the number of outputs is unlikely
    # to change any time soon; it is currently more convenient to fix them as global constants;
    # this function performs a sanity check to make sure the loaded data has the correct dimensions
    n_samples_xdata, n_features = xdata.shape
    n_samples_ydata, n_outputs = ydata.shape
    assert n_samples_xdata == n_samples_ydata
    assert n_features == n_features_expected
    assert n_outputs == n_outputs_expected


def prepared_data(data_filepath: Path, params: TrainingParameters) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the sidelengths and energies from 'data_filepath', apply the transformations,
    and turn the sidelengths and energies into torch tensors.
    """
    sidelengths, energies = load_fourbody_training_data(data_filepath)
    sidelengths = transform_sidelengths_data(sidelengths, params.transformations)
    energies = add_dummy_dimension(energies)

    x = torch.from_numpy(sidelengths.astype(np.float32))
    y = torch.from_numpy(energies.astype(np.float32))

    check_data_sizes(x, y, N_FEATURES, N_OUTPUTS)

    return x, y


def evaluate_model_loss(
    model: RegressionMultilayerPerceptron,
    loss_calculator: torch.nn.MSELoss,
    input_data: torch.Tensor,
    output_data: torch.Tensor,
) -> float:
    """Compare the true labelled 'output_data' to the outputs of the 'model' when given the 'input_data'."""
    model.eval()

    with torch.no_grad():
        output_data_predicted = model(input_data)
        loss = loss_calculator(output_data, output_data_predicted)

    model.train()
    return loss.item()


def train_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_valid: torch.Tensor,
    y_valid: torch.Tensor,
    params: TrainingParameters,
    model: RegressionMultilayerPerceptron,
    modelpath: Path,
    model_saver: ModelSaver,
    *,
    save_every: int,
) -> None:
    np.random.seed(params.seed)

    training_error_writer = ErrorWriter(modelpath, "training_error_vs_epoch.dat")
    validation_error_writer = ErrorWriter(modelpath, "validation_error_vs_epoch.dat")

    trainset = PotentialDataset(x_train, y_train)
    trainloader = DataLoader(trainset, batch_size=params.batch_size, shuffle=True, num_workers=1)

    loss_calculator = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params.learning_rate
    )  # , weight_decay=params.weight_decay) # TODO: turn weight decay back on

    for i_epoch in range(params.total_epochs):
        for x_batch, y_batch in trainloader:
            y_batch_predicted = model(x_batch)

            loss = loss_calculator(y_batch, y_batch_predicted)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        epoch_training_loss = evaluate_model_loss(model, loss_calculator, x_train, y_train)
        training_error_writer.append(i_epoch, epoch_training_loss)

        epoch_validation_loss = evaluate_model_loss(model, loss_calculator, x_valid, y_valid)
        validation_error_writer.append(i_epoch, epoch_validation_loss)

        print(f"(epoch, training_loss) = ({i_epoch}, {epoch_training_loss:.4f})")
        print(f"(epoch, validation_loss) = ({i_epoch}, {epoch_validation_loss:.4f})")

        if i_epoch % save_every == 0 and i_epoch != 0:
            model_saver.save_model(model, epoch=i_epoch)

    model_saver.save_model(model, epoch=params.total_epochs - 1)


def test_model(
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    model: RegressionMultilayerPerceptron,
    modelfile: Path,
) -> None:
    model.load_state_dict(torch.load(modelfile))
    loss_calculator = torch.nn.MSELoss()

    testing_loss = evaluate_model_loss(model, loss_calculator, x_test, y_test)

    return testing_loss
