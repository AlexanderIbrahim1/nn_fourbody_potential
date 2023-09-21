from __future__ import annotations

from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader

from nn_fourbody_potential.dataio import load_fourbody_training_data
from nn_fourbody_potential.dataset import PotentialDataset
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.models import TrainingParameters
from nn_fourbody_potential.transformations import SixSideLengthsTransformer

from nn_fourbody_potential.modelio import CheckpointSaver
from nn_fourbody_potential.modelio import CheckpointLoader
from nn_fourbody_potential.modelio import ErrorWriter

from nn_fourbody_potential.transformations import transform_sidelengths_data

import model_info

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


def prepared_data(
    data_filepath: Path, transformers: Sequence[SixSideLengthsTransformer]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the sidelengths and energies from 'data_filepath', apply the transformations,
    and turn the sidelengths and energies into torch tensors.
    """
    sidelengths, energies = load_fourbody_training_data(data_filepath)
    sidelengths = transform_sidelengths_data(sidelengths, transformers)
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
    default_model: RegressionMultilayerPerceptron,
    modelpath: Path,
    *,
    save_every: int,
    continue_training_from_epoch: Optional[int] = None,
) -> None:
    np.random.seed(params.seed)

    saved_models_dirpath = model_info.get_saved_models_dirpath(params)
    checkpoint_saver = CheckpointSaver(saved_models_dirpath)

    loss_calculator = torch.nn.MSELoss()
    default_optimizer = torch.optim.Adam(
        default_model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )

    if continue_training_from_epoch is not None:
        checkpoint_loader = CheckpointLoader(saved_models_dirpath)
        checkpoint_data = checkpoint_loader.load_checkpoint(
            continue_training_from_epoch, default_model, default_optimizer
        )
        epoch_start = checkpoint_data.epoch
        model = checkpoint_data.model
        optimizer = checkpoint_data.optimizer
        error_file_mode = "a"
    else:
        epoch_start = 0
        model = default_model
        optimizer = default_optimizer
        error_file_mode = "w"

    training_error_writer = ErrorWriter(modelpath, "training_error_vs_epoch.dat", mode=error_file_mode)
    validation_error_writer = ErrorWriter(modelpath, "validation_error_vs_epoch.dat", mode=error_file_mode)

    trainset = PotentialDataset(x_train, y_train)
    trainloader = DataLoader(trainset, batch_size=params.batch_size, num_workers=0, shuffle=True)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.99)

    for i_epoch in range(epoch_start, params.total_epochs):
        for x_batch, y_batch in trainloader:
            y_batch_predicted = model(x_batch)

            loss: torch.Tensor = loss_calculator(y_batch, y_batch_predicted)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

        if i_epoch >= 100:
            scheduler.step()

        epoch_training_loss = evaluate_model_loss(model, loss_calculator, x_train, y_train)
        training_error_writer.append(i_epoch, epoch_training_loss)

        epoch_validation_loss = evaluate_model_loss(model, loss_calculator, x_valid, y_valid)
        validation_error_writer.append(i_epoch, epoch_validation_loss)

        print(f"(epoch, training_loss)   = ({i_epoch}, {epoch_training_loss:.8f})")
        print(f"(epoch, validation_loss) = ({i_epoch}, {epoch_validation_loss:.8f})")

        if i_epoch % save_every == 0 and i_epoch != 0:
            checkpoint_saver.save_checkpoint(model=model, optimizer=optimizer, epoch=i_epoch)

    checkpoint_saver.save_checkpoint(model=model, optimizer=optimizer, epoch=params.total_epochs - 1)


def test_model(
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    model: RegressionMultilayerPerceptron,
    modelfile: Path,
) -> None:
    checkpoint = torch.load(modelfile)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_calculator = torch.nn.MSELoss()

    testing_loss = evaluate_model_loss(model, loss_calculator, x_test, y_test)

    return testing_loss
