from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from nn_fourbody_potential.dataset import PotentialDataset
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.models import TrainingParameters

import training_io
import training_state
import training_utils


def update_training_state_data(
    state_data: training_state.TrainingStateData,
    saved_models_dirpath: Path,
    continue_training_from_epoch: Optional[int],
) -> training_state.TrainingStateData:
    if continue_training_from_epoch is not None:
        model_, optimizer_, scheduler_, _ = state_data.unpack()
        state_dict = training_state.load_training_state_dict(saved_models_dirpath, continue_training_from_epoch)
        new_state_data = training_state.apply_training_state_dict(state_dict, model_, optimizer_, scheduler_)
        new_state_data.epoch_start += 1
    else:
        new_state_data = state_data

    return new_state_data


def get_error_file_mode(continue_training_from_epoch: Optional[int]) -> str:
    if continue_training_from_epoch is not None:
        error_file_mode = "a"
    else:
        error_file_mode = "w"

    return error_file_mode


def train_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_train_nohcp: torch.Tensor,
    y_train_nohcp: torch.Tensor,
    x_valid: torch.Tensor,
    y_valid: torch.Tensor,
    params: TrainingParameters,
    model_: RegressionMultilayerPerceptron,
    optimizer_: torch.optim.Optimizer,
    scheduler_: torch.optim.lr_scheduler._LRScheduler,
    modelpath: Path,
    loss_calculator: torch.Module,
    *,
    save_every: int,
    continue_training_from_epoch: Optional[int] = None,
) -> None:
    saved_models_dirpath = training_io.get_saved_models_dirpath(params, Path.cwd())

    # fmt: off
    error_file_mode = get_error_file_mode(continue_training_from_epoch)
    training_error_writer = training_utils.ErrorWriter(modelpath, "training_error_vs_epoch.dat", error_file_mode)
    training_nohcp_error_writer = training_utils.ErrorWriter(modelpath, "training_nohcp_error_vs_epoch.dat", error_file_mode)
    validation_error_writer = training_utils.ErrorWriter(modelpath, "validation_error_vs_epoch.dat", error_file_mode)
    # fmt: on

    training_loss_accumulator = training_utils.TrainingLossAccumulator()

    trainset = PotentialDataset(x_train, y_train)
    trainloader = DataLoader(trainset, batch_size=params.batch_size, num_workers=0, shuffle=True)

    state_data = training_state.TrainingStateData(model_, optimizer_, scheduler_, 0)
    state_data = update_training_state_data(state_data, saved_models_dirpath, continue_training_from_epoch)
    model, optimizer, scheduler, epoch_start = state_data.unpack()

    for i_epoch in range(epoch_start, params.total_epochs):
        for x_batch, y_batch in trainloader:
            y_batch_predicted: torch.Tensor = model(x_batch)

            loss: torch.Tensor = loss_calculator(y_batch, y_batch_predicted)
            loss.backward()

            training_loss_accumulator.accumulate(y_batch_predicted.shape[0], loss)

            optimizer.step()
            optimizer.zero_grad()

        if i_epoch >= 100:
            scheduler.step()

        # fmt: off
        epoch_training_loss = training_loss_accumulator.get_and_reset_total_loss()
        training_error_writer.append(i_epoch, epoch_training_loss)

        epoch_training_nohcp_loss = training_utils.evaluate_model_loss(model, loss_calculator, x_train_nohcp, y_train_nohcp)
        training_nohcp_error_writer.append(i_epoch, epoch_training_nohcp_loss)

        epoch_validation_loss = training_utils.evaluate_model_loss(model, loss_calculator, x_valid, y_valid)
        validation_error_writer.append(i_epoch, epoch_validation_loss)
        # fmt: on

        print(f"(epoch, training_loss)       = ({i_epoch}, {epoch_training_loss:.8f})")
        print(f"(epoch, training_nohcp_loss) = ({i_epoch}, {epoch_training_nohcp_loss:.8f})")
        print(f"(epoch, validation_loss)     = ({i_epoch}, {epoch_validation_loss:.8f})")

        if i_epoch % save_every == 0 and i_epoch != 0:
            state_dict = training_state.create_training_state_dict(model, optimizer, scheduler, i_epoch)
            training_state.save_training_state_dict(saved_models_dirpath, state_dict)

    state_dict = training_state.create_training_state_dict(model, optimizer, scheduler, params.total_epochs - 1)
    training_state.save_training_state_dict(saved_models_dirpath, state_dict)


def test_model(
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    model: RegressionMultilayerPerceptron,
    modelfile: Path,
) -> float:
    checkpoint = torch.load(modelfile)
    model.load_state_dict(checkpoint["model_state_dict"])
    loss_calculator = torch.nn.MSELoss()

    testing_loss = training_utils.evaluate_model_loss(model, loss_calculator, x_test, y_test)

    return testing_loss
