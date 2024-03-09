from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

import torch
from torch.utils.data import Dataset

from nn_fourbody_potential.constants import N_FEATURES
from nn_fourbody_potential.constants import N_OUTPUTS
from nn_fourbody_potential.dataio import load_fourbody_training_data
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import transform_sidelengths_data


def add_dummy_dimension(data: NDArray) -> NDArray:
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
) -> tuple[torch.Tensor, torch.Tensor]:
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
        loss: torch.Tensor = loss_calculator(output_data, output_data_predicted)

    model.train()
    return loss.item()


class TrainingLossAccumulator:
    """Accumulate the loss over the different mini-batches of each epoch."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._total_accumulated_samples = 0
        self._total_accumulated_loss = 0.0

    def accumulate(self, minibatch_size: int, loss_value: torch.Tensor) -> None:
        self._total_accumulated_samples += minibatch_size
        total_batch_loss = minibatch_size * loss_value.item()
        self._total_accumulated_loss += total_batch_loss

    def get_and_reset_total_loss(self) -> float:
        average_loss = self._total_accumulated_loss / self._total_accumulated_samples
        self.reset()

        return average_loss


class ErrorWriter:
    """Saves the error as a function of the epoch."""

    def __init__(self, savepath: Path, filename: str, mode: str) -> None:
        if not savepath.exists():
            savepath.mkdir()

        self._savefile = Path(savepath, filename)

        if mode not in ["w", "a"]:
            raise ValueError("The only accepted writing modes for the error file are 'w' and 'a'.\n" f"Found: {mode}")

        if mode == "w":
            with open(self._savefile, "w") as fout:
                fout.write("[Epoch]   [Error]\n")

    def append(self, epoch: int, error: float) -> None:
        with open(self._savefile, "a") as fout:
            fout.write(f"{epoch: >7d} : {error: .14e}\n")


class PotentialDataset(Dataset):
    """
    Prepare the dataset of the mapping of a multidimensional feature space to a
    single output dimension.
    """

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index):
        x_output = self.x[index]
        y_output = self.y[index]
        return x_output, y_output
