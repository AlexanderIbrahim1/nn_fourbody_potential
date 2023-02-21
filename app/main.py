from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from torchtyping import TensorType
from torchtyping import patch_typeguard
from typeguard import typechecked

from regression1d.models import RegressionMultilayerPerceptron
from regression1d.dataset import PotentialDataset

from dataio import load_fourbody_training_data

from generate_pes_data import generate_training_data
from generate_pes_data import get_distribution

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


def train_model(traindata_filename: Path, model: RegressionMultilayerPerceptron, modelfile: Path):
    seed = 0
    learning_rate = 5.0e-3
    n_epochs = 200
    batch_size = 1000

    np.random.seed(seed)

    sidelengths_train, energies_train = load_fourbody_training_data(traindata_filename)
    energies_train = add_dummy_dimension(energies_train)

    x_train = torch.from_numpy(sidelengths_train.astype(np.float32))
    y_train = torch.from_numpy(energies_train.astype(np.float32))

    check_training_data_sizes(x_train, y_train)
    
    trainset = PotentialDataset(x_train, y_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=1
    )

    loss_calculator = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for i_epoch in range(n_epochs):

        current_loss = 0.0
        for (i_batch, batch_data) in enumerate(trainloader):
            y_predicted = model(x_train)

            loss = loss_calculator(y_train, y_predicted)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            current_loss += loss.item()
            if i_epoch % 20 == 0 and i_epoch != 0:
                print(f"(epoch, loss) = ({i_epoch}, {current_loss:.4f})")
                current_loss = 0.0

    torch.save(model.state_dict(), modelfile)


def calculate_rmse(y_test: torch.Tensor, y_estim: torch.Tensor) -> float:
    assert y_test.shape == y_estim.shape

    n_samples = y_test.shape[0]

    total_square_error = np.sum([
        (y_test_val - y_estim_val)**2
        for (y_test_val, y_estim_val) in zip(y_test[:,0], y_estim[:,0])
    ])

    return np.sqrt(total_square_error / n_samples)


def test_model(model: RegressionMutlilayerPerceptron, modelfile: Path, n_samples):
    model.load_state_dict(torch.load(modelfile))
    model.eval()

    distrib = get_distribution()
    sidelengths_test, energies_test = generate_training_data(n_samples, distrib)
    energies_test = add_dummy_dimension(energies_test)

    with torch.no_grad():
        x_test = torch.from_numpy(sidelengths_test.astype(np.float32))
        y_test = torch.from_numpy(energies_test.astype(np.float32))
    
        y_estim = model(x_test)

        y_test.detach()
        y_estim.detach()
        rmse = calculate_rmse(y_test, y_estim)
        print(f"rmse = {rmse: .12f}")


if __name__ == "__main__":
    model = RegressionMultilayerPerceptron(N_FEATURES, N_OUTPUTS, [16, 32, 32, 16])
    modelfile = Path(".", "models", "nn_pes_model_16_32_32_16.pth")
    #traindata_filename = Path('.', 'data', 'training_data.dat')
    #train_model(traindata_filename, model, modelfile)

    test_model(model, modelfile, 50000)
