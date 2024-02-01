"""
Create plots of the error in the PES against the actual energy.

Samples with energies of small absolute value will have energies of small absolute value,
and will result in a lower mean squared error. Because we want the model to perform well
especially for samples of large energies, this makes the mean squared error a misleading
measure for the performance of the sample.
"""


# PLAN
# prepare the model (load state dictionary, etc.)
# read in the samples (x_data and y_data)
# calculate y_predicted using the model
# plot (abs(y_data - y_predicted) vs y_data)

from pathlib import Path
from typing import Tuple

import torch

import matplotlib.pyplot as plt
import numpy as np

from nn_fourbody_potential.dataio import load_fourbody_training_data
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.transformations.transformers import MinimumPermutationTransformer
from nn_fourbody_potential.transformations.transformers import ReciprocalTransformer
from nn_fourbody_potential.transformations.transformers import SixSideLengthsTransformer
from nn_fourbody_potential.transformations.transformers import StandardizeTransformer
from nn_fourbody_potential.transformations import transform_sidelengths_data

import model_info


def get_model(modelfile: Path, layer_sizes: list[int]) -> RegressionMultilayerPerceptron:
    # create the PyTorch model; the following parameters (input features, outputs, and the layer sizes)
    # are specific to the model that was trained;
    # so far, the weights have not been initialized
    n_features = 6
    n_outputs = 1
    model = RegressionMultilayerPerceptron(n_features, n_outputs, layer_sizes)

    # the path to the specific .pth file
    # 2999 corresponds to the very last batch
    # modelfile = Path(
    #    "models", "nnpes_pruned_layers64_128_128_64_lr_0.000200_datasize_5000", "models", "nnpes_02999.pth"
    # )

    # fill the weights of the model
    model.load_state_dict(torch.load(modelfile))

    # prepare the model for evaluation
    model.eval()

    return model


def feature_transformers() -> list[SixSideLengthsTransformer]:
    min_sidelen = 2.2
    max_sidelen = 4.5

    return [
        ReciprocalTransformer(),
        StandardizeTransformer((1.0 / max_sidelen, 1.0 / min_sidelen), (0.0, 1.0)),
        MinimumPermutationTransformer(),
    ]


def add_dummy_dimension(data: np.ndarray[float]) -> np.ndarray[float, float]:
    """Takes an array of shape (n,), and returns an array of shape (n, 1)"""
    assert len(data.shape) == 1

    return data.reshape(data.size, -1)


def prepared_data(data_filepath: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the sidelengths and energies from 'data_filepath', apply the transformations,
    and turn the sidelengths and energies into torch tensors.
    """
    sidelengths, energies = load_fourbody_training_data(data_filepath)
    transformers = feature_transformers()
    sidelengths = transform_sidelengths_data(sidelengths, transformers)
    energies = add_dummy_dimension(energies)

    x = torch.from_numpy(sidelengths.astype(np.float32))
    y = torch.from_numpy(energies.astype(np.float32))

    return x, y


def main() -> None:
    modelfile8 = Path(
        "models", "nnpes_small8_withfast_layers8_16_16_8_lr_0.000200_datasize_5000", "models", "nnpes_05999.pth"
    )
    modelfile16 = Path(
        "models", "nnpes_small16_withfast_layers16_32_32_16_lr_0.000200_datasize_5000", "models", "nnpes_05999.pth"
    )
    modelfile32 = Path(
        "models", "nnpes_small32_withfast_layers32_64_64_32_lr_0.000200_datasize_5000", "models", "nnpes_05999.pth"
    )
    model8 = get_model(modelfile8, [8, 16, 16, 8])
    model16 = get_model(modelfile16, [16, 32, 32, 16])
    model32 = get_model(modelfile32, [32, 64, 64, 32])

    training_data_filepath = model_info.get_training_data_filepath()
    fastdecay_training_data_filepath = model_info.get_fastdecay_training_data_filepath()
    veryfastdecay_training_data_filepath = model_info.get_veryfastdecay_training_data_filepath()

    x_slowdecay_train, y_slowdecay_train = prepared_data(training_data_filepath)
    x_fastdecay_train, y_fastdecay_train = prepared_data(fastdecay_training_data_filepath)
    x_veryfastdecay_train, y_veryfastdecay_train = prepared_data(veryfastdecay_training_data_filepath)

    x_train = torch.concatenate((x_slowdecay_train, x_fastdecay_train, x_veryfastdecay_train))
    y_train = torch.concatenate((y_slowdecay_train, y_fastdecay_train, y_veryfastdecay_train))

    with torch.no_grad():
        y_predicted8 = model8(x_train)
        y_predicted16 = model16(x_train)
        y_predicted32 = model32(x_train)

    #    energies_error = torch.abs(y_predicted - y_data).reshape(-1)
    #    energies = y_data.reshape(-1)

    fig, ax = plt.subplots()
    # ax.plot(energies, energies_error, lw=0, marker="o", ms=1.0)
    ax.plot(y_predicted8.reshape(-1), y_train.reshape(-1), lw=0, marker="o", ms=1.0)
    ax.plot(y_predicted16.reshape(-1), y_train.reshape(-1), lw=0, marker="o", ms=1.0)
    ax.plot(y_predicted32.reshape(-1), y_train.reshape(-1), lw=0, marker="o", ms=1.0)
    plt.show()


if __name__ == "__main__":
    main()
