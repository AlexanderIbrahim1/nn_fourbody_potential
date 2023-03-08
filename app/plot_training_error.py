"""
Plot the training error of a trained model as a function of the epoch.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import model_info


def plot_training_error_vs_epoch() -> None:
    training_data_filepath = model_info.get_training_data_filepath()
    data_transforms = model_info.get_data_transforms()
    params = model_info.get_training_parameters(training_data_filepath, data_transforms)
    model_dirpath = model_info.get_path_to_model(params)
    filename = "testing_error_vs_epoch.dat"

    epoch, training_mse = np.loadtxt(
        model_dirpath / filename, unpack=True, skiprows=1, delimiter=":"
    )
    training_rmse = np.sqrt(training_mse)

    fig, ax = plt.subplots()
    ax.plot(epoch, np.log(training_rmse))
    plt.show()


if __name__ == "__main__":
    plot_training_error_vs_epoch()
