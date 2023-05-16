"""
Plot the training and validation error of a trained model as a function of the epoch.
"""

import matplotlib.pyplot as plt
import numpy as np

import model_info


def plot_error_vs_epoch() -> None:
    training_data_filepath = model_info.get_training_data_filepath()
    validation_data_filepath = model_info.get_validation_data_filepath()

    data_transforms = model_info.get_data_transforms()
    params = model_info.get_training_parameters(training_data_filepath, data_transforms)

    model_dirpath = model_info.get_path_to_model(params)
    train_filename = "training_error_vs_epoch.dat"
    valid_filename = "validation_error_vs_epoch.dat"

    epoch, train_mse = np.loadtxt(model_dirpath / train_filename, unpack=True, skiprows=1, delimiter=":")
    epoch, valid_mse = np.loadtxt(model_dirpath / valid_filename, unpack=True, skiprows=1, delimiter=":")

    fig, ax = plt.subplots()
    ax.set_xlabel("epoch", fontsize=18)
    ax.set_ylabel("ln(mse loss)", fontsize=18)
    ax.plot(epoch, np.log(train_mse), label="training")
    ax.plot(epoch, np.log(valid_mse), label="validation")

    ax.legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_error_vs_epoch()
