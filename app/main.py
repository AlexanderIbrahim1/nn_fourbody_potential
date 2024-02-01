from __future__ import annotations

import torch
import numpy as np

from nn_fourbody_potential.modelio import ModelSaver
from nn_fourbody_potential.modelio import write_training_parameters
from nn_fourbody_potential.models import RegressionMultilayerPerceptron

import model_info
import training
import prune


def train_with_slow_decay_data() -> None:
    training_data_filepath = model_info.get_training_data_filepath()
    hcp_data_filepath = model_info.get_hcp_data_filepath()
    validation_data_filepath = model_info.get_validation_data_filepath()
    testing_data_filepath = model_info.get_testing_data_filepath()

    transforms = model_info.get_data_transforms()
    params = model_info.get_training_parameters(training_data_filepath, transforms)

    x_train_hcp, y_train_hcp = training.prepared_data(hcp_data_filepath, transforms)
    x_train_gen, y_train_gen = training.prepared_data(training_data_filepath, transforms)
    x_valid, y_valid = training.prepared_data(validation_data_filepath, transforms)
    x_test, y_test = training.prepared_data(testing_data_filepath, transforms)

    hcp_mask = torch.Tensor(
        [
            prune.sidelengths_filter(sidelens, 4.2) and prune.energy_filter(energy, 1.0e-3)
            for (sidelens, energy) in zip(x_train_hcp, y_train_hcp)
        ]
    )
    x_train_hcp = x_train_hcp[torch.nonzero(hcp_mask).reshape(-1)]
    y_train_hcp = y_train_hcp[torch.nonzero(hcp_mask).reshape(-1)]

    x_train = torch.concatenate((x_train_gen, x_train_hcp))
    y_train = torch.concatenate((y_train_gen, y_train_hcp))

    model = RegressionMultilayerPerceptron(training.N_FEATURES, training.N_OUTPUTS, params.layers)

    modelpath = model_info.get_path_to_model(params)
    if not modelpath.exists():
        modelpath.mkdir()

    training_parameters_filepath = model_info.get_training_parameters_filepath(params)

    saved_models_dirpath = model_info.get_saved_models_dirpath(params)
    model_saver = ModelSaver(saved_models_dirpath)
    last_model_filename = model_saver.get_model_filename(params.total_epochs - 1)

    write_training_parameters(training_parameters_filepath, params, overwrite=False)
    training.train_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        params,
        model,
        modelpath,
        model_saver,
        save_every=20,
    )

    test_loss = training.test_model(x_test, y_test, model, last_model_filename)
    print(f"test loss mse = {test_loss}")
    print(f"test loss rmse = {np.sqrt(test_loss)}")


def train_with_fast_decay_data() -> None:
    training_data_filepath = model_info.get_training_data_filepath()
    hcp_data_filepath = model_info.get_hcp_data_filepath()
    validation_data_filepath = model_info.get_validation_data_filepath()
    testing_data_filepath = model_info.get_testing_data_filepath()

    fastdecay_training_data_filepath = model_info.get_fastdecay_training_data_filepath()
    fastdecay_testing_data_filepath = model_info.get_fastdecay_testing_data_filepath()
    fastdecay_validation_data_filepath = model_info.get_fastdecay_validation_data_filepath()
    veryfastdecay_training_data_filepath = model_info.get_veryfastdecay_training_data_filepath()
    veryfastdecay_testing_data_filepath = model_info.get_veryfastdecay_testing_data_filepath()
    veryfastdecay_validation_data_filepath = model_info.get_veryfastdecay_validation_data_filepath()

    transforms = model_info.get_data_transforms()
    params = model_info.get_training_parameters(training_data_filepath, transforms)

    x_train_hcp, y_train_hcp = training.prepared_data(hcp_data_filepath, transforms)
    x_train_gen, y_train_gen = training.prepared_data(training_data_filepath, transforms)
    x_valid, y_valid = training.prepared_data(validation_data_filepath, transforms)
    x_test, y_test = training.prepared_data(testing_data_filepath, transforms)
    x_fastdecay_test, y_fastdecay_test = training.prepared_data(fastdecay_testing_data_filepath, transforms)
    x_fastdecay_train, y_fastdecay_train = training.prepared_data(fastdecay_training_data_filepath, transforms)
    x_fastdecay_valid, y_fastdecay_valid = training.prepared_data(fastdecay_validation_data_filepath, transforms)
    x_veryfastdecay_test, y_veryfastdecay_test = training.prepared_data(veryfastdecay_testing_data_filepath, transforms)
    x_veryfastdecay_train, y_veryfastdecay_train = training.prepared_data(
        veryfastdecay_training_data_filepath, transforms
    )
    x_veryfastdecay_valid, y_veryfastdecay_valid = training.prepared_data(
        veryfastdecay_validation_data_filepath, transforms
    )

    # hcp_mask = torch.Tensor(
    #     [
    #         prune.sidelengths_filter(sidelens, 4.2) and prune.energy_filter(energy, 1.0e-3)
    #         for (sidelens, energy) in zip(x_train_hcp, y_train_hcp)
    #     ]
    # )
    # x_train_hcp = x_train_hcp[torch.nonzero(hcp_mask).reshape(-1)]
    # y_train_hcp = y_train_hcp[torch.nonzero(hcp_mask).reshape(-1)]

    x_train = torch.concatenate((x_train_gen, x_train_hcp, x_fastdecay_train, x_veryfastdecay_train))
    y_train = torch.concatenate((y_train_gen, y_train_hcp, y_fastdecay_train, y_veryfastdecay_train))
    x_valid = torch.concatenate((x_valid, x_fastdecay_valid, x_veryfastdecay_valid))
    y_valid = torch.concatenate((y_valid, y_fastdecay_valid, y_veryfastdecay_valid))
    x_test = torch.concatenate((x_test, x_fastdecay_test, x_veryfastdecay_test))
    y_test = torch.concatenate((y_test, y_fastdecay_test, y_veryfastdecay_test))

    model = RegressionMultilayerPerceptron(training.N_FEATURES, training.N_OUTPUTS, params.layers)

    modelpath = model_info.get_path_to_model(params)
    if not modelpath.exists():
        modelpath.mkdir()

    training_parameters_filepath = model_info.get_training_parameters_filepath(params)

    saved_models_dirpath = model_info.get_saved_models_dirpath(params)
    model_saver = ModelSaver(saved_models_dirpath)

    write_training_parameters(training_parameters_filepath, params, overwrite=False)
    training.train_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        params,
        model,
        modelpath,
        model_saver,
        save_every=20,
    )

    last_model_filename = model_saver.get_model_filename(params.total_epochs - 1)
    test_loss = training.test_model(x_test, y_test, model, last_model_filename)
    print(f"test loss mse = {test_loss}")
    print(f"test loss rmse = {np.sqrt(test_loss)}")


if __name__ == "__main__":
    train_with_fast_decay_data()
