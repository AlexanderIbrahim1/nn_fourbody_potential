from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "..")

import numpy as np

from nn_fourbody_potential.modelio import ModelSaver
from nn_fourbody_potential.modelio.utils import get_model_filename
from nn_fourbody_potential.modelio import write_training_parameters
from nn_fourbody_potential.models import RegressionMultilayerPerceptron
from nn_fourbody_potential.models import TrainingParameters
from nn_fourbody_potential.transformations import SixSideLengthsTransformer

import model_info
import training


def get_training_parameters(
    data_filepath: Path,
    data_transforms: list[SixSideLengthsTransformer],
    other_info: str,
) -> TrainingParameters:
    return TrainingParameters(
        seed=0,
        layers=[64, 128, 128, 64],
        learning_rate=1.0e-4,
        weight_decay=1.0e-4,
        training_size=model_info.number_of_lines(data_filepath),
        total_epochs=6000,
        batch_size=2000,
        transformations=data_transforms,
        apply_batch_norm=False,
        other=other_info,
    )


def train(
    training_data_filepath: Path, testing_data_filepath: Path, validation_data_filepath: Path, other_info: str
) -> None:
    transforms = model_info.get_data_transforms()
    params = get_training_parameters(training_data_filepath, transforms, other_info)

    x_train, y_train = training.prepared_data(training_data_filepath, transforms)
    x_test, y_test = training.prepared_data(testing_data_filepath, transforms)
    x_valid, y_valid = training.prepared_data(validation_data_filepath, transforms)

    model = RegressionMultilayerPerceptron(training.N_FEATURES, training.N_OUTPUTS, params.layers)
    modelpath = model_info.get_path_to_model(params)
    if not modelpath.exists():
        modelpath.mkdir()

    training_parameters_filepath = model_info.get_training_parameters_filepath(params)

    saved_models_dirpath = model_info.get_saved_models_dirpath(params)

    if not training_parameters_filepath.exists():
        write_training_parameters(training_parameters_filepath, params, overwrite=False)

    training.train_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        params,
        model,
        modelpath,
        save_every=10,
        continue_training_from_epoch=30,
    )

    last_model_filename = get_model_filename(saved_models_dirpath, params.total_epochs - 1)
    test_loss = training.test_model(x_test, y_test, model, last_model_filename)
    print(f"test loss mse = {test_loss}")
    print(f"test loss rmse = {np.sqrt(test_loss)}")


# def train_model(
#     x_train: torch.Tensor,
#     y_train: torch.Tensor,
#     x_valid: torch.Tensor,
#     y_valid: torch.Tensor,
#     params: TrainingParameters,
#     default_model: RegressionMultilayerPerceptron,
#     modelpath: Path,
#     *,
#     save_every: int,
#     continue_training_from_epoch: Optional[int] = None,
# ) -> None:


if __name__ == "__main__":
    training_data_filepath = Path("data", "all_energy_train.dat")
    testing_data_filepath = Path("data", "all_energy_test.dat")
    validation_data_filepath = Path("data", "all_energy_valid.dat")
    other_info = "_all_energies_with_continue"

    train(training_data_filepath, testing_data_filepath, validation_data_filepath, other_info)
