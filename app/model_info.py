"""
This module contains the information needed to create the NN model. This information is used in
several other modules, such as 'main.py' (to train and test the model) and 'plot_model.py' (to 
plot the energies of the model for a given path in coordinate space), and others too.
"""

from pathlib import Path
from nn_fourbody_potential.modelio import model_directory_name

from nn_fourbody_potential.models import TrainingParameters

from nn_fourbody_potential.transformations import SixSideLengthsTransformer
from nn_fourbody_potential.transformations import ReciprocalTransformer
from nn_fourbody_potential.transformations import MinimumPermutationTransformer
from nn_fourbody_potential.transformations import StandardizeTransformer


def number_of_lines(file: Path) -> int:
    with open(file, "r") as fin:
        counter = 0
        for _ in fin:
            counter += 1

        return counter


def get_training_data_filepath() -> Path:
    return Path(".", "data", "training_data_5000_2.2_5.0.dat")


def get_data_transforms() -> list[SixSideLengthsTransformer]:
    min_sidelen = 2.2
    max_sidelen = 5.0

    return [
        ReciprocalTransformer(),
        StandardizeTransformer((1.0 / max_sidelen, 1.0 / min_sidelen), (0.0, 1.0)),
        MinimumPermutationTransformer(),
    ]


def get_training_parameters(
    training_data_filepath: Path, data_transforms: list[SixSideLengthsTransformer]
) -> TrainingParameters:
    return TrainingParameters(
        seed=0,
        layers=[64, 128, 128, 64],
        learning_rate=5.0e-3,
        weight_decay=1.0e-4,
        training_size=number_of_lines(training_data_filepath),
        total_epochs=100,
        batch_size=1000,
        transformations=data_transforms,
        other="",
    )


def get_path_to_model(params: TrainingParameters) -> Path:
    base_models_path = Path.cwd() / "models"
    specific_model_dir = model_directory_name(
        params.layers, params.learning_rate, params.training_size, other="nnpes"
    )

    return base_models_path / specific_model_dir


def get_training_parameters_filepath(params: TrainingParameters) -> Path:
    modelpath = get_path_to_model(params)
    return modelpath / "training_parameters.dat"


def get_saved_models_dirpath(params: TrainingParameters) -> Path:
    modelpath = get_path_to_model(params)
    return modelpath / "models"


if __name__ == "__main__":

    model = RegressionMultilayerPerceptron(N_FEATURES, N_OUTPUTS, params.layers)

    modelpath = get_path_to_model(params)
    if not modelpath.exists():
        modelpath.mkdir()

    training_parameters_filepath = get_training_parameters_filepath(params)
    write_training_parameters(training_parameters_filepath, params, overwrite=False)

    saved_models_filepath = get_saved_models_dirpath(params)
    model_saver = ModelSaver(saved_models_filepath)

    train_model(traindata_filename, params, model, modelpath, model_saver, 20)
    test_model(
        model,
        model_saver.get_model_filename(params.total_epochs - 1),
        50000,
        data_transforms,
    )
