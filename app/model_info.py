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
        return sum([1 for _ in fin])


def get_training_data_filepath() -> Path:
    return Path(".", "data", "abinitio_training_data_5000_2.2_4.5.dat")


def get_hcp_data_filepath() -> Path:
    return Path(".", "data", "abinitio_hcp_data_3901_2.2_4.5.dat")


def get_validation_data_filepath() -> Path:
    return Path(".", "data", "abinitio_validation_data_2000_2.2_4.5.dat")


def get_testing_data_filepath() -> Path:
    return Path(".", "data", "abinitio_testing_data_2000_2.2_4.5.dat")


def get_data_transforms() -> list[SixSideLengthsTransformer]:
    min_sidelen = 2.2
    max_sidelen = 4.5

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
        learning_rate=2.0e-4,
        weight_decay=1.0e-4,
        training_size=number_of_lines(training_data_filepath),
        total_epochs=3000,
        batch_size=2000,
        transformations=data_transforms,
        apply_batch_norm=False,
        other="_pruned",
    )


def get_path_to_model(params: TrainingParameters) -> Path:
    base_models_path = Path.cwd() / "models"
    specific_model_dir = model_directory_name(
        params.layers, params.learning_rate, params.training_size, other=f"nnpes{params.other}"
    )

    return base_models_path / specific_model_dir


def get_training_parameters_filepath(params: TrainingParameters) -> Path:
    modelpath = get_path_to_model(params)
    return modelpath / "training_parameters.dat"


def get_saved_models_dirpath(params: TrainingParameters) -> Path:
    modelpath = get_path_to_model(params)
    return modelpath / "models"
