"""
The 'modelio' subpackage contains code for managing the saving of models and other
data related to the models.
"""

from pathlib import Path

from nn_fourbody_potential.modelio.checkpoint_loader import CheckpointLoader
from nn_fourbody_potential.modelio.checkpoint_saver import CheckpointSaver
from nn_fourbody_potential.modelio.error_writer import ErrorWriter
from nn_fourbody_potential.modelio.model_saver import ModelSaver
from nn_fourbody_potential.modelio.utils import write_training_parameters

from nn_fourbody_potential.models import TrainingParameters


def model_directory_name(layers: list[int], learning_rate: float, training_size: int, other: str = "") -> Path:
    """The name of the directory where this set of models will be saved."""
    layer_part = "layers" + "_".join([str(layer) for layer in layers])
    lr_part = f"lr_{learning_rate:.6f}"
    training_size_part = f"datasize_{training_size}"

    return f"{other}_{layer_part}_{lr_part}_{training_size_part}"


def get_model_filename(savepath: Path, epoch: int) -> Path:
    return Path(savepath, f"nnpes_{epoch:0>5d}.pth")


def number_of_lines(file: Path) -> int:
    with open(file, "r") as fin:
        return sum([1 for _ in fin])


def get_path_to_model(params: TrainingParameters, base_dirpath: Path) -> Path:
    base_models_path = base_dirpath / "models"
    specific_model_dir = model_directory_name(
        params.layers, params.learning_rate, params.training_size, other=f"nnpes{params.other}"
    )

    return base_models_path / specific_model_dir


def get_training_parameters_filepath(params: TrainingParameters, base_dirpath: Path) -> Path:
    modelpath = get_path_to_model(params, base_dirpath)
    return modelpath / "training_parameters.dat"


def get_saved_models_dirpath(params: TrainingParameters, base_dirpath: Path) -> Path:
    modelpath = get_path_to_model(params, base_dirpath)
    return modelpath / "models"
